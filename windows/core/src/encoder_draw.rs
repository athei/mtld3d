//! Dirty draw state captured directly into the frame wire buffer.
//!
//! Canonical state and shader-source leaves borrow the retained command arena.
//! Immutable aggregate roots use native scratch; both owners remain live
//! through submit replay. Borrowed addresses never transfer allocation ownership.

use std::ptr::NonNull;

use mtld3d_shared::{VertexAttrDesc, encoder_wire::WireError};

use crate::{
    draw_data::{
        AttrSnapshot, CurrentSnapshot, CurrentSnapshotPtr, DeclarationHeader, DepthStencilFlags,
        DrawOp, PsSource, PsSourceView, RenderStatePtr, RenderStateSnapshot, ScratchSlice,
        StageBinding, StageBindingsPtr, VsSource, VsSourceView,
    },
    dxso::VariantKey,
    scratch::ScratchArena,
};

mod shader_record;

#[cfg(test)]
mod tests;

/// Cursor over already initialized canonical snapshot records.
struct SnapshotReader<'a> {
    remaining: &'a [u8],
}
impl<'a> SnapshotReader<'a> {
    fn bytes(&mut self, count: u32) -> Result<&'a [u8], WireError> {
        let (value, tail) = self
            .remaining
            .split_at_checked(count as usize)
            .ok_or(WireError::Truncated)?;
        self.remaining = tail;
        Ok(value)
    }
}

/// Exclusive reservation cursor; it has no scalar encoding operations.
struct SnapshotWriter<'a> {
    destination: &'a mut [u8],
    used: usize,
}
impl SnapshotWriter<'_> {
    fn reserve_bytes(&mut self, count: usize) -> Result<&mut [u8], WireError> {
        let end = self.used.checked_add(count).ok_or(WireError::TooLarge)?;
        let value = self
            .destination
            .get_mut(self.used..end)
            .ok_or(WireError::TooLarge)?;
        self.used = end;
        Ok(value)
    }
    fn bytes(&mut self, value: &[u8]) -> Result<(), WireError> {
        self.reserve_bytes(value.len())?.copy_from_slice(value);
        Ok(())
    }
}

#[repr(C, align(8))]
struct SnapshotHeader {
    changed: u32,
    reserved: u32,
}
#[repr(C, align(8))]
struct StageHeader {
    mask: u16,
    count: u16,
    reserved: u32,
}
#[repr(C, align(8))]
struct VariantRecord {
    key: VariantKey,
    reserved: [u8; 2],
}
#[repr(C, align(8))]
struct ByteBindingRecord {
    address: u64,
    length: u32,
    present: u32,
}
#[repr(C, align(8))]
struct DepthFlagsRecord {
    flags: u32,
    reserved: u32,
}

// SAFETY: all fields are padding-free initialized integer or transparent bitflag records.
unsafe impl crate::encoder_records::CommandRecord for SnapshotHeader {}
// SAFETY: every field is an initialized integer and all eight bytes are occupied.
unsafe impl crate::encoder_records::CommandRecord for StageHeader {}
// SAFETY: canonical VariantKey has no invalid bit patterns or implicit padding.
unsafe impl crate::encoder_records::CommandRecord for VariantRecord {}
// SAFETY: every byte belongs to an integer field with every bit pattern valid.
unsafe impl crate::encoder_records::CommandRecord for ByteBindingRecord {}
// SAFETY: every byte belongs to an integer field with every bit pattern valid.
unsafe impl crate::encoder_records::CommandRecord for DepthFlagsRecord {}

fn capture_record<T: crate::encoder_records::CommandRecord>(
    writer: &mut SnapshotWriter<'_>,
    value: T,
) -> Result<(), WireError> {
    align_snapshot_leaf(writer)?;
    crate::encoder_records::write(writer.reserve_bytes(size_of::<T>())?, value)
}
fn borrow_record<'a, T: crate::encoder_records::CommandRecord>(
    reader: &mut SnapshotReader<'a>,
) -> Result<&'a T, WireError> {
    crate::encoder_records::borrow(read_aligned_bytes(
        reader,
        u32::try_from(size_of::<T>()).map_err(|_| WireError::TooLarge)?,
        8,
    )?)
}

/// Reserved upper bound for a delta with every structural field and byte binding.
///
/// The full-width declaration, stage and fixed-function fixture pins this limit.
pub const SNAPSHOT_DELTA_MAX_BYTES: usize = 4096;

/// API-owned shader keys needed to build subsequent dirty state.
///
/// Structural snapshots and uniform byte bindings are owned by the native decoder.
pub struct ApiSnapshotCache {
    pub vs: Option<VsSource>,
    pub ps: Option<PsSource>,
    pub variant: Option<VariantKey>,
    pub depth_stencil: DepthStencilFlags,
}

impl ApiSnapshotCache {
    pub const EMPTY: Self = Self {
        vs: None,
        ps: None,
        variant: None,
        depth_stencil: DepthStencilFlags::empty(),
    };
}

/// Borrowed declaration data emitted once when the declaration becomes dirty.
pub struct SnapshotAttributes<'a> {
    pub attrs: &'a [VertexAttrDesc],
    pub extents: &'a [u32; 16],
    pub used_streams: u16,
    pub vdecl_hash: u64,
}

/// Changes built by the API's existing dirty-state gates.
///
/// Byte bindings use outer `None` for unchanged and `Some(None)` for cleared.
/// Their order is VS constants, PS constants, alpha, fog, bump environment,
/// VS integer, VS boolean, PS integer, PS boolean, and per-draw VS uniforms.
#[derive(Default)]
pub struct SnapshotDelta<'a> {
    pub render_state: Option<&'a RenderStateSnapshot>,
    pub stages: Option<(u16, &'a [StageBinding])>,
    pub attrs: Option<SnapshotAttributes<'a>>,
    pub vs: Option<VsSourceView<'a>>,
    pub ps: Option<PsSourceView<'a>>,
    pub variant: Option<VariantKey>,
    pub bytes: [Option<Option<ScratchSlice>>; 10],
    pub depth_stencil: Option<DepthStencilFlags>,
}

/// Wire capture context for one frame. Failed writes poison subsequent records.
#[derive(Default)]
pub struct DrawWriter {
    poisoned: bool,
}

impl DrawWriter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn clear(&mut self) {
        self.poisoned = false;
    }

    /// Capture only changed state, directly from its API builder output.
    ///
    /// # Errors
    /// Returns malformed-field, size, allocation, or previous capture errors.
    pub fn capture_snapshot(
        &mut self,
        delta: &SnapshotDelta<'_>,
        destination: &mut [u8],
    ) -> Result<usize, WireError> {
        if self.poisoned {
            return Err(WireError::InvalidValue);
        }
        let mut writer = SnapshotWriter {
            destination,
            used: 0,
        };
        let result = write_snapshot_delta(delta, &mut writer).map(|()| writer.used);
        self.poisoned = result.is_err();
        result
    }
}

fn write_snapshot_delta(
    delta: &SnapshotDelta<'_>,
    writer: &mut SnapshotWriter<'_>,
) -> Result<(), WireError> {
    let mut mask = u32::from(delta.render_state.is_some())
        | (u32::from(delta.stages.is_some()) << 1)
        | (u32::from(delta.attrs.is_some()) << 2)
        | (u32::from(delta.vs.is_some()) << 3)
        | (u32::from(delta.ps.is_some()) << 4)
        | (u32::from(delta.variant.is_some()) << 5)
        | (u32::from(delta.depth_stencil.is_some()) << 16);
    for (index, value) in delta.bytes.iter().enumerate() {
        mask |= u32::from(value.is_some()) << (index + 6);
    }
    capture_record(
        writer,
        SnapshotHeader {
            changed: mask,
            reserved: 0,
        },
    )?;
    if let Some(value) = delta.render_state {
        if value.reserved != 0 {
            return Err(WireError::InvalidValue);
        }
        align_snapshot_leaf(writer)?;
        // SAFETY: the canonical C layout is pinned below with no implicit padding.
        // All fields, including its explicit reserved byte, are initialized scalars.
        let bytes =
            unsafe { core::slice::from_raw_parts(core::ptr::from_ref(value).cast::<u8>(), 60) };
        writer.bytes(bytes)?;
    }
    if let Some((mask, values)) = delta.stages {
        if mask.count_ones() as usize != values.len() {
            return Err(WireError::InvalidValue);
        }
        capture_record(
            writer,
            StageHeader {
                mask,
                count: u16::try_from(values.len()).map_err(|_| WireError::TooLarge)?,
                reserved: 0,
            },
        )?;
        // SAFETY: StageBinding has a pinned, padding-free canonical layout. The
        // borrowed slice contains initialized IDs and scalar sampler fields only.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                core::mem::size_of_val(values),
            )
        };
        writer.bytes(bytes)?;
    }
    if let Some(value) = &delta.attrs {
        if value.attrs.len() > 16 {
            return Err(WireError::InvalidValue);
        }
        capture_record(
            writer,
            DeclarationHeader {
                vdecl_hash: value.vdecl_hash,
                extents: *value.extents,
                count: u32::try_from(value.attrs.len()).map_err(|_| WireError::TooLarge)?,
                used_streams: value.used_streams,
                reserved: 0,
            },
        )?;
        // SAFETY: VertexAttrDesc has four initialized 32-bit fields and no padding.
        // Its enum field is already valid because capture accepts typed descriptors.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                value.attrs.as_ptr().cast::<u8>(),
                core::mem::size_of_val(value.attrs),
            )
        };
        writer.bytes(bytes)?;
    }
    if let Some(value) = delta.vs {
        shader_record::write_vs(value, writer)?;
    }
    if let Some(value) = delta.ps {
        shader_record::write_ps(value, writer)?;
    }
    if let Some(value) = delta.variant {
        capture_record(
            writer,
            VariantRecord {
                key: value,
                reserved: [0; 2],
            },
        )?;
    }
    for value in delta.bytes.iter().flatten() {
        write_optional_bytes(*value, writer)?;
    }
    if let Some(value) = delta.depth_stencil {
        capture_record(
            writer,
            DepthFlagsRecord {
                flags: u32::from(value.bits()),
                reserved: 0,
            },
        )?;
    }
    Ok(())
}

/// Keep canonical leaves aligned inside the reserved command payload.
fn align_snapshot_leaf(writer: &mut SnapshotWriter<'_>) -> Result<(), WireError> {
    let address = writer.reserve_bytes(0)?.as_ptr() as usize;
    let padding = address.wrapping_neg() & 7;
    writer.reserve_bytes(padding)?.fill(0);
    Ok(())
}

// These assertions run on every supported PE and Unix target. The complete field
// offsets rule out implicit padding before exposing canonical values as bytes.
const _: () = {
    use core::mem::{align_of, offset_of, size_of};

    use crate::{
        depth_stencil_state::{DepthStencilSnapshot, StencilFaceState},
        draw_data::DepthScissorFlags,
        pipeline_state::{PipelineRsBits, PipelineRsFlags},
    };
    assert!(cfg!(target_endian = "little"));
    assert!(size_of::<crate::ids::TextureId>() == 8);
    assert!(size_of::<PipelineRsFlags>() == 1);
    assert!(align_of::<PipelineRsFlags>() == 1);
    assert!(size_of::<DepthScissorFlags>() == 1);
    assert!(align_of::<DepthScissorFlags>() == 1);
    assert!(size_of::<PipelineRsBits>() == 11);
    assert!(align_of::<PipelineRsBits>() == 1);
    assert!(offset_of!(PipelineRsBits, flags) == 0);
    assert!(offset_of!(PipelineRsBits, src_blend) == 1);
    assert!(offset_of!(PipelineRsBits, dst_blend) == 2);
    assert!(offset_of!(PipelineRsBits, blend_op) == 3);
    assert!(offset_of!(PipelineRsBits, src_blend_alpha) == 4);
    assert!(offset_of!(PipelineRsBits, dst_blend_alpha) == 5);
    assert!(offset_of!(PipelineRsBits, blend_op_alpha) == 6);
    assert!(offset_of!(PipelineRsBits, color_write_mask) == 7);
    assert!(offset_of!(PipelineRsBits, color_write_mask_ext) == 8);
    assert!(size_of::<StencilFaceState>() == 4);
    assert!(align_of::<StencilFaceState>() == 1);
    assert!(offset_of!(StencilFaceState, func) == 0);
    assert!(offset_of!(StencilFaceState, fail_op) == 1);
    assert!(offset_of!(StencilFaceState, depth_fail_op) == 2);
    assert!(offset_of!(StencilFaceState, pass_op) == 3);
    assert!(size_of::<DepthStencilSnapshot>() == 20);
    assert!(align_of::<DepthStencilSnapshot>() == 4);
    assert!(offset_of!(DepthStencilSnapshot, depth_enable) == 0);
    assert!(offset_of!(DepthStencilSnapshot, depth_write) == 1);
    assert!(offset_of!(DepthStencilSnapshot, depth_func) == 2);
    assert!(offset_of!(DepthStencilSnapshot, stencil_enable) == 3);
    assert!(offset_of!(DepthStencilSnapshot, front) == 4);
    assert!(offset_of!(DepthStencilSnapshot, back) == 8);
    assert!(offset_of!(DepthStencilSnapshot, read_mask) == 12);
    assert!(offset_of!(DepthStencilSnapshot, write_mask) == 16);
    assert!(size_of::<RenderStateSnapshot>() == 60);
    assert!(align_of::<RenderStateSnapshot>() == 4);
    assert!(offset_of!(RenderStateSnapshot, pipeline_rs) == 0);
    assert!(offset_of!(RenderStateSnapshot, depth_scissor) == 11);
    assert!(offset_of!(RenderStateSnapshot, depth_stencil_state) == 12);
    assert!(offset_of!(RenderStateSnapshot, cull_mode) == 32);
    assert!(offset_of!(RenderStateSnapshot, fill_mode) == 33);
    assert!(offset_of!(RenderStateSnapshot, sample_mask) == 34);
    assert!(offset_of!(RenderStateSnapshot, reserved) == 35);
    assert!(offset_of!(RenderStateSnapshot, scissor_rect) == 36);
    assert!(offset_of!(RenderStateSnapshot, blend_factor) == 44);
    assert!(offset_of!(RenderStateSnapshot, depth_bias) == 48);
    assert!(offset_of!(RenderStateSnapshot, slope_scale_depth_bias) == 52);
    assert!(offset_of!(RenderStateSnapshot, stencil_ref) == 56);
    assert!(size_of::<StageBinding>() == 64);
    assert!(align_of::<StageBinding>() == 8);
    assert!(offset_of!(StageBinding, texture_id) == 0);
    assert!(offset_of!(StageBinding, sampler_state) == 8);
    assert!(size_of::<DeclarationHeader>() == 80);
    assert!(align_of::<DeclarationHeader>() == 8);
    assert!(offset_of!(DeclarationHeader, vdecl_hash) == 0);
    assert!(offset_of!(DeclarationHeader, extents) == 8);
    assert!(offset_of!(DeclarationHeader, count) == 72);
    assert!(offset_of!(DeclarationHeader, used_streams) == 76);
    assert!(offset_of!(DeclarationHeader, reserved) == 78);
    assert!(size_of::<SnapshotHeader>() == 8);
    assert!(offset_of!(SnapshotHeader, changed) == 0);
    assert!(offset_of!(SnapshotHeader, reserved) == 4);
    assert!(size_of::<StageHeader>() == 8);
    assert!(offset_of!(StageHeader, mask) == 0);
    assert!(offset_of!(StageHeader, count) == 2);
    assert!(offset_of!(StageHeader, reserved) == 4);
    assert!(size_of::<VariantRecord>() == 24);
    assert!(offset_of!(VariantRecord, key) == 0);
    assert!(offset_of!(VariantRecord, reserved) == 22);
    assert!(size_of::<ByteBindingRecord>() == 16);
    assert!(offset_of!(ByteBindingRecord, address) == 0);
    assert!(offset_of!(ByteBindingRecord, length) == 8);
    assert!(offset_of!(ByteBindingRecord, present) == 12);
    assert!(size_of::<DepthFlagsRecord>() == 8);
    assert!(offset_of!(DepthFlagsRecord, flags) == 0);
    assert!(offset_of!(DepthFlagsRecord, reserved) == 4);
    assert!(size_of::<VertexAttrDesc>() == 16);
    assert!(align_of::<VertexAttrDesc>() == 4);
    assert!(offset_of!(VertexAttrDesc, attr_index) == 0);
    assert!(offset_of!(VertexAttrDesc, buffer_index) == 4);
    assert!(offset_of!(VertexAttrDesc, format) == 12);
    assert!(offset_of!(VertexAttrDesc, offset) == 8);
};

/// Native decoder for one retained frame lease.
///
/// Decoded tokens borrow native scratch and leased PE byte ranges through replay.
pub struct DrawReader {
    current: CurrentSnapshot,
    poisoned: bool,
}

impl DrawReader {
    /// Establish the immutable backing contract for decoded byte ranges.
    ///
    /// # Safety
    /// Every command byte slice and nonempty byte range must name initialized,
    /// immutable storage retained through submit replay. Keep every arena alive
    /// and unchanged until all returned tokens and their copies are forgotten.
    /// The contract applies to every frame after `clear` as well.
    #[must_use]
    pub const unsafe fn new() -> Self {
        Self {
            current: CurrentSnapshot::EMPTY,
            poisoned: false,
        }
    }

    pub const fn clear(&mut self) {
        self.current = CurrentSnapshot::EMPTY;
        self.poisoned = false;
    }

    /// Apply changed canonical records and retain an immutable native snapshot root.
    ///
    /// # Safety
    /// The paired typed producer must construct every canonical record with valid fields.
    /// Keep that initialized payload and every referenced byte range immutable
    /// and allocated through every returned token use, including submit replay.
    ///
    /// # Errors
    /// Returns a truncated, malformed, or previously poisoned capture error.
    pub unsafe fn decode_snapshot(
        &mut self,
        payload: &[u8],
        scratch: &mut ScratchArena,
    ) -> Result<CurrentSnapshotPtr, WireError> {
        if self.poisoned {
            return Err(WireError::InvalidValue);
        }
        let mut reader = SnapshotReader { remaining: payload };
        let result = self
            .read_snapshot_delta(&mut reader, scratch)
            .and_then(|snapshot| {
                if reader.remaining.is_empty() {
                    Ok(snapshot)
                } else {
                    Err(WireError::InvalidValue)
                }
            });
        self.poisoned = result.is_err();
        result
    }

    fn read_snapshot_delta(
        &mut self,
        reader: &mut SnapshotReader<'_>,
        scratch: &mut ScratchArena,
    ) -> Result<CurrentSnapshotPtr, WireError> {
        let mask = read_snapshot_mask(reader)?;
        if mask & 1 != 0 {
            let ptr = read_borrowed_render_state(reader)?;
            // SAFETY: the admitted immutable command arena retains this canonical leaf.
            self.current.render_state = Some(unsafe { RenderStatePtr::new(ptr) });
        }
        if mask & 2 != 0 {
            self.current.stage_bindings = Some(read_borrowed_stages(reader)?);
        }
        if mask & 4 != 0 {
            self.current.attrs = Some(read_borrowed_attrs(reader)?);
        }
        if mask & 8 != 0 {
            self.current.vs = Some(shader_record::read_vs(reader)?);
        }
        if mask & 16 != 0 {
            self.current.ps = Some(shader_record::read_ps(reader)?);
        }
        if mask & 32 != 0 {
            self.current.variant = Some(read_variant(reader)?);
        }
        let byte_fields = [
            &mut self.current.vs_constants,
            &mut self.current.ps_constants,
            &mut self.current.alpha_ref_bytes,
            &mut self.current.fog_color_bytes,
            &mut self.current.bump_env_bytes,
            &mut self.current.vs_int_const_bytes,
            &mut self.current.vs_bool_const_bytes,
            &mut self.current.ps_int_const_bytes,
            &mut self.current.ps_bool_const_bytes,
            &mut self.current.vs_draw_bytes,
        ];
        for (index, value) in byte_fields.into_iter().enumerate() {
            if mask & (1 << (index + 6)) != 0 {
                *value = read_optional_bytes(reader)?;
            }
        }
        if mask & (1 << 16) != 0 {
            self.current.depth_stencil = read_depth_flags(reader)?;
        }
        // SAFETY: CurrentSnapshot contains only trivial-Drop tokens and scalar values;
        // native scratch and every borrowed PE range remain retained through replay.
        let ptr = unsafe { scratch.alloc_from(&self.current) };
        let ptr = NonNull::new(ptr).ok_or(WireError::InvalidValue)?;
        // SAFETY: the initialized snapshot and its referents remain live through replay.
        Ok(unsafe { CurrentSnapshotPtr::new(ptr) })
    }
}

fn read_snapshot_mask(reader: &mut SnapshotReader<'_>) -> Result<u32, WireError> {
    let header: &SnapshotHeader = borrow_record(reader)?;
    if header.changed & !0x1ffff != 0 || header.reserved != 0 {
        return Err(WireError::InvalidValue);
    }
    Ok(header.changed)
}

// Inline canonical leaves are aligned relative to the command arena, not a Rust enum.
fn read_aligned_bytes<'a>(
    reader: &mut SnapshotReader<'a>,
    count: u32,
    alignment: usize,
) -> Result<&'a [u8], WireError> {
    let address = reader.bytes(0)?.as_ptr() as usize;
    let padding = address.wrapping_neg() & (alignment - 1);
    if reader
        .bytes(u32::try_from(padding).map_err(|_| WireError::TooLarge)?)?
        .iter()
        .any(|&byte| byte != 0)
    {
        return Err(WireError::InvalidValue);
    }
    reader.bytes(count)
}

fn read_borrowed_render_state(
    reader: &mut SnapshotReader<'_>,
) -> Result<NonNull<RenderStateSnapshot>, WireError> {
    let bytes = read_aligned_bytes(
        reader,
        u32::try_from(size_of::<RenderStateSnapshot>()).map_err(|_| WireError::TooLarge)?,
        8,
    )?;
    let ptr = NonNull::new(
        std::ptr::with_exposed_provenance_mut::<RenderStateSnapshot>(
            bytes.as_ptr().expose_provenance(),
        ),
    )
    .ok_or(WireError::InvalidValue)?;
    // SAFETY: the canonical record contains only integer/bitflag fields; all bit patterns
    // are valid Rust values, and the aligned immutable packet retains the reference.
    #[cfg(debug_assertions)]
    let state = unsafe { ptr.as_ref() };
    #[cfg(debug_assertions)]
    if state.reserved != 0
        || crate::pipeline_state::PipelineRsFlags::from_bits(state.pipeline_rs.flags.bits())
            .is_none()
        || crate::draw_data::DepthScissorFlags::from_bits(state.depth_scissor.bits()).is_none()
    {
        return Err(WireError::InvalidValue);
    }
    Ok(ptr)
}

fn read_borrowed_stages(reader: &mut SnapshotReader<'_>) -> Result<StageBindingsPtr, WireError> {
    let header: &StageHeader = borrow_record(reader)?;
    let mask = header.mask;
    let count = header.count;
    if u32::from(count) != mask.count_ones() || header.reserved != 0 {
        return Err(WireError::InvalidValue);
    }
    let values = read_aligned_bytes(
        reader,
        u32::from(count)
            * u32::try_from(size_of::<StageBinding>()).map_err(|_| WireError::TooLarge)?,
        8,
    )?;
    let ptr = if count == 0 {
        NonNull::dangling()
    } else {
        NonNull::new(std::ptr::with_exposed_provenance_mut::<StageBinding>(
            values.as_ptr().expose_provenance(),
        ))
        .ok_or(WireError::InvalidValue)?
    };
    // SAFETY: StageBinding's canonical fields admit every bit pattern. The mask-sized
    // array is aligned and retained by the immutable command arena through submission.
    Ok(unsafe { StageBindingsPtr::from_raw_parts(mask, ptr) })
}

fn read_borrowed_attrs(reader: &mut SnapshotReader<'_>) -> Result<AttrSnapshot, WireError> {
    let header: &DeclarationHeader = borrow_record(reader)?;
    let count = header.count;
    if count > 16 || header.reserved != 0 {
        return Err(WireError::InvalidValue);
    }
    let bytes = read_aligned_bytes(
        reader,
        count * u32::try_from(size_of::<VertexAttrDesc>()).map_err(|_| WireError::TooLarge)?,
        8,
    )?;
    #[cfg(debug_assertions)]
    for attr in bytes.as_chunks::<{ size_of::<VertexAttrDesc>() }>().0 {
        let format = u32::from_le_bytes(attr[12..16].try_into().map_err(|_| WireError::Truncated)?);
        mtld3d_shared::mtl::VertexFormat::from_repr(format).ok_or(WireError::InvalidValue)?;
    }
    let ptr = if count == 0 {
        NonNull::dangling()
    } else {
        NonNull::new(std::ptr::with_exposed_provenance_mut::<VertexAttrDesc>(
            bytes.as_ptr().expose_provenance(),
        ))
        .ok_or(WireError::InvalidValue)?
    };
    // SAFETY: the paired typed producer initialized valid attributes; debug builds check
    // discriminants before borrowing. Immutable command storage outlives every token use.
    Ok(unsafe { AttrSnapshot::new(ptr, NonNull::from(header)) })
}

fn write_optional_bytes(
    value: Option<ScratchSlice>,
    writer: &mut SnapshotWriter<'_>,
) -> Result<(), WireError> {
    let (address, length) = value.map_or((0, 0), |value| value.as_raw());
    capture_record(
        writer,
        ByteBindingRecord {
            address,
            length,
            present: u32::from(value.is_some()),
        },
    )
}

fn read_optional_bytes(reader: &mut SnapshotReader<'_>) -> Result<Option<ScratchSlice>, WireError> {
    let record: &ByteBindingRecord = borrow_record(reader)?;
    if record.present == 0 && record.address == 0 && record.length == 0 {
        return Ok(None);
    }
    if record.present != 1 {
        return Err(WireError::InvalidValue);
    }
    if record.length == 0 {
        return Ok(Some(ScratchSlice::EMPTY));
    }
    if usize::try_from(record.length).map_err(|_| WireError::InvalidValue)? > isize::MAX as usize {
        return Err(WireError::InvalidValue);
    }
    let address = usize::try_from(record.address).map_err(|_| WireError::InvalidValue)?;
    address
        .checked_add(record.length as usize)
        .ok_or(WireError::InvalidValue)?;
    let pointer = NonNull::new(address as *mut u8).ok_or(WireError::InvalidValue)?;
    // SAFETY: the trusted frame producer retains this immutable range through every consumer.
    Ok(Some(unsafe {
        ScratchSlice::from_raw_parts(pointer, record.length)
    }))
}

fn read_variant(reader: &mut SnapshotReader<'_>) -> Result<VariantKey, WireError> {
    let record: &VariantRecord = borrow_record(reader)?;
    #[cfg(debug_assertions)]
    if record.reserved != [0; 2]
        || record.key.reserved != 0
        || crate::dxso::VariantFlags::from_bits(record.key.flags.bits()).is_none()
    {
        return Err(WireError::InvalidValue);
    }
    Ok(record.key)
}

fn read_depth_flags(reader: &mut SnapshotReader<'_>) -> Result<DepthStencilFlags, WireError> {
    let record: &DepthFlagsRecord = borrow_record(reader)?;
    if record.reserved != 0 {
        return Err(WireError::InvalidValue);
    }
    DepthStencilFlags::from_bits(u8::try_from(record.flags).map_err(|_| WireError::InvalidValue)?)
        .ok_or(WireError::InvalidValue)
}

/// Canonical fixed-layout draw payloads and retained stream views.
pub mod draw_record;

/// Exact fixed-record payload size, excluding the flat command header.
///
/// # Errors
/// Rejects more than sixteen bound vertex streams.
pub const fn draw_payload_size(draw: &DrawOp) -> Result<usize, WireError> {
    draw_record::payload_size(draw)
}

/// Fill one final draw payload reservation without temporary operation storage.
///
/// # Errors
/// Rejects an incorrect destination size or excessive stream count.
pub fn write_draw_into(draw: &DrawOp, destination: &mut [u8]) -> Result<(), WireError> {
    draw_record::write_into(draw, destination)
}
