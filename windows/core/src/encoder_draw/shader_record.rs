//! Direct capture and borrowed views of canonical shader-source records.

use mtld3d_shared::encoder_wire::WireError;

use super::{SnapshotReader, SnapshotWriter};
use crate::draw_data::{
    FixedPsSource, FixedVsSource, ProgrammablePsSource, ProgrammableVsSource, PsSourcePtr,
    PsSourceView, VsSourcePtr, VsSourceView,
};

#[repr(u8)]
enum SourceKind {
    Programmable = 0,
    FixedFunction = 1,
}

/// Source selector followed by only the selected canonical record.
#[repr(C, align(8))]
struct SourceHeader {
    kind: u8,
    reserved: [u8; 7],
}

/// Copy a padding-free canonical record into its reserved destination.
///
/// # Safety
/// T must have a pinned cross-target layout and every byte must be initialized.
unsafe fn capture<T>(value: &T, writer: &mut SnapshotWriter<'_>) -> Result<(), WireError> {
    let destination = writer.reserve_bytes(core::mem::size_of::<T>())?;
    // SAFETY: caller guarantees initialized bytes and the destination is disjoint and sized for T.
    unsafe {
        core::ptr::copy_nonoverlapping(
            core::ptr::from_ref(value).cast::<u8>(),
            destination.as_mut_ptr(),
            destination.len(),
        );
    };
    Ok(())
}

fn header(writer: &mut SnapshotWriter<'_>, kind: SourceKind) -> Result<(), WireError> {
    super::align_snapshot_leaf(writer)?;
    let value = SourceHeader {
        kind: kind as u8,
        reserved: [0; 7],
    };
    // SAFETY: SourceHeader has no implicit padding and all bytes were initialized above.
    unsafe { capture(&value, writer) }
}

pub(super) fn write_vs(
    source: VsSourceView<'_>,
    writer: &mut SnapshotWriter<'_>,
) -> Result<(), WireError> {
    match source {
        VsSourceView::Programmable(value) => {
            header(writer, SourceKind::Programmable)?;
            // SAFETY: the canonical programmable layout contains only initialized scalar fields.
            unsafe { capture(value, writer) }
        }
        VsSourceView::FixedFunction(value) => {
            if value.reserved != [0; 6] || value.key.reserved != 0 {
                return Err(WireError::InvalidValue);
            }
            header(writer, SourceKind::FixedFunction)?;
            // SAFETY: the canonical key and source have explicit initialized padding only.
            unsafe { capture(value, writer) }
        }
    }
}

pub(super) fn write_ps(
    source: PsSourceView<'_>,
    writer: &mut SnapshotWriter<'_>,
) -> Result<(), WireError> {
    match source {
        PsSourceView::Programmable(value) => {
            if value.reserved != [0; 4] {
                return Err(WireError::InvalidValue);
            }
            header(writer, SourceKind::Programmable)?;
            // SAFETY: the canonical programmable layout has initialized explicit padding only.
            unsafe { capture(value, writer) }
        }
        PsSourceView::FixedFunction(value) => {
            if value.reserved != [0; 3] {
                return Err(WireError::InvalidValue);
            }
            header(writer, SourceKind::FixedFunction)?;
            // SAFETY: key scalar fields, including its bool, and explicit padding are initialized.
            unsafe { capture(value, writer) }
        }
    }
}

fn read_header(reader: &mut SnapshotReader<'_>) -> Result<SourceKind, WireError> {
    let bytes = super::read_aligned_bytes(reader, 8, 8)?;
    if bytes[1..].iter().any(|&byte| byte != 0) {
        return Err(WireError::InvalidValue);
    }
    match bytes[0] {
        value if value == SourceKind::Programmable as u8 => Ok(SourceKind::Programmable),
        value if value == SourceKind::FixedFunction as u8 => Ok(SourceKind::FixedFunction),
        _ => Err(WireError::InvalidValue),
    }
}

/// Borrow bytes after checking their layout and validity at the callsite.
///
/// # Safety
/// The bytes must be correctly aligned, exactly sized and valid for T.
unsafe fn borrow<T>(bytes: &[u8]) -> &T {
    let pointer = bytes.as_ptr() as usize as *const T;
    // SAFETY: the caller checked T's alignment, extent and field validity.
    unsafe { &*pointer }
}

pub(super) fn read_vs(reader: &mut SnapshotReader<'_>) -> Result<VsSourcePtr, WireError> {
    let source = if matches!(read_header(reader)?, SourceKind::FixedFunction) {
        let bytes = super::read_aligned_bytes(reader, 64, 8)?;
        #[cfg(debug_assertions)]
        if bytes[55] != 0
            || bytes[58..].iter().any(|&b| b != 0)
            || crate::dxso::FfVsFlags::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).is_none()
        {
            return Err(WireError::InvalidValue);
        }
        // SAFETY: the paired typed producer initialized valid canonical fields; debug checks
        // above run before the typed borrow. Size and alignment remain checked in every build.
        VsSourceView::FixedFunction(unsafe { borrow::<FixedVsSource>(bytes) })
    } else {
        let bytes = super::read_aligned_bytes(reader, 16, 8)?;
        #[cfg(debug_assertions)]
        if crate::draw_data::ShaderSourceFlags::from_bits(bytes[12]).is_none() {
            return Err(WireError::InvalidValue);
        }
        // SAFETY: all fields are integers or validated flags in the pinned canonical layout.
        VsSourceView::Programmable(unsafe { borrow::<ProgrammableVsSource>(bytes) })
    };
    // SAFETY: trusted reader retains the initialized immutable command arena through all token uses.
    Ok(unsafe { VsSourcePtr::from_view(source) })
}

pub(super) fn read_ps(reader: &mut SnapshotReader<'_>) -> Result<PsSourcePtr, WireError> {
    let source = if matches!(read_header(reader)?, SourceKind::FixedFunction) {
        let bytes = super::read_aligned_bytes(reader, 64, 8)?;
        #[cfg(debug_assertions)]
        if bytes[56] > 1
            || bytes[61..].iter().any(|&b| b != 0)
            || (0..8)
                .any(|stage| crate::dxso::FfStageFlags::from_bits(bytes[stage * 7 + 6]).is_none())
        {
            return Err(WireError::InvalidValue);
        }
        // SAFETY: the paired typed producer initialized the bool and flags; debug validation
        // occurs before this reference, and size/alignment are checked in every build.
        PsSourceView::FixedFunction(unsafe { borrow::<FixedPsSource>(bytes) })
    } else {
        let bytes = super::read_aligned_bytes(reader, 16, 8)?;
        #[cfg(debug_assertions)]
        if bytes[12..].iter().any(|&b| b != 0)
            || crate::draw_data::ShaderSourceFlags::from_bits(bytes[10]).is_none()
        {
            return Err(WireError::InvalidValue);
        }
        // SAFETY: remaining fields are integers and validated flags in the pinned canonical layout.
        PsSourceView::Programmable(unsafe { borrow::<ProgrammablePsSource>(bytes) })
    };
    // SAFETY: trusted reader retains the initialized immutable command arena through all token uses.
    Ok(unsafe { PsSourcePtr::from_view(source) })
}

// Field-by-field layout checks exclude implicit padding on every supported target.
macro_rules! packed_fields {
    ($kind:ty; $($field:ident: $size:expr),+ $(,)?) => {
        const _: () = {
            let mut offset = 0;
            $(assert!(core::mem::offset_of!($kind, $field) == offset); offset += $size;)+
            assert!(core::mem::size_of::<$kind>() == offset);
        };
    };
}
packed_fields!(crate::dxso::FfVsKey;
    flags:2, input_tex_coord_count:1, tex_coord_count:1, light_active_mask:1,
    light_directional_mask:1, light_spot_mask:1, diffuse_source:1, ambient_source:1,
    specular_source:1, emissive_source:1, fog_mode:1, tci_modes:8, tci_coord_indices:8,
    tex_coord_dims:8, tt_flags:8, vertex_blend_count:1, declared_weights_count:1,
    clip_plane_count:1, passthrough:8, reserved:1);
packed_fields!(crate::dxso::FfStage;
    color_op:1, color_arg1:1, color_arg2:1, alpha_op:1, alpha_arg1:1, alpha_arg2:1, flags:1);
packed_fields!(crate::dxso::FfPsKey; stages:56, specular_add:1, tt_projected_mask:1);
packed_fields!(crate::dxso::VsSamplerKinds; volume_mask:1, cube_mask:1);
packed_fields!(crate::dxso::VariantKey;
    alpha_func:1, fog_mode:1, fog_table_mode:1, linked_input_mask:1, depth_sampler_mask:2,
    depth_fetch_mask:2, fetch4_mask:2, fetch4_alpha_mask:2, raw_depth_red_mask:2,
    volume_sampler_mask:2, cube_sampler_mask:2, tt_projected_mask:1, color_out_mask:1,
    sample_mask:1, flags:1);

const _: () = {
    use core::mem::{align_of, offset_of, size_of};

    use crate::dxso::{FfPsKey, FfStage, FfVsKey, VsSamplerKinds};
    assert!(size_of::<SourceHeader>() == 8);
    assert!(size_of::<VsSamplerKinds>() == 2);
    assert!(size_of::<FfVsKey>() == 56);
    assert!(offset_of!(FfVsKey, reserved) == 55);
    assert!(size_of::<FfStage>() == 7);
    assert!(size_of::<FfPsKey>() == 58);
    assert!(offset_of!(FfPsKey, specular_add) == 56);
    assert!(size_of::<ProgrammableVsSource>() == 16);
    assert!(align_of::<ProgrammableVsSource>() == 8);
    assert!(offset_of!(ProgrammableVsSource, vs_id) == 0);
    assert!(offset_of!(ProgrammableVsSource, max_const_used) == 8);
    assert!(offset_of!(ProgrammableVsSource, provided_input_mask) == 10);
    assert!(offset_of!(ProgrammableVsSource, flags) == 12);
    assert!(offset_of!(ProgrammableVsSource, clip_plane_count) == 13);
    assert!(offset_of!(ProgrammableVsSource, sampler_kinds) == 14);
    assert!(size_of::<ProgrammablePsSource>() == 16);
    assert!(align_of::<ProgrammablePsSource>() == 8);
    assert!(offset_of!(ProgrammablePsSource, ps_id) == 0);
    assert!(offset_of!(ProgrammablePsSource, max_const_used) == 8);
    assert!(offset_of!(ProgrammablePsSource, flags) == 10);
    assert!(offset_of!(ProgrammablePsSource, color_out_mask) == 11);
    assert!(offset_of!(ProgrammablePsSource, reserved) == 12);
    assert!(size_of::<FixedVsSource>() == 64);
    assert!(align_of::<FixedVsSource>() == 8);
    assert!(offset_of!(FixedVsSource, key) == 0);
    assert!(offset_of!(FixedVsSource, max_row_count) == 56);
    assert!(offset_of!(FixedVsSource, reserved) == 58);
    assert!(size_of::<FixedPsSource>() == 64);
    assert!(align_of::<FixedPsSource>() == 8);
    assert!(offset_of!(FixedPsSource, key) == 0);
    assert!(offset_of!(FixedPsSource, sampled_stage_mask) == 58);
    assert!(offset_of!(FixedPsSource, constant_rows) == 60);
    assert!(offset_of!(FixedPsSource, reserved) == 61);
};
