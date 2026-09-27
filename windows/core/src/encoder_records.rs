//! Fixed, aligned command payloads shared by API capture and native execution.
//!
//! Fields have one representation on every supported target. Owners stay in the
//! packet ledgers; these records contain only scalar values and retained spans.

use mtld3d_shared::encoder_wire::WireError;

use crate::{
    guest_pages::{GuestOwnedPageDescriptor, GuestPageDescriptor},
    guest_queries::GuestQueryDescriptor,
    upload_redirty::GuestRedirtyDescriptor,
};

/// An initialized fixed-layout command with no implicit padding or invalid bit patterns.
///
/// # Safety
/// Every byte must belong to an initialized field. All field bit patterns must be valid.
/// The layout must be identical on all four supported targets.
pub unsafe trait CommandRecord {}

/// Borrow an exact record from an aligned command payload.
///
/// # Errors
/// Returns an error if the payload size or alignment differs from the record.
pub fn borrow<T: CommandRecord>(bytes: &[u8]) -> Result<&T, WireError> {
    let values = borrow_array::<T>(bytes)?;
    if values.len() != 1 {
        return Err(WireError::InvalidValue);
    }
    Ok(&values[0])
}

/// Initialize a fixed command directly in its final reserved payload.
///
/// # Errors
/// Returns an error if the destination does not have the exact record size.
pub fn write<T: CommandRecord>(destination: &mut [u8], value: T) -> Result<(), WireError> {
    // SAFETY: MaybeUninit accepts all current destination byte patterns. The split
    // exposes only aligned slots, and the exact one-record extent is checked below.
    let (prefix, values, suffix) =
        unsafe { destination.align_to_mut::<core::mem::MaybeUninit<T>>() };
    if !prefix.is_empty() || !suffix.is_empty() || values.len() != 1 {
        return Err(WireError::InvalidValue);
    }
    values[0].write(value);
    Ok(())
}

macro_rules! record {
    ($name:ident { $($field:ident : $ty:ty),+ $(,)? }) => {
        #[repr(C, align(8))]
        pub struct $name { $(pub $field: $ty),+ }
        const _: () = {
            assert!(cfg!(target_endian = "little"));
            assert!(align_of::<$name>() == 8);
            let mut offset = 0;
            $(assert!(core::mem::offset_of!($name, $field) == offset); offset += size_of::<$ty>();)+
            assert!(size_of::<$name>() == offset);
        };
        // SAFETY: every field is an integer, float, or canonical record. The assertions
        // check every field offset and the total size, excluding implicit padding.
        unsafe impl CommandRecord for $name {}
    };
}

record!(ByteSpan {
    address: u64,
    length: u64
});
record!(RectRecord {
    x: i32,
    y: i32,
    right: i32,
    bottom: i32
});
record!(TextureRecord {
    id: u64,
    pixel_format: u32,
    d3d_format: u32,
    width: u32,
    height: u32,
    depth: u32,
    levels: u32,
    create_flags: u32,
    usage_flags: u32,
    swizzle: [u32; 4]
});
record!(DepthTransferRecord {
    source: u64,
    destination: u64,
    source_level: u32,
    source_width: u32,
    source_height: u32,
    source_format: u32,
    source_samples: u32,
    destination_width: u32,
    destination_height: u32,
    destination_format: u32
});
record!(SetViewportRecord {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    min_z: f32,
    max_z: f32
});
record!(SetVertexSamplerRecord {
    slot: u32,
    reserved: u32,
    state: [u32; mtld3d_types::SAMPLER_STATE_COUNT]
});
record!(SetVertexTextureRecord {
    id: u64,
    slot: u32,
    present: u32
});
record!(BindDepthNoneRecord {
    sample_count: u32,
    flags: u32
});
record!(BindDepthEagerRecord {
    handle: u64,
    width: u32,
    height: u32,
    scale: u32,
    sample_count: u32,
    flags: u32,
    reserved: u32
});
record!(BindDepthLazyRecord {
    texture: TextureRecord,
    level: u32,
    scale: u32,
    sample_count: u32,
    flags: u32
});
record!(BindColorBackbufferRecord {
    handle: u64,
    msaa: u64,
    msaa_srgb: u64,
    width: u32,
    height: u32,
    slot: u32,
    scale: u32,
    sample_count: u32,
    reserved: u32
});
record!(BindColorStandaloneRecord {
    handle: u64,
    srgb: u64,
    msaa: u64,
    msaa_srgb: u64,
    width: u32,
    height: u32,
    slot: u32,
    scale: u32,
    sample_count: u32,
    format: u32,
    has_alpha: u32,
    reserved: u32
});
record!(BindColorTextureRecord {
    texture: TextureRecord,
    width: u32,
    height: u32,
    slot: u32,
    scale: u32,
    slice: u32,
    level: u32,
    has_alpha: u32,
    reserved: u32
});
record!(IdRecord { id: u64 });
record!(SlotRecord {
    value: u32,
    reserved: u32
});
record!(RetireColorRecord {
    base: u64,
    srgb: u64,
    msaa: u64,
    msaa_srgb: u64
});
record!(ReadHandleRecord {
    id: u64,
    reply: u64
});
record!(ReadDeviceBufferRecord {
    id: u64,
    destination: u64,
    length: u64,
    reply: u64
});
record!(QueryRecord {
    generation: u64,
    descriptor: GuestQueryDescriptor
});
record!(CarryDepthRecord {
    previous: u64,
    current: u64,
    width: u32,
    height: u32
});
record!(ClearColorRecord {
    rgba: [u32; 4],
    srgb_write: u32,
    reserved: u32
});
record!(ClearDepthStencilRecord {
    depth: u32,
    stencil: u32,
    present: u32,
    reserved: u32
});
record!(ResolveDynamicDepthRecord {
    id: u64,
    texture: TextureRecord
});
record!(ResolveDepthTextureRecord {
    id: u64,
    width: u32,
    height: u32,
    format: u32,
    reserved: u32
});
record!(UploadColorRecord {
    handle: u64,
    bytes: ByteSpan,
    width: u32,
    height: u32,
    stride: u32,
    reserved: u32
});
record!(ResampledTargetRecord {
    handle: u64,
    msaa: u64,
    msaa_srgb: u64,
    format: u32,
    logical_width: u32,
    logical_height: u32,
    texture_width: u32,
    texture_height: u32,
    stride: u32,
    sample_count: u32,
    reserved: u32
});
record!(UploadResampledRecord {
    target: ResampledTargetRecord,
    bytes: ByteSpan
});
// A surface kind is texture (0), backbuffer (1), or standalone depth (2).
// The unused identity branch is initialized to zero, never interpreted as a Rust enum.
record!(SurfaceIdentityRecord {
    texture: TextureRecord,
    handle: u64,
    kind: u32,
    reserved: u32
});
record!(SurfaceRecord {
    identity: SurfaceIdentityRecord,
    autogen_id: u64,
    msaa: u64,
    msaa_srgb: u64,
    width: u32,
    height: u32,
    texture_width: u32,
    texture_height: u32,
    scale: u32,
    format: u32,
    mip_level: u32,
    slice: u32,
    pool: u32,
    flags: u32,
    sample_count: u32,
    present: u32
});
record!(StretchBlitRecord {
    source: SurfaceRecord,
    destination: SurfaceRecord,
    source_region: [u32; 4],
    destination_region: [u32; 4],
    mip_level: u32,
    render_quad: u32,
    filter: u32,
    reserved: u32
});
record!(ColorFillRecord {
    identity: SurfaceIdentityRecord,
    handle: u64,
    msaa: u64,
    msaa_srgb: u64,
    logical_width: u32,
    logical_height: u32,
    texture_width: u32,
    texture_height: u32,
    format: u32,
    scale: u32,
    slice: u32,
    level: u32,
    rect: [u32; 4],
    rgba: [u32; 4],
    sample_count: u32,
    regenerate_mipmaps: u32
});
record!(StageUploadRecord {
    id: u64,
    offset: u64,
    size: u64,
    page: GuestOwnedPageDescriptor
});
const _: () = {
    assert!(size_of::<StageUploadRecord>() == 56);
    assert!(align_of::<StageUploadRecord>() == 8);
};
record!(TextureUploadRecord {
    texture: TextureRecord,
    page: GuestPageDescriptor,
    redirty: GuestRedirtyDescriptor,
    mip_texture: u64,
    level: u32,
    destination_slice: u32,
    staging_index: u32,
    origin_x: u32,
    origin_y: u32,
    width: u32,
    height: u32,
    source_format: u32,
    pitch: u32,
    bytes_per_pixel: u32,
    depth: u32,
    slice_pitch: u32,
    release_staging: u32,
    upload_generation: u32,
    mip_flags: u32,
    reserved: u32
});

impl TextureRecord {
    #[must_use]
    pub const fn capture(value: &crate::encoder_data::TextureInfo) -> Self {
        Self {
            id: value.texture_id.raw(),
            pixel_format: value.pixel_format as u32,
            d3d_format: value.d3d_format,
            width: value.width,
            height: value.height,
            depth: value.depth,
            levels: value.levels,
            create_flags: value.create_flags.bits(),
            usage_flags: value.usage_flags.bits(),
            swizzle: [
                value.swizzle[0] as u32,
                value.swizzle[1] as u32,
                value.swizzle[2] as u32,
                value.swizzle[3] as u32,
            ],
        }
    }
    pub const ZERO: Self = Self {
        id: 0,
        pixel_format: 0,
        d3d_format: 0,
        width: 0,
        height: 0,
        depth: 0,
        levels: 0,
        create_flags: 0,
        usage_flags: 0,
        swizzle: [0; 4],
    };
}

/// Borrow a fixed prefix and return its remaining inline array bytes.
///
/// # Errors
/// Returns an error for a short or unaligned prefix.
pub fn borrow_prefix<T: CommandRecord>(bytes: &[u8]) -> Result<(&T, &[u8]), WireError> {
    let (head, tail) = bytes
        .split_at_checked(size_of::<T>())
        .ok_or(WireError::Truncated)?;
    Ok((borrow(head)?, tail))
}

/// Borrow an inline array of fixed records without allocating or copying.
///
/// # Errors
/// Returns an error for incomplete or unaligned records.
pub fn borrow_array<T: CommandRecord>(bytes: &[u8]) -> Result<&[T], WireError> {
    // SAFETY: CommandRecord permits all bit patterns; align_to identifies alignment gaps.
    let (prefix, values, suffix) = unsafe { bytes.align_to::<T>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(WireError::InvalidValue);
    }
    Ok(values)
}

impl TextureRecord {
    #[must_use]
    pub const fn texture_id(&self) -> crate::ids::TextureId {
        crate::ids::TextureId::from_raw(self.id)
    }
    /// Read the Metal pixel format.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown Metal pixel format.
    pub fn format(&self) -> Result<mtld3d_shared::mtl::PixelFormat, WireError> {
        mtld3d_shared::mtl::PixelFormat::from_repr(self.pixel_format).ok_or(WireError::InvalidValue)
    }
    /// Read the texture creation flags.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown creation flags.
    pub fn creation_flags(&self) -> Result<mtld3d_shared::mtl::TextureCreateFlags, WireError> {
        mtld3d_shared::mtl::TextureCreateFlags::from_bits(self.create_flags)
            .ok_or(WireError::InvalidValue)
    }
    /// Read the texture usage flags.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown usage flags.
    pub fn usage(&self) -> Result<mtld3d_shared::mtl::TextureUsage, WireError> {
        mtld3d_shared::mtl::TextureUsage::from_bits(self.usage_flags).ok_or(WireError::InvalidValue)
    }
    /// Read the channel swizzle.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown swizzle component.
    pub fn channels(&self) -> Result<[mtld3d_shared::mtl::Swizzle; 4], WireError> {
        use mtld3d_shared::mtl::Swizzle;
        Ok([
            Swizzle::from_repr(self.swizzle[0]).ok_or(WireError::InvalidValue)?,
            Swizzle::from_repr(self.swizzle[1]).ok_or(WireError::InvalidValue)?,
            Swizzle::from_repr(self.swizzle[2]).ok_or(WireError::InvalidValue)?,
            Swizzle::from_repr(self.swizzle[3]).ok_or(WireError::InvalidValue)?,
        ])
    }
}

impl TextureUploadRecord {
    /// Check the retained source extent without adopting any resource.
    ///
    /// # Errors
    /// Returns an error for an invalid subresource, layout or source span.
    pub fn validate_source(&self, logical_len: u64) -> Result<(), WireError> {
        use mtld3d_shared::{blit_geometry::source_rows_end, mtl::TextureCreateFlags};
        if self.level >= 32 || self.level >= self.texture.levels || self.depth == 0 {
            return Err(WireError::InvalidValue);
        }
        let cube = self
            .texture
            .creation_flags()?
            .contains(TextureCreateFlags::TYPE_CUBE);
        if self.destination_slice >= if cube { 6 } else { 1 } {
            return Err(WireError::InvalidValue);
        }
        let expected_index = self
            .destination_slice
            .checked_mul(self.texture.levels)
            .and_then(|base| base.checked_add(self.level))
            .ok_or(WireError::InvalidValue)?;
        if self.staging_index != expected_index {
            return Err(WireError::InvalidValue);
        }
        let source_bpp = if matches!(
            self.source_format,
            mtld3d_types::D3DFMT_YV12 | mtld3d_types::D3DFMT_NV12
        ) {
            1
        } else if let Some(bytes) = crate::format::depth_format_bytes_per_pixel(self.source_format)
        {
            bytes
        } else {
            crate::format::map_d3d_format(self.source_format)
                .ok_or(WireError::InvalidValue)?
                .bytes_per_pixel()
        };
        if self.bytes_per_pixel != source_bpp {
            return Err(WireError::InvalidValue);
        }
        let mip_depth = (self.texture.depth >> self.level).max(1);
        if self.depth > mip_depth {
            return Err(WireError::InvalidValue);
        }
        // Planar API capture describes the full storage texture, including chroma rows.
        // It is never a partial luma upload, even when the application dirtied a subrectangle.
        if matches!(
            self.source_format,
            mtld3d_types::D3DFMT_YV12 | mtld3d_types::D3DFMT_NV12
        ) && (self.level != 0
            || self.origin_x != 0
            || self.origin_y != 0
            || self.width != self.texture.width
            || self.height != self.texture.height
            || u64::from(self.pitch) * u64::from(self.texture.height) > logical_len)
        {
            return Err(WireError::InvalidValue);
        }
        let end_x = self
            .origin_x
            .checked_add(self.width)
            .ok_or(WireError::InvalidValue)?;
        let end_y = self
            .origin_y
            .checked_add(self.height)
            .ok_or(WireError::InvalidValue)?;
        let (row_bytes, row_end) = if self.bytes_per_pixel == 0 {
            let format =
                crate::format::map_d3d_format(self.source_format).ok_or(WireError::InvalidValue)?;
            if !format.is_compressed() {
                return Err(WireError::InvalidValue);
            }
            let mip_width = (self.texture.width >> self.level).max(1);
            let mip_height = (self.texture.height >> self.level).max(1);
            if end_x > mip_width || end_y > mip_height {
                return Err(WireError::InvalidValue);
            }
            let row_bytes = u64::from(mip_width.div_ceil(format.block_width()))
                * u64::from(format.block_bytes());
            (row_bytes, mip_height.div_ceil(format.block_height()))
        } else {
            (u64::from(end_x) * u64::from(self.bytes_per_pixel), end_y)
        };
        if row_bytes > u64::from(self.pitch) {
            return Err(WireError::InvalidValue);
        }
        let first_slice_end =
            source_rows_end(0, self.pitch, row_end).ok_or(WireError::InvalidValue)?;
        let required = if self.depth > 1 {
            if u64::from(self.slice_pitch) < first_slice_end {
                return Err(WireError::InvalidValue);
            }
            u64::from(self.slice_pitch)
                .checked_mul(u64::from(self.depth - 1))
                .and_then(|last| last.checked_add(first_slice_end))
                .ok_or(WireError::InvalidValue)?
        } else {
            first_slice_end
        };
        if required > logical_len {
            return Err(WireError::InvalidValue);
        }
        Ok(())
    }
}
