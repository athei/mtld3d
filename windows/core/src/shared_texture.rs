//! Shared `D3DPOOL_DEFAULT` textures: what may be shared, and the record a handle names.
//!
//! An extended device shares a texture with a Direct3D 11 device through the
//! `pSharedHandle` of `CreateTexture`: a null `*pSharedHandle` creates a shared
//! texture and receives its handle, any other value is the handle of a texture
//! to open. The handle is a D3DKMT global handle whose private runtime data is
//! a [`SharedTextureRecord`]: the D3D11 description of the texture, and the
//! bootstrap service name of a Mach port through which the Metal texture
//! behind it is opened. That is the record DXMT, the Direct3D 11 runtime of an
//! arm64 Wine, writes for a `D3D11_RESOURCE_MISC_SHARED` texture and reads in
//! `ID3D11Device::OpenSharedResource`, so a texture either runtime creates the
//! other opens.
//!
//! This module is the part of that with no platform call in it: which creates
//! take a share, how a handle value is read, the record's bytes, and the
//! service name. The D3DKMT calls are `d3d9.dll`'s and the Mach and Metal
//! calls `mtld3d.so`'s.

use std::io::Write as _;

use mtld3d_types::{
    D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE, D3D11_RESOURCE_DIMENSION_TEXTURE2D,
    D3D11_RESOURCE_MISC_SHARED, D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8, D3DKMT_GLOBAL_HANDLE_MASK,
    D3DUSAGE_RENDERTARGET, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8X8_UNORM,
};

/// Bytes of the service name in a record, its terminator included.
pub const SERVICE_NAME_LEN: usize = 54;

/// Bytes of a record: the name, two of padding, the dimension, twelve words, a mutex.
pub const RECORD_LEN: usize = 112;

/// What every service name this layer registers starts with.
pub const SERVICE_NAME_PREFIX: &str = "MTLD3D_shared_";

/// Byte offset of the resource dimension in a record.
const DIMENSION_OFFSET: usize = 56;
/// Byte offset of the twelve description words in a record.
const DESC_OFFSET: usize = 60;
/// Byte offset of the keyed-mutex handle in a record.
const MUTEX_OFFSET: usize = 108;

/// Why a `CreateTexture` that asks for a share is refused.
#[derive(Debug, PartialEq, Eq)]
pub enum CreateRefusal {
    /// More or fewer than one level, a runtime-generated chain included.
    Levels,
    /// A usage other than none or `D3DUSAGE_RENDERTARGET`.
    Usage,
    /// A format the two runtimes do not both map to one Metal format.
    Format,
}

impl CreateRefusal {
    /// The reason, for the line that reports the refusal.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::Levels => "a shared texture has exactly one level",
            Self::Usage => "a shared texture takes no usage but D3DUSAGE_RENDERTARGET",
            Self::Format => "only A8R8G8B8 and X8R8G8B8 textures are shared",
        }
    }
}

/// Whether a shared `CreateTexture` of this shape is refused, and why.
///
/// A shared texture is one level of a colour format both runtimes store as
/// the same Metal format, drawn into by one of them and sampled by the other:
/// no mip chain, no depth, and none of the usages that make a texture
/// lockable or have the runtime write it.
#[must_use]
pub const fn create_refusal(levels: u32, usage: u32, d3d_format: u32) -> Option<CreateRefusal> {
    if levels != 1 {
        Some(CreateRefusal::Levels)
    } else if usage & !D3DUSAGE_RENDERTARGET != 0 {
        Some(CreateRefusal::Usage)
    } else if dxgi_format(d3d_format).is_none() {
        Some(CreateRefusal::Format)
    } else {
        None
    }
}

/// The DXGI format a D3D9 format is shared as, `None` for one that is not shared.
#[must_use]
pub const fn dxgi_format(d3d_format: u32) -> Option<u32> {
    match d3d_format {
        D3DFMT_A8R8G8B8 => Some(DXGI_FORMAT_B8G8R8A8_UNORM),
        D3DFMT_X8R8G8B8 => Some(DXGI_FORMAT_B8G8R8X8_UNORM),
        _ => None,
    }
}

/// The D3DKMT global handle a non-null `*pSharedHandle` holds, `None` when it holds none.
///
/// A global handle is a 32-bit value with one of its two top bits set. A
/// wider value, or one with both clear, is an NT handle or no handle at all,
/// neither of which D3D9 opens.
#[must_use]
pub fn global_handle(slot: usize) -> Option<u32> {
    let handle = u32::try_from(slot).ok()?;
    (handle & D3DKMT_GLOBAL_HANDLE_MASK != 0).then_some(handle)
}

/// The private runtime data of the D3DKMT resource behind a shared texture's handle.
///
/// On the wire it is [`RECORD_LEN`] bytes: the NUL-terminated service name in
/// [`SERVICE_NAME_LEN`] bytes, two bytes of padding, the
/// `D3D11_RESOURCE_DIMENSION`, the twelve 32-bit words of a
/// `D3D11_TEXTURE2D_DESC1` and the handle of a keyed mutex, all little-endian.
#[derive(Debug, PartialEq, Eq)]
pub struct SharedTextureRecord {
    service_name: [u8; SERVICE_NAME_LEN],
    dimension: u32,
    width: u32,
    height: u32,
    mip_levels: u32,
    array_size: u32,
    format: u32,
    sample_count: u32,
    bind_flags: u32,
    misc_flags: u32,
    keyed_mutex: u32,
}

impl SharedTextureRecord {
    /// The record of a texture this layer creates shared.
    ///
    /// Described the way the opening runtime has to see it: one level, one
    /// sample, a render target that is also sampled, shared through a legacy
    /// handle, with no keyed mutex.
    #[must_use]
    pub const fn for_created_texture(
        service_name: [u8; SERVICE_NAME_LEN],
        width: u32,
        height: u32,
        dxgi_format: u32,
    ) -> Self {
        Self {
            service_name,
            dimension: D3D11_RESOURCE_DIMENSION_TEXTURE2D,
            width,
            height,
            mip_levels: 1,
            array_size: 1,
            format: dxgi_format,
            sample_count: 1,
            bind_flags: D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,
            misc_flags: D3D11_RESOURCE_MISC_SHARED,
            keyed_mutex: 0,
        }
    }

    /// Read a record from the runtime data of a queried resource.
    #[must_use]
    pub fn from_bytes(bytes: &[u8; RECORD_LEN]) -> Self {
        let mut service_name = [0; SERVICE_NAME_LEN];
        service_name.copy_from_slice(&bytes[..SERVICE_NAME_LEN]);
        let desc = |index: usize| word(bytes, DESC_OFFSET + index * 4);
        Self {
            service_name,
            dimension: word(bytes, DIMENSION_OFFSET),
            width: desc(desc_word::WIDTH),
            height: desc(desc_word::HEIGHT),
            mip_levels: desc(desc_word::MIP_LEVELS),
            array_size: desc(desc_word::ARRAY_SIZE),
            format: desc(desc_word::FORMAT),
            sample_count: desc(desc_word::SAMPLE_COUNT),
            bind_flags: desc(desc_word::BIND_FLAGS),
            misc_flags: desc(desc_word::MISC_FLAGS),
            keyed_mutex: word(bytes, MUTEX_OFFSET),
        }
    }

    /// The record's bytes, for the runtime data of a resource being created.
    ///
    /// The description words this type does not carry (sample quality, usage,
    /// CPU access, texture layout) are zero, which are their defaults.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; RECORD_LEN] {
        let mut bytes = [0; RECORD_LEN];
        bytes[..SERVICE_NAME_LEN].copy_from_slice(&self.service_name);
        put_word(&mut bytes, DIMENSION_OFFSET, self.dimension);
        for (index, value) in [
            (desc_word::WIDTH, self.width),
            (desc_word::HEIGHT, self.height),
            (desc_word::MIP_LEVELS, self.mip_levels),
            (desc_word::ARRAY_SIZE, self.array_size),
            (desc_word::FORMAT, self.format),
            (desc_word::SAMPLE_COUNT, self.sample_count),
            (desc_word::BIND_FLAGS, self.bind_flags),
            (desc_word::MISC_FLAGS, self.misc_flags),
        ] {
            put_word(&mut bytes, DESC_OFFSET + index * 4, value);
        }
        put_word(&mut bytes, MUTEX_OFFSET, self.keyed_mutex);
        bytes
    }

    /// Whether the record describes the single-sample 2D texture of this extent a create asks for.
    ///
    /// The format is not judged here: the Metal texture behind the record is
    /// checked against the Metal format of the D3D9 one when it is opened,
    /// which also covers the typeless formats a D3D11 texture can have.
    #[must_use]
    pub const fn describes_texture(&self, width: u32, height: u32) -> bool {
        self.dimension == D3D11_RESOURCE_DIMENSION_TEXTURE2D
            && self.width == width
            && self.height == height
            && self.array_size == 1
            && self.sample_count <= 1
    }

    /// The NUL-terminated bootstrap service name of the texture's Mach port.
    #[must_use]
    pub const fn service_name(&self) -> &[u8; SERVICE_NAME_LEN] {
        &self.service_name
    }

    /// The `D3D11_RESOURCE_DIMENSION` of the resource.
    #[must_use]
    pub const fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Width and height of the top level.
    #[must_use]
    pub const fn extent(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Mip level count of the resource.
    #[must_use]
    pub const fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    /// Array size of the resource.
    #[must_use]
    pub const fn array_size(&self) -> u32 {
        self.array_size
    }

    /// The `DXGI_FORMAT` of the resource.
    #[must_use]
    pub const fn format(&self) -> u32 {
        self.format
    }

    /// Samples per texel.
    #[must_use]
    pub const fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// The `D3D11_BIND_FLAG` bits of the resource.
    #[must_use]
    pub const fn bind_flags(&self) -> u32 {
        self.bind_flags
    }

    /// The `D3D11_RESOURCE_MISC_FLAG` bits of the resource.
    #[must_use]
    pub const fn misc_flags(&self) -> u32 {
        self.misc_flags
    }

    /// The handle of the keyed mutex guarding the texture, 0 when it has none.
    #[must_use]
    pub const fn keyed_mutex(&self) -> u32 {
        self.keyed_mutex
    }
}

/// The service name a texture created shared registers its Mach port under.
///
/// Bootstrap names are one namespace for the login session, so the name
/// carries the process, the texture's id, which is unique in the process, and
/// the low 48 bits of a clock reading, which keeps a recycled process id from
/// colliding with a name a predecessor left registered. The result is
/// NUL-terminated within [`SERVICE_NAME_LEN`] bytes.
///
/// # Panics
///
/// Never: the formatted name is 50 bytes, which the field holds.
#[must_use]
pub fn service_name(pid: u32, texture_id: u64, ticks: u64) -> [u8; SERVICE_NAME_LEN] {
    let mut name = [0; SERVICE_NAME_LEN];
    let ticks = ticks & 0xffff_ffff_ffff;
    let mut text = &mut name[..SERVICE_NAME_LEN - 1];
    write!(
        text,
        "{SERVICE_NAME_PREFIX}{pid:08x}{texture_id:016x}{ticks:012x}"
    )
    .expect("the service name fits its field");
    name
}

/// Indices of the description words a record carries, in a `D3D11_TEXTURE2D_DESC1`.
mod desc_word {
    pub const WIDTH: usize = 0;
    pub const HEIGHT: usize = 1;
    pub const MIP_LEVELS: usize = 2;
    pub const ARRAY_SIZE: usize = 3;
    pub const FORMAT: usize = 4;
    pub const SAMPLE_COUNT: usize = 5;
    pub const BIND_FLAGS: usize = 8;
    pub const MISC_FLAGS: usize = 10;
}

fn word(bytes: &[u8; RECORD_LEN], offset: usize) -> u32 {
    let mut raw = [0; 4];
    raw.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(raw)
}

fn put_word(bytes: &mut [u8; RECORD_LEN], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests;
