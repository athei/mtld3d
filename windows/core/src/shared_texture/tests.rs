use mtld3d_types::{
    D3DFMT_A8R8G8B8, D3DFMT_A16B16G16R16F, D3DFMT_D24S8, D3DFMT_DXT1, D3DFMT_R5G6B5,
    D3DFMT_X8R8G8B8, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_DYNAMIC,
    D3DUSAGE_RENDERTARGET,
};

use super::*;

/// A handle value, for the creates that open one.
const HANDLE: usize = 0x4000_0042;

/// The bytes of a record as the other runtime writes them for a 2D texture.
///
/// A `port` service name, one level, one slice, one sample, a sampled render
/// target shared through a legacy handle, no keyed mutex.
fn record_bytes(width: u32, height: u32, format: u32) -> [u8; RECORD_LEN] {
    let mut bytes = [0; RECORD_LEN];
    bytes[..4].copy_from_slice(b"port");
    for (offset, value) in [
        (56, D3D11_RESOURCE_DIMENSION_TEXTURE2D),
        (60, width),
        (64, height),
        (68, 1),
        (72, 1),
        (76, format),
        (80, 1),
        (92, 0x28),
        (100, 0x2),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[test]
fn only_the_two_bgra_formats_share() {
    assert!(shares_format(D3DFMT_A8R8G8B8));
    assert!(shares_format(D3DFMT_X8R8G8B8));
    for format in [
        D3DFMT_R5G6B5,
        D3DFMT_A16B16G16R16F,
        D3DFMT_DXT1,
        D3DFMT_D24S8,
    ] {
        assert!(!shares_format(format), "format {format}");
    }
}

#[test]
fn a_single_level_plain_or_render_target_texture_opens() {
    for usage in [0, D3DUSAGE_RENDERTARGET] {
        for format in [D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8] {
            assert_eq!(create_refusal(HANDLE, 1, usage, format), None);
        }
    }
}

#[test]
fn a_null_handle_is_refused_before_anything_else() {
    assert_eq!(
        create_refusal(0, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8),
        Some(CreateRefusal::NewTexture)
    );
    assert_eq!(
        create_refusal(0, 4, D3DUSAGE_DYNAMIC, D3DFMT_DXT1),
        Some(CreateRefusal::NewTexture)
    );
}

#[test]
fn a_mip_chain_is_refused_before_usage_and_format() {
    for levels in [0, 2, 12] {
        assert_eq!(
            create_refusal(HANDLE, levels, D3DUSAGE_DYNAMIC, D3DFMT_DXT1),
            Some(CreateRefusal::Levels)
        );
    }
}

#[test]
fn every_usage_but_render_target_is_refused() {
    for usage in [
        D3DUSAGE_DYNAMIC,
        D3DUSAGE_DEPTHSTENCIL,
        D3DUSAGE_AUTOGENMIPMAP,
        D3DUSAGE_RENDERTARGET | D3DUSAGE_DYNAMIC,
    ] {
        assert_eq!(
            create_refusal(HANDLE, 1, usage, D3DFMT_A8R8G8B8),
            Some(CreateRefusal::Usage),
            "usage {usage:#x}"
        );
    }
}

#[test]
fn an_unshared_format_is_refused() {
    assert_eq!(
        create_refusal(HANDLE, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A16B16G16R16F),
        Some(CreateRefusal::Format)
    );
}

#[test]
fn every_refusal_has_its_own_reason() {
    let reasons = [
        CreateRefusal::NewTexture.reason(),
        CreateRefusal::Levels.reason(),
        CreateRefusal::Usage.reason(),
        CreateRefusal::Format.reason(),
    ];
    assert!(reasons.iter().all(|reason| !reason.is_empty()));
    for (index, reason) in reasons.iter().enumerate() {
        assert!(
            !reasons[..index].contains(reason),
            "reason {index} repeats an earlier one"
        );
    }
}

#[test]
fn a_global_handle_has_one_of_its_two_top_bits() {
    assert_eq!(global_handle(0x4000_0042), Some(0x4000_0042));
    assert_eq!(global_handle(0x8000_0001), Some(0x8000_0001));
    assert_eq!(global_handle(0xc000_0abc), Some(0xc000_0abc));
}

#[test]
fn an_nt_handle_or_a_pointer_is_no_global_handle() {
    assert_eq!(global_handle(0), None);
    assert_eq!(global_handle(0x1f4), None);
    assert_eq!(global_handle(0x3fff_ffff), None);
    #[cfg(target_pointer_width = "64")]
    assert_eq!(global_handle(0x1_4000_0042), None);
}

#[test]
fn a_foreign_record_is_read_field_by_field() {
    let mut bytes = [0; RECORD_LEN];
    bytes[..4].copy_from_slice(b"peer");
    for (offset, value) in [
        (56, 3_u32),
        (60, 2048),
        (64, 1024),
        (68, 1),
        (72, 1),
        (76, 87),
        (80, 1),
        (92, 0x28),
        (100, 0x802),
        (108, 0x1234),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    let record = SharedTextureRecord::from_bytes(&bytes);
    assert_eq!(&record.service_name()[..5], b"peer\0");
    assert_eq!(record.dimension(), D3D11_RESOURCE_DIMENSION_TEXTURE2D);
    assert_eq!(record.extent(), (2048, 1024));
    assert_eq!(record.mip_levels(), 1);
    assert_eq!(record.array_size(), 1);
    assert_eq!(record.sample_count(), 1);
    assert_eq!(record.format(), 87);
    assert_eq!(record.bind_flags(), 0x28);
    assert_eq!(record.misc_flags(), 0x802);
    assert_eq!(record.keyed_mutex(), 0x1234);
    assert!(record.describes_texture(2048, 1024));
}

#[test]
fn a_record_of_another_shape_does_not_describe_the_texture() {
    let texture = || record_bytes(640, 480, 87);
    assert!(SharedTextureRecord::from_bytes(&texture()).describes_texture(640, 480));
    assert!(!SharedTextureRecord::from_bytes(&texture()).describes_texture(640, 481));
    assert!(!SharedTextureRecord::from_bytes(&texture()).describes_texture(641, 480));
    // A 3D texture, a two-slice array and a four-sample texture of the extent.
    for (offset, value) in [(56, 4_u32), (72, 2), (80, 4)] {
        let mut bytes = texture();
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        assert!(
            !SharedTextureRecord::from_bytes(&bytes).describes_texture(640, 480),
            "word at {offset} = {value}"
        );
    }
    // A sample count of zero is what a description leaves when it never set one.
    let mut bytes = texture();
    bytes[80..84].copy_from_slice(&0_u32.to_le_bytes());
    assert!(SharedTextureRecord::from_bytes(&bytes).describes_texture(640, 480));
}
