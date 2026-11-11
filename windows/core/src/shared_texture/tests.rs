use std::ffi::CStr;

use mtld3d_types::{
    D3DFMT_A8R8G8B8, D3DFMT_A16B16G16R16F, D3DFMT_D24S8, D3DFMT_DXT1, D3DFMT_R5G6B5,
    D3DFMT_X8R8G8B8, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_DYNAMIC,
    D3DUSAGE_RENDERTARGET,
};

use super::*;

fn name(text: &str) -> [u8; SERVICE_NAME_LEN] {
    let mut name = [0; SERVICE_NAME_LEN];
    name[..text.len()].copy_from_slice(text.as_bytes());
    name
}

fn le(bytes: &[u8; RECORD_LEN], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
}

#[test]
fn only_the_two_bgra_formats_share() {
    assert_eq!(
        dxgi_format(D3DFMT_A8R8G8B8),
        Some(DXGI_FORMAT_B8G8R8A8_UNORM)
    );
    assert_eq!(
        dxgi_format(D3DFMT_X8R8G8B8),
        Some(DXGI_FORMAT_B8G8R8X8_UNORM)
    );
    for format in [
        D3DFMT_R5G6B5,
        D3DFMT_A16B16G16R16F,
        D3DFMT_DXT1,
        D3DFMT_D24S8,
    ] {
        assert_eq!(dxgi_format(format), None, "format {format}");
    }
}

#[test]
fn a_single_level_plain_or_render_target_texture_shares() {
    for usage in [0, D3DUSAGE_RENDERTARGET] {
        for format in [D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8] {
            assert_eq!(create_refusal(1, usage, format), None);
        }
    }
}

#[test]
fn a_mip_chain_is_refused_before_anything_else() {
    for levels in [0, 2, 12] {
        assert_eq!(
            create_refusal(levels, D3DUSAGE_DYNAMIC, D3DFMT_DXT1),
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
            create_refusal(1, usage, D3DFMT_A8R8G8B8),
            Some(CreateRefusal::Usage),
            "usage {usage:#x}"
        );
    }
}

#[test]
fn an_unshared_format_is_refused() {
    assert_eq!(
        create_refusal(1, D3DUSAGE_RENDERTARGET, D3DFMT_A16B16G16R16F),
        Some(CreateRefusal::Format)
    );
}

#[test]
fn every_refusal_has_its_own_reason() {
    let reasons = [
        CreateRefusal::Levels.reason(),
        CreateRefusal::Usage.reason(),
        CreateRefusal::Format.reason(),
    ];
    assert!(reasons.iter().all(|reason| !reason.is_empty()));
    assert_ne!(reasons[0], reasons[1]);
    assert_ne!(reasons[1], reasons[2]);
    assert_ne!(reasons[0], reasons[2]);
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
fn a_created_texture_is_described_as_a_shared_sampled_render_target() {
    let record = SharedTextureRecord::for_created_texture(
        name("port"),
        640,
        480,
        DXGI_FORMAT_B8G8R8A8_UNORM,
    );
    assert_eq!(record.dimension(), D3D11_RESOURCE_DIMENSION_TEXTURE2D);
    assert_eq!(record.extent(), (640, 480));
    assert_eq!(record.mip_levels(), 1);
    assert_eq!(record.array_size(), 1);
    assert_eq!(record.sample_count(), 1);
    assert_eq!(record.format(), DXGI_FORMAT_B8G8R8A8_UNORM);
    assert_eq!(
        record.bind_flags(),
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET
    );
    assert_eq!(record.misc_flags(), D3D11_RESOURCE_MISC_SHARED);
    assert_eq!(record.keyed_mutex(), 0);
    assert!(record.describes_texture(640, 480));
}

#[test]
fn the_record_lays_its_fields_out_at_the_wire_offsets() {
    let record = SharedTextureRecord::for_created_texture(
        name("port"),
        640,
        480,
        DXGI_FORMAT_B8G8R8X8_UNORM,
    );
    let bytes = record.to_bytes();
    assert_eq!(&bytes[..5], b"port\0");
    assert!(bytes[4..56].iter().all(|byte| *byte == 0));
    assert_eq!(le(&bytes, 56), D3D11_RESOURCE_DIMENSION_TEXTURE2D);
    // Width, Height, MipLevels, ArraySize, Format, SampleDesc.Count.
    assert_eq!(le(&bytes, 60), 640);
    assert_eq!(le(&bytes, 64), 480);
    assert_eq!(le(&bytes, 68), 1);
    assert_eq!(le(&bytes, 72), 1);
    assert_eq!(le(&bytes, 76), DXGI_FORMAT_B8G8R8X8_UNORM);
    assert_eq!(le(&bytes, 80), 1);
    // SampleDesc.Quality and Usage stay at their defaults.
    assert_eq!(le(&bytes, 84), 0);
    assert_eq!(le(&bytes, 88), 0);
    assert_eq!(le(&bytes, 92), 0x28);
    // CPUAccessFlags, then MiscFlags, then TextureLayout.
    assert_eq!(le(&bytes, 96), 0);
    assert_eq!(le(&bytes, 100), 0x2);
    assert_eq!(le(&bytes, 104), 0);
    // The keyed mutex ends the record.
    assert_eq!(le(&bytes, 108), 0);
}

#[test]
fn a_record_survives_its_bytes() {
    let record = SharedTextureRecord::for_created_texture(
        name("port"),
        2048,
        1024,
        DXGI_FORMAT_B8G8R8A8_UNORM,
    );
    assert_eq!(SharedTextureRecord::from_bytes(&record.to_bytes()), record);
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
    assert_eq!(record.extent(), (2048, 1024));
    assert_eq!(record.format(), 87);
    assert_eq!(record.bind_flags(), 0x28);
    assert_eq!(record.misc_flags(), 0x802);
    assert_eq!(record.keyed_mutex(), 0x1234);
    assert!(record.describes_texture(2048, 1024));
}

#[test]
fn a_record_of_another_shape_does_not_describe_the_texture() {
    let texture = || {
        SharedTextureRecord::for_created_texture(name("port"), 640, 480, DXGI_FORMAT_B8G8R8A8_UNORM)
            .to_bytes()
    };
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

#[test]
fn a_service_name_is_terminated_and_names_its_texture() {
    let name = service_name(0x1234, 0x77, 0xabcd_ef01_2345_6789);
    let text = CStr::from_bytes_until_nul(&name)
        .expect("terminated")
        .to_str()
        .expect("ASCII");
    assert_eq!(text, "MTLD3D_shared_000012340000000000000077ef0123456789");
    assert!(text.len() < SERVICE_NAME_LEN);
}

#[test]
fn the_widest_inputs_still_fit_the_field() {
    let name = service_name(u32::MAX, u64::MAX, u64::MAX);
    let text = CStr::from_bytes_until_nul(&name).expect("terminated");
    assert_eq!(text.to_bytes().len(), 50);
}

#[test]
fn two_textures_of_one_process_get_two_names() {
    assert_ne!(service_name(7, 1, 99), service_name(7, 2, 99));
    assert_ne!(service_name(7, 1, 99), service_name(8, 1, 99));
    assert_ne!(service_name(7, 1, 99), service_name(7, 1, 100));
}
