use mtld3d_shared::{
    MetalHandle,
    mtl::{PixelFormat, Swizzle, TextureCreateFlags, TextureUsage},
};

use crate::{
    encoder_data::{StretchKind, StretchSurfaceFlags, StretchSurfaceInfo, TextureInfo},
    ids::TextureId,
    render_scale::RenderScale,
};

fn texture() -> TextureInfo {
    TextureInfo {
        texture_id: TextureId::new_unique(),
        d3d_format: 21,
        width: 64,
        height: 64,
        depth: 1,
        levels: 1,
        pixel_format: PixelFormat::Bgra8Unorm,
        create_flags: TextureCreateFlags::empty(),
        swizzle: [Swizzle::Blue, Swizzle::Green, Swizzle::Red, Swizzle::One],
        usage_flags: TextureUsage::empty(),
    }
}

fn surface(
    kind: StretchKind,
    flags: StretchSurfaceFlags,
    slice: Option<u32>,
) -> StretchSurfaceInfo {
    StretchSurfaceInfo {
        kind,
        width: 64,
        height: 64,
        texture_size: (64, 64),
        scale: RenderScale::IDENTITY,
        format: 21,
        mip_level: 0,
        slice,
        pool: 0,
        flags,
        autogen_texture_id: None,
        msaa: MetalHandle::NULL,
        msaa_srgb: MetalHandle::NULL,
        sample_count: 1,
    }
}

#[test]
fn stretch_surface_class_names_every_resolved_kind() {
    let cases = [
        (
            surface(
                StretchKind::DepthStencil(MetalHandle::NULL),
                StretchSurfaceFlags::IS_DEPTH_STENCIL,
                None,
            ),
            "depth-stencil surface",
        ),
        (
            surface(
                StretchKind::Backbuffer(MetalHandle::NULL),
                StretchSurfaceFlags::IS_RENDER_TARGET,
                None,
            ),
            "standalone render target",
        ),
        (
            surface(
                StretchKind::Texture(texture()),
                StretchSurfaceFlags::IS_OFFSCREEN_PLAIN_DEFAULT,
                None,
            ),
            "offscreen-plain surface",
        ),
        (
            surface(
                StretchKind::Texture(texture()),
                StretchSurfaceFlags::IS_RENDER_TARGET,
                None,
            ),
            "render-target texture level",
        ),
        (
            surface(
                StretchKind::Texture(texture()),
                StretchSurfaceFlags::IS_RENDER_TARGET,
                Some(3),
            ),
            "render-target cube face",
        ),
        (
            surface(
                StretchKind::Texture(texture()),
                StretchSurfaceFlags::empty(),
                Some(0),
            ),
            "cube face",
        ),
        (
            surface(
                StretchKind::Texture(texture()),
                StretchSurfaceFlags::empty(),
                None,
            ),
            "texture level",
        ),
    ];
    for (info, expected) in &cases {
        assert_eq!(info.class_name(), *expected);
    }
}
