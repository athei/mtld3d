use mtld3d_shared::mtl::{Swizzle, TextureCreateFlags, TextureUsage};

use super::*;
use crate::ids::TextureId;

fn handle(raw: u64) -> MetalHandle<MTLTextureKind> {
    // SAFETY: an opaque test value nothing dereferences.
    unsafe { MetalHandle::new(raw) }
}

const THREE_QUARTERS: RenderScale = RenderScale::from_percent(75);

fn surface(format: PixelFormat) -> DepthSurface {
    sized_surface((640, 480), format)
}

fn sized_surface(logical_size: (u32, u32), format: PixelFormat) -> DepthSurface {
    DepthSurface {
        texture: DepthTexture::Surface(handle(0x10)),
        texture_size: THREE_QUARTERS.size(logical_size),
        logical_size,
        format,
        scale: THREE_QUARTERS,
    }
}

/// An INTZ texture at 640x480 under three quarters, three levels, as `texture_info` builds it.
fn intz(id: TextureId) -> TextureInfo {
    let (width, height) = THREE_QUARTERS.size((640, 480));
    TextureInfo {
        texture_id: id,
        d3d_format: mtld3d_types::D3DFMT_INTZ,
        width,
        height,
        depth: 1,
        levels: 3,
        pixel_format: PixelFormat::Depth32FloatStencil8,
        create_flags: TextureCreateFlags::empty(),
        swizzle: [Swizzle::Zero; 4],
        usage_flags: TextureUsage::DEPTH_STENCIL,
    }
}

/// Level `level` of [`intz`], as the device reads it for a span.
fn intz_level(id: TextureId, level: u32) -> DepthSurface {
    let info = intz(id);
    let texture_size = ((info.width >> level).max(1), (info.height >> level).max(1));
    DepthSurface {
        texture: DepthTexture::Level(info, level),
        texture_size,
        logical_size: (640 >> level, 480 >> level),
        format: PixelFormat::Depth32FloatStencil8,
        scale: THREE_QUARTERS,
    }
}

/// The handle a transfer end names; panics on a texture end.
fn end_handle(end: &StretchKind) -> u64 {
    match end {
        StretchKind::DepthStencil(handle) | StretchKind::Backbuffer(handle) => handle.raw(),
        StretchKind::Texture(_) => panic!("a texture end where a handle was expected"),
    }
}

/// The texture a transfer end names; panics on a handle end.
fn end_texture(end: &StretchKind) -> TextureId {
    match end {
        StretchKind::Texture(info) => info.texture_id,
        StretchKind::DepthStencil(_) | StretchKind::Backbuffer(_) => {
            panic!("a handle end where a texture was expected")
        }
    }
}

fn targets(rt0_scale: RenderScale, depth_scale: RenderScale, samples: (u8, u8)) -> PassTargets {
    PassTargets {
        rt0_scale,
        rt0_is_null: false,
        rt0_samples: samples.0,
        depth_scale,
        depth_samples: samples.1,
    }
}

#[test]
fn only_an_unscaled_target_over_a_scaled_surface_is_mixed() {
    let id = RenderScale::IDENTITY;
    assert_eq!(
        pairing(&targets(id, THREE_QUARTERS, (1, 1))),
        Pairing::Mixed
    );
    assert_eq!(
        pairing(&targets(id, THREE_QUARTERS, (4, 4))),
        Pairing::MixedMultisampled
    );
    assert_eq!(
        pairing(&targets(id, id, (1, 1))),
        Pairing::Shared,
        "identity"
    );
    assert_eq!(
        pairing(&targets(THREE_QUARTERS, THREE_QUARTERS, (1, 1))),
        Pairing::Shared,
        "both scaled"
    );
    assert_eq!(
        pairing(&targets(THREE_QUARTERS, id, (1, 1))),
        Pairing::Unserved,
        "a scaled target over an unscaled surface"
    );
    assert_eq!(
        pairing(&targets(id, THREE_QUARTERS, (4, 1))),
        Pairing::Shared,
        "a pass whose samples disagree leaves the depth surface out"
    );
    let mut null = targets(id, THREE_QUARTERS, (1, 1));
    null.rt0_is_null = true;
    assert_eq!(
        pairing(&null),
        Pairing::Shared,
        "a NULL render target 0 follows the depth surface"
    );
}

/// Two different scales pair only when they are equal, whichever kind made them.
#[test]
fn two_different_scales_are_unserved() {
    let half = RenderScale::from_percent(50);
    // The panel fit of a 16-inch "More Space" screen, a per-axis ratio.
    let fit =
        crate::render_scale::panel_fit(true, (4112, 2658), Some((3456, 2234)), Some((4112, 2658)));
    assert_eq!(
        pairing(&targets(THREE_QUARTERS, half, (1, 1))),
        Pairing::Unserved
    );
    assert_eq!(
        pairing(&targets(half, THREE_QUARTERS, (1, 1))),
        Pairing::Unserved
    );
    assert_eq!(
        pairing(&targets(fit, THREE_QUARTERS, (1, 1))),
        Pairing::Unserved
    );
    assert_eq!(
        pairing(&targets(THREE_QUARTERS, fit, (4, 4))),
        Pairing::Unserved,
        "a scaled target is unserved whatever the samples"
    );
    assert_eq!(
        pairing(&targets(fit, fit, (1, 1))),
        Pairing::Shared,
        "one per-axis scale"
    );
    assert_eq!(
        pairing(&targets(RenderScale::IDENTITY, fit, (1, 1))),
        Pairing::Mixed,
        "an unscaled target over the fitted surface"
    );
    // A ratio reduces, so the same scale built two ways pairs as one.
    let three_quarters = RenderScale::from_wire(&mtld3d_shared::render_size::RenderScaleWire {
        width_num: 75,
        width_den: 100,
        height_num: 3,
        height_den: 4,
    });
    assert_eq!(
        pairing(&targets(three_quarters, THREE_QUARTERS, (1, 1))),
        Pairing::Shared
    );
    let mut null = targets(THREE_QUARTERS, half, (1, 1));
    null.rt0_is_null = true;
    assert_eq!(pairing(&null), Pairing::Shared, "a NULL render target 0");
    assert_eq!(
        pairing(&targets(THREE_QUARTERS, half, (1, 4))),
        Pairing::Shared,
        "samples that disagree leave the depth surface out"
    );
}

#[test]
fn a_span_carries_the_surface_into_the_plane_and_back() {
    let mut planes = DepthPlane::new();
    let format = PixelFormat::Depth32FloatStencil8;
    assert!(planes.open(surface(format)).is_none());
    assert!(
        planes
            .replace_plane(handle(0x20), &surface(format))
            .is_none()
    );
    let into = planes
        .open(surface(format))
        .expect("a plane of the surface's size and format");
    assert!(planes.span_open());
    assert_eq!(end_handle(&into.source), 0x10);
    assert_eq!((into.source_level, into.destination_level), (0, 0));
    assert_eq!(into.source_size, (480, 360));
    assert_eq!(end_handle(&into.destination), 0x20);
    assert_eq!(into.destination_size, (640, 480));
    assert_eq!(into.source_samples, 1);
    assert!(planes.open(surface(format)).is_none(), "one span at a time");

    let (back, rebind) = planes.close().expect("the open span");
    assert!(!planes.span_open());
    assert_eq!(end_handle(&back.source), 0x20);
    assert_eq!(back.source_size, (640, 480));
    assert_eq!(end_handle(&back.destination), 0x10);
    assert_eq!((back.source_level, back.destination_level), (0, 0));
    assert_eq!(back.destination_size, (480, 360));
    assert_eq!(back.destination_format, format);
    assert!(!rebind.is_sampleable());
    let DepthBinding::Eager(rebound, extent, scale) = rebind.binding() else {
        panic!("a depth surface binds again by its handle");
    };
    assert_eq!(
        (rebound.raw(), extent, scale),
        (0x10, (480, 360), THREE_QUARTERS)
    );
    assert!(planes.close().is_none(), "closed once");
}

#[test]
fn a_span_over_a_depth_texture_level_names_the_texture_and_its_level() {
    let id = TextureId::new_unique();
    let mut planes = DepthPlane::new();
    assert!(
        planes
            .replace_plane(handle(0x20), &intz_level(id, 1))
            .is_none()
    );
    assert!(
        planes.plane_for(&intz_level(id, 0)).is_none(),
        "a plane takes the level's reported size"
    );
    let into = planes
        .open(intz_level(id, 1))
        .expect("a plane of the level's size and format");
    assert_eq!(end_texture(&into.source), id);
    assert_eq!(into.source_level, 1);
    assert_eq!(into.source_size, (240, 180), "the Metal level's extent");
    assert_eq!(end_handle(&into.destination), 0x20);
    assert_eq!(into.destination_level, 0);
    assert_eq!(into.destination_size, (320, 240));

    let (back, rebind) = planes.close().expect("the open span");
    assert_eq!(end_handle(&back.source), 0x20);
    assert_eq!(back.source_level, 0);
    assert_eq!(end_texture(&back.destination), id);
    assert_eq!(back.destination_level, 1);
    assert_eq!(back.destination_size, (240, 180));
    assert!(rebind.is_sampleable());
    let DepthBinding::Lazy(texture, level, scale) = rebind.binding() else {
        panic!("a texture level binds again through its texture's info");
    };
    assert_eq!((texture.texture_id, level, scale), (id, 1, THREE_QUARTERS));
}

#[test]
fn a_level_past_a_shortened_chain_transfers_the_last_metal_level() {
    // The reported level the binding names is kept; the transfer addresses
    // the Metal level it lands on, as every other texture-level record does.
    let id = TextureId::new_unique();
    let mut surface = intz_level(id, 2);
    if let DepthTexture::Level(info, _) = &mut surface.texture {
        info.levels = 2;
    }
    let mut planes = DepthPlane::new();
    planes.replace_plane(handle(0x20), &surface);
    let into = planes.open(surface).expect("a plane");
    assert_eq!(into.source_level, 1);
    let (back, rebind) = planes.close().expect("the open span");
    assert_eq!(back.destination_level, 1);
    let DepthBinding::Lazy(_, level, _) = rebind.binding() else {
        panic!("a texture level");
    };
    assert_eq!(level, 2);
}

#[test]
fn a_plane_of_another_size_or_format_serves_no_surface() {
    let mut planes = DepthPlane::new();
    let format = PixelFormat::Depth32Float;
    planes.replace_plane(handle(0x20), &sized_surface((800, 600), format));
    assert!(planes.plane_for(&surface(format)).is_none());
    let old = planes.replace_plane(handle(0x30), &surface(format));
    assert_eq!(
        old.map(MetalHandle::raw),
        Some(0x20),
        "the replaced plane retires"
    );
    assert!(
        planes
            .plane_for(&surface(PixelFormat::Depth32FloatStencil8))
            .is_none(),
        "format"
    );
    assert_eq!(
        planes.plane_for(&surface(format)).map(MetalHandle::raw),
        Some(0x30)
    );
}

#[test]
fn a_refused_plane_is_remembered_until_a_plane_is_installed_or_taken() {
    let mut planes = DepthPlane::new();
    let format = PixelFormat::Depth32Float;
    planes.note_refused(&surface(format));
    assert!(planes.refused_for(&surface(format)));
    assert!(
        !planes.refused_for(&sized_surface((800, 600), format)),
        "another size is worth a try"
    );
    assert!(
        !planes.refused_for(&surface(PixelFormat::Depth32FloatStencil8)),
        "another format is worth a try"
    );
    planes.replace_plane(handle(0x20), &sized_surface((800, 600), format));
    assert!(!planes.refused_for(&surface(format)), "a new plane");
    planes.note_refused(&surface(format));
    assert_eq!(planes.take_plane().map(MetalHandle::raw), Some(0x20));
    assert!(!planes.refused_for(&surface(format)), "a taken plane");
}

#[test]
fn an_abandoned_span_carries_nothing_back() {
    let mut planes = DepthPlane::new();
    let format = PixelFormat::Depth32Float;
    planes.replace_plane(handle(0x20), &surface(format));
    assert!(planes.open(surface(format)).is_some());
    planes.abandon_span();
    assert!(planes.close().is_none());
    assert_eq!(planes.take_plane().map(MetalHandle::raw), Some(0x20));
    assert!(planes.take_plane().is_none());
}

#[test]
fn a_classification_is_kept_until_the_targets_are_forgotten() {
    let mut planes = DepthPlane::new();
    let format = PixelFormat::Depth32FloatStencil8;
    assert!(
        !planes.targets_known(),
        "a new device has classified nothing"
    );

    planes.note_targets(None);
    assert!(planes.targets_known());
    assert!(
        planes.mixed_has_stencil().is_none(),
        "no plane serves the pair"
    );
    assert!(planes.rules_out_span(), "targets on one grid open nothing");

    planes.forget_targets();
    assert!(!planes.targets_known(), "a changed target asks again");
    assert!(
        !planes.rules_out_span(),
        "unclassified targets are asked, not assumed to share a grid"
    );

    planes.note_targets(Some(&surface(format)));
    assert_eq!(planes.mixed_has_stencil(), Some(true));
    assert!(!planes.rules_out_span(), "a mixed pair may open a span");
    planes.note_targets(Some(&surface(PixelFormat::Depth32Float)));
    assert_eq!(planes.mixed_has_stencil(), Some(false));

    planes.replace_plane(handle(0x20), &surface(format));
    assert!(planes.open(surface(format)).is_some());
    assert!(planes.rules_out_span(), "an open span opens no second one");
    assert!(planes.close().is_some());
    assert!(
        planes.targets_known() && !planes.rules_out_span(),
        "the targets a span closed over are still the mixed pair"
    );

    planes.forget_targets();
    assert!(
        planes.mixed_has_stencil().is_none(),
        "a forgotten answer serves no draw"
    );
}

#[test]
fn targets_kept_as_mixed_and_found_unserved_rule_out_the_next_span() {
    // The device notes "no plane serves the pair" over a kept mixed answer
    // when the targets it reads again no longer pair as a pass the plane
    // serves, so the draws after it stop at the kept answer instead of
    // reading the surface again.
    let mut planes = DepthPlane::new();
    planes.note_targets(Some(&surface(PixelFormat::Depth32FloatStencil8)));
    assert!(!planes.rules_out_span());

    planes.note_targets(None);
    assert!(planes.targets_known(), "the answer stays kept");
    assert!(
        planes.mixed_has_stencil().is_none(),
        "no plane serves the pair"
    );
    assert!(planes.rules_out_span(), "the next draw opens nothing");
}

#[test]
fn a_dropped_frame_and_a_taken_plane_forget_the_targets() {
    let mut planes = DepthPlane::new();
    let format = PixelFormat::Depth32Float;
    planes.replace_plane(handle(0x20), &surface(format));

    planes.note_targets(Some(&surface(format)));
    planes.abandon_span();
    assert!(!planes.targets_known(), "the dropped frame's bindings");

    planes.note_targets(None);
    assert_eq!(planes.take_plane().map(MetalHandle::raw), Some(0x20));
    assert!(
        !planes.targets_known(),
        "Reset and release replace the targets"
    );
}
