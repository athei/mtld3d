use super::*;

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
        handle: handle(0x10),
        texture_size: (
            THREE_QUARTERS.dimension(logical_size.0),
            THREE_QUARTERS.dimension(logical_size.1),
        ),
        logical_size,
        format,
        scale: THREE_QUARTERS,
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
        Pairing::Reverse,
        "the reverse pairing"
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
    assert_eq!(into.source.raw(), 0x10);
    assert_eq!(into.source_size, (480, 360));
    assert_eq!(into.destination.raw(), 0x20);
    assert_eq!(into.destination_size, (640, 480));
    assert_eq!(into.source_samples, 1);
    assert!(planes.open(surface(format)).is_none(), "one span at a time");

    let (back, rebind) = planes.close().expect("the open span");
    assert!(!planes.span_open());
    assert_eq!(back.source.raw(), 0x20);
    assert_eq!(back.source_size, (640, 480));
    assert_eq!(back.destination.raw(), 0x10);
    assert_eq!(back.destination_size, (480, 360));
    assert_eq!(back.destination_format, format);
    assert_eq!(rebind.handle.raw(), 0x10);
    assert_eq!(rebind.scale, THREE_QUARTERS);
    assert!(planes.close().is_none(), "closed once");
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
