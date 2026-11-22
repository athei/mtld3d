use super::*;

fn handle(raw: u64) -> MetalHandle<MTLTextureKind> {
    // SAFETY: an opaque test value nothing dereferences.
    unsafe { MetalHandle::new(raw) }
}

const THREE_QUARTERS: RenderScale = RenderScale::from_percent(75);

fn surface(format: PixelFormat) -> DepthSurface {
    DepthSurface {
        handle: handle(0x10),
        texture_size: (480, 360),
        logical_size: (640, 480),
        format,
        scale: THREE_QUARTERS,
    }
}

#[test]
fn only_an_unscaled_target_over_a_scaled_surface_is_mixed() {
    let id = RenderScale::IDENTITY;
    assert_eq!(pairing(id, false, THREE_QUARTERS, 1), Pairing::Mixed);
    assert_eq!(
        pairing(id, false, THREE_QUARTERS, 4),
        Pairing::MixedMultisampled
    );
    assert_eq!(pairing(id, false, id, 1), Pairing::Shared, "identity");
    assert_eq!(
        pairing(THREE_QUARTERS, false, THREE_QUARTERS, 1),
        Pairing::Shared,
        "both scaled"
    );
    assert_eq!(
        pairing(THREE_QUARTERS, false, id, 1),
        Pairing::Shared,
        "the reverse pairing"
    );
    assert_eq!(
        pairing(id, true, THREE_QUARTERS, 1),
        Pairing::Shared,
        "a NULL render target 0 follows the depth surface"
    );
}

#[test]
fn a_span_carries_the_surface_into_the_plane_and_back() {
    let mut planes = DepthPlane::new();
    assert!(
        planes
            .open(surface(PixelFormat::Depth32FloatStencil8))
            .is_none()
    );
    assert!(
        planes
            .replace_plane(handle(0x20), (640, 480), PixelFormat::Depth32FloatStencil8)
            .is_none()
    );
    let into = planes
        .open(surface(PixelFormat::Depth32FloatStencil8))
        .expect("a plane of the surface's size and format");
    assert!(planes.span_open());
    assert_eq!(into.source.raw(), 0x10);
    assert_eq!(into.source_size, (480, 360));
    assert_eq!(into.destination.raw(), 0x20);
    assert_eq!(into.destination_size, (640, 480));
    assert_eq!(into.source_samples, 1);
    assert!(
        planes
            .open(surface(PixelFormat::Depth32FloatStencil8))
            .is_none(),
        "one span at a time"
    );

    let (back, rebind) = planes.close().expect("the open span");
    assert!(!planes.span_open());
    assert_eq!(back.source.raw(), 0x20);
    assert_eq!(back.source_size, (640, 480));
    assert_eq!(back.destination.raw(), 0x10);
    assert_eq!(back.destination_size, (480, 360));
    assert_eq!(back.destination_format, PixelFormat::Depth32FloatStencil8);
    assert_eq!(rebind.handle.raw(), 0x10);
    assert_eq!(rebind.scale, THREE_QUARTERS);
    assert!(planes.close().is_none(), "closed once");
}

#[test]
fn a_plane_of_another_size_or_format_serves_no_surface() {
    let mut planes = DepthPlane::new();
    planes.replace_plane(handle(0x20), (800, 600), PixelFormat::Depth32Float);
    assert!(
        planes
            .plane_for(&surface(PixelFormat::Depth32Float))
            .is_none()
    );
    let old = planes.replace_plane(handle(0x30), (640, 480), PixelFormat::Depth32Float);
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
        planes
            .plane_for(&surface(PixelFormat::Depth32Float))
            .map(MetalHandle::raw),
        Some(0x30)
    );
}

#[test]
fn an_abandoned_span_carries_nothing_back() {
    let mut planes = DepthPlane::new();
    planes.replace_plane(handle(0x20), (640, 480), PixelFormat::Depth32Float);
    assert!(planes.open(surface(PixelFormat::Depth32Float)).is_some());
    planes.abandon_span();
    assert!(planes.close().is_none());
    assert_eq!(planes.take_plane().map(MetalHandle::raw), Some(0x20));
    assert!(planes.take_plane().is_none());
}
