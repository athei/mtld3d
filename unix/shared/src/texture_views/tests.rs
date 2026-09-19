use super::*;

#[test]
fn aliased_roles_own_each_retain_once() {
    let linear = handle(1);
    let srgb = handle(2);
    let views = TextureViews {
        linear,
        srgb,
        sample_linear: linear,
        sample_srgb: srgb,
    };
    assert_eq!(views.owned_handles().collect::<Vec<_>>(), [linear, srgb]);
    let views = TextureViews {
        linear,
        srgb: MetalHandle::NULL,
        sample_linear: linear,
        sample_srgb: MetalHandle::NULL,
    };
    assert_eq!(views.owned_handles().collect::<Vec<_>>(), [linear]);
    assert_eq!(TextureViews::EMPTY.owned_handles().count(), 0);
}

#[test]
fn distinct_sampling_views_each_own_one_retain() {
    let views = TextureViews {
        linear: handle(1),
        srgb: handle(2),
        sample_linear: handle(3),
        sample_srgb: handle(4),
    };
    assert_eq!(
        views
            .owned_handles()
            .map(MetalHandle::raw)
            .collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );
}

fn handle(raw: u64) -> MetalHandle<MTLTextureKind> {
    // SAFETY: opaque test identities used only for comparisons, never dereferenced.
    unsafe { MetalHandle::new(raw) }
}
