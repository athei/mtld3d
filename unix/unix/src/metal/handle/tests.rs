//! Unit tests for the handle conversions, and for which of them touch the refcount.

use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDepthStencilDescriptor, MTLLanguageVersion,
    MTLPixelFormat, MTLRenderPipelineDescriptor, MTLResourceOptions, MTLSamplerDescriptor,
    MTLTextureDescriptor,
};

use super::*;

/// A trivial vertex plus fragment pair, the cheapest real render pipeline state.
const PIPELINE_MSL: &str = "
using namespace metal;
struct VSOut { float4 position [[position]]; };
vertex VSOut handle_test_vs() { return VSOut{ float4(0.0, 0.0, 0.0, 1.0) }; }
fragment half4 handle_test_ps() { return half4(0.0); }
";

/// A null handle borrows nothing, exactly as it retains nothing.
///
/// One assertion per borrowable kind: the filter is what keeps a slot the PE
/// side left empty from being read as an object address.
#[test]
fn borrow_retained_filters_the_null_handle() {
    // SAFETY: the null handle addresses no object, so no retain has to
    // outlive the (absent) reference the call returns.
    assert!(unsafe { MetalHandle::<MTLBufferKind>::NULL.borrow_retained() }.is_none());
    // SAFETY: as above.
    assert!(unsafe { MetalHandle::<MTLTextureKind>::NULL.borrow_retained() }.is_none());
    // SAFETY: as above.
    assert!(unsafe { MetalHandle::<MTLSamplerStateKind>::NULL.borrow_retained() }.is_none());
    // SAFETY: as above.
    assert!(unsafe { MetalHandle::<MTLDepthStencilStateKind>::NULL.borrow_retained() }.is_none());
    // SAFETY: as above.
    assert!(unsafe { MetalHandle::<MTLRenderPipelineStateKind>::NULL.borrow_retained() }.is_none());
    assert!(MetalHandle::<MTLBufferKind>::NULL.into_retained().is_none());
}

/// The borrow reads through the canonical retain; `into_retained` takes one of its own.
///
/// Takes the object's only retain, checks both conversions against it, and
/// releases it at the end, which is the whole life cycle a replayed command
/// and its destroy thunk put a handle through.
fn borrow_reads_through_the_canonical_retain<K: ToMetalProtocol>(
    object: Retained<ProtocolObject<K::Real>>,
) where
    MetalHandle<K>: BorrowRetained<Object = K::Real>,
{
    let canonical = Retained::into_raw(object) as u64;
    // SAFETY: `canonical` is the address of the retain `Retained::into_raw`
    // just gave up, so the handle stands for a live `id<K::Real>`.
    let handle = unsafe { MetalHandle::<K>::new(canonical) };

    // SAFETY: the canonical retain above is released only at the end of this
    // check, after the borrow and every read through it.
    let borrowed = unsafe { handle.borrow_retained() }.expect("non-null handle borrows");
    let before = borrowed.retainCount();
    assert_eq!(core::ptr::from_ref(borrowed) as u64, canonical);
    // SAFETY: as the borrow above.
    let second = unsafe { handle.borrow_retained() }.expect("non-null handle borrows");
    assert_eq!(second.retainCount(), before);

    let retained = handle.into_retained().expect("non-null handle retains");
    assert_eq!(retained.retainCount(), before + 1);
    drop(retained);
    assert_eq!(borrowed.retainCount(), before);

    // SAFETY: the handle holds the canonical retain and no copy of it is used
    // after this call; `borrowed` and `second` are dead here.
    unsafe { handle.release_retain() };
}

/// An index or vertex buffer binding borrows its `MTLBuffer`.
#[test]
fn buffer_handle_borrows_without_a_refcount_bump() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let buffer = device
        .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
        .expect("Metal buffer");
    borrow_reads_through_the_canonical_retain::<MTLBufferKind>(
        ProtocolObject::<dyn MTLBuffer>::from_retained(buffer),
    );
}

/// A texture binding, and a blit endpoint, borrow their `MTLTexture`.
#[test]
fn texture_handle_borrows_without_a_refcount_bump() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    // SAFETY: plain descriptor factory; the arguments describe a 1x1 texture.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::RGBA8Unorm,
            1,
            1,
            false,
        )
    };
    let texture = device
        .newTextureWithDescriptor(&desc)
        .expect("Metal texture");
    borrow_reads_through_the_canonical_retain::<MTLTextureKind>(
        ProtocolObject::<dyn MTLTexture>::from_retained(texture),
    );
}

/// A sampler binding borrows its `MTLSamplerState`.
#[test]
fn sampler_state_handle_borrows_without_a_refcount_bump() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let sampler = device
        .newSamplerStateWithDescriptor(&MTLSamplerDescriptor::new())
        .expect("Metal sampler state");
    borrow_reads_through_the_canonical_retain::<MTLSamplerStateKind>(ProtocolObject::<
        dyn MTLSamplerState,
    >::from_retained(sampler));
}

/// A depth-stencil bind borrows its `MTLDepthStencilState`.
#[test]
fn depth_stencil_state_handle_borrows_without_a_refcount_bump() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let state = device
        .newDepthStencilStateWithDescriptor(&MTLDepthStencilDescriptor::new())
        .expect("Metal depth-stencil state");
    borrow_reads_through_the_canonical_retain::<MTLDepthStencilStateKind>(ProtocolObject::<
        dyn MTLDepthStencilState,
    >::from_retained(
        state
    ));
}

/// A pipeline bind borrows its `MTLRenderPipelineState`.
#[test]
fn render_pipeline_state_handle_borrows_without_a_refcount_bump() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(PIPELINE_MSL), Some(&options))
        .expect("the test MSL must compile");
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(
        library
            .newFunctionWithName(&NSString::from_str("handle_test_vs"))
            .as_deref(),
    );
    desc.setFragmentFunction(
        library
            .newFunctionWithName(&NSString::from_str("handle_test_ps"))
            .as_deref(),
    );
    // SAFETY: `colorAttachments()` returns a non-null descriptor array;
    // subscript 0 is always valid.
    let color0 = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
    color0.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    let pipeline = device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .expect("Metal render pipeline state");
    borrow_reads_through_the_canonical_retain::<MTLRenderPipelineStateKind>(ProtocolObject::<
        dyn MTLRenderPipelineState,
    >::from_retained(
        pipeline
    ));
}

/// A second handle stored under a cached key is released, and the first one is kept.
///
/// Two threads that miss the same helper-cache key both build an object; the
/// loser's retain must be dropped rather than leaked with its `u64`.
#[test]
fn keep_first_releases_the_losing_handle() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let first = device
        .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
        .expect("Metal buffer");
    let second = device
        .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
        .expect("Metal buffer");
    let first_base = first.retainCount();
    let second_base = second.retainCount();
    // SAFETY: `into_raw` hands the cloned retain to the handle.
    let first_handle =
        unsafe { MetalHandle::<MTLBufferKind>::new(Retained::into_raw(first.clone()) as u64) };
    // SAFETY: as above.
    let second_handle =
        unsafe { MetalHandle::<MTLBufferKind>::new(Retained::into_raw(second.clone()) as u64) };
    let mut cache = FxHashMap::default();

    // SAFETY: `first_handle` holds the only copy of its retain.
    let kept = unsafe { keep_first(&mut cache, 7u8, first_handle) };
    assert_eq!(kept.raw(), first_handle.raw());
    // SAFETY: `second_handle` holds the only copy of its retain.
    let kept = unsafe { keep_first(&mut cache, 7u8, second_handle) };
    assert_eq!(
        kept.raw(),
        first_handle.raw(),
        "the first stored stays cached"
    );
    assert_eq!(
        second.retainCount(),
        second_base,
        "the loser's retain is released"
    );
    assert_eq!(
        first.retainCount(),
        first_base + 1,
        "the cached retain stays"
    );

    // SAFETY: the cache's copy is the only one left and is not used after.
    unsafe { first_handle.release_retain() };
    assert_eq!(first.retainCount(), first_base);
}
