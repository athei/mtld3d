use std::sync::mpsc;

use mtld3d_core::{
    pipeline_state::{
        ExtraColorAttachments, PipelineAttachFlags, PipelineRsBits, PipelineSnapshot, StreamLayout,
    },
    shader_cache::{CachedKind, ShaderRecordRef},
};
use mtld3d_shared::{MetalHandle, mtl::PixelFormat};
use mtld3d_types::MAX_STREAMS;
use objc2::{
    rc::Retained,
    runtime::{NSObjectProtocol, ProtocolObject},
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCreateSystemDefaultDevice, MTLDevice, MTLFunction, MTLLibrary, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState,
};

use super::{DestroyKind, StageLibHandles, WarmCache, destroy_resources_bulk};

struct NativeObjects {
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    function: Retained<ProtocolObject<dyn MTLFunction>>,
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

impl NativeObjects {
    fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device for ownership test");
        let source = NSString::from_str(
            "#include <metal_stdlib>\nusing namespace metal;\nvertex void ownership_probe(uint id [[vertex_id]]) {}",
        );
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .expect("library");
        let function = library
            .newFunctionWithName(&NSString::from_str("ownership_probe"))
            .expect("function");
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&function));
        descriptor.setRasterizationEnabled(false);
        let pipeline = device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .expect("pipeline");
        Self {
            library,
            function,
            pipeline,
        }
    }

    fn counts(&self) -> [usize; 3] {
        [
            self.pipeline.retainCount(),
            self.function.retainCount(),
            self.library.retainCount(),
        ]
    }

    fn warm(&self) -> WarmCache {
        // SAFETY: each handle owns the real object's extra retain transferred by into_raw.
        let library = unsafe { MetalHandle::new(Retained::into_raw(self.library.clone()) as u64) };
        // SAFETY: this handle owns the real function's transferred extra retain.
        let func = unsafe { MetalHandle::new(Retained::into_raw(self.function.clone()) as u64) };
        // SAFETY: this handle owns the real pipeline's transferred extra retain.
        let pipeline =
            unsafe { MetalHandle::new(Retained::into_raw(self.pipeline.clone()) as u64) };
        let snapshot = PipelineSnapshot {
            vdecl_hash: 0,
            vs_fn: func,
            ps_fn: MetalHandle::NULL,
            stream_layouts: [StreamLayout::UNUSED; MAX_STREAMS as usize],
            color_format: PixelFormat::Bgra8Unorm,
            attach: PipelineAttachFlags::empty(),
            rs: PipelineRsBits::default(),
            extra: ExtraColorAttachments::NONE,
            ps_color_out_mask: 0,
            sample_count: 1,
        };
        WarmCache {
            libraries: vec![(
                ShaderRecordRef::new(CachedKind::Sm2Vs, 1),
                StageLibHandles { library, func },
            )],
            pipelines: vec![(
                mtld3d_core::pipeline_state::key_from_snapshot(&snapshot, &[]),
                pipeline,
            )],
            no_color_siblings: vec![(pipeline.raw(), pipeline)],
        }
    }
}

#[test]
fn failed_prewarm_send_releases_native_payload() {
    objc2::rc::autoreleasepool(|_| {
        let objects = NativeObjects::new();
        let baseline = objects.counts();
        let (sender, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        let result = sender.send(objects.warm());
        assert!(result.is_err());
        drop(result);
        assert_eq!(
            objects.counts(),
            baseline,
            "failed delivery releases all three retains"
        );
    });
}

#[test]
fn unread_prewarm_payload_releases_native_objects() {
    objc2::rc::autoreleasepool(|_| {
        let objects = NativeObjects::new();
        let baseline = objects.counts();
        let (sender, receiver) = mpsc::sync_channel(1);
        assert!(sender.send(objects.warm()).is_ok());
        drop(receiver);
        assert_eq!(
            objects.counts(),
            baseline,
            "queued payload is still an owner"
        );
    });
}

#[test]
fn adopted_prewarm_retains_transfer_once_and_release_in_dependency_order() {
    objc2::rc::autoreleasepool(|_| {
        let objects = NativeObjects::new();
        let baseline = objects.counts();
        let mut source = objects.warm();
        let retained = objects.counts();
        let mut destination = WarmCache {
            libraries: std::mem::take(&mut source.libraries),
            pipelines: std::mem::take(&mut source.pipelines),
            no_color_siblings: std::mem::take(&mut source.no_color_siblings),
        };
        drop(source);
        assert_eq!(
            objects.counts(),
            retained,
            "adoption leaves the old owner empty"
        );
        let mut releases = Vec::new();
        destination.release_with(|kind, handles| {
            releases.push((kind, handles.to_vec()));
            destroy_resources_bulk(kind, handles);
        });
        assert_eq!(
            releases
                .iter()
                .map(|(kind, handles)| (*kind, handles.len()))
                .collect::<Vec<_>>(),
            vec![
                (DestroyKind::RenderPipeline, 1),
                (DestroyKind::ShaderFunction, 1),
                (DestroyKind::ShaderLibrary, 1),
            ]
        );
        drop(destination);
        assert_eq!(
            objects.counts(),
            baseline,
            "aliases and empty owners release nothing extra"
        );
    });
}

#[test]
fn native_shader_build_resets_empty_inputs_and_retains_outputs_after_pool_drain() {
    use mtld3d_shared::{mtl::StageTag, perf::ShaderTimings};

    use crate::metal::handle::IntoRetained;

    objc2::rc::autoreleasepool(|_| {
        for (source, entry) in [("", "probe"), ("invalid MSL", "")] {
            let mut timings = ShaderTimings {
                preparation_ns: u64::MAX,
                library_ns: u64::MAX,
                function_ns: u64::MAX,
            };
            assert!(
                super::compile_stage_library(
                    MetalHandle::NULL,
                    StageTag::Vertex,
                    source,
                    entry,
                    &mut timings,
                )
                .is_none()
            );
            assert_eq!(
                (
                    timings.preparation_ns,
                    timings.library_ns,
                    timings.function_ns
                ),
                (0, 0, 0),
                "empty input resets all phases before native compilation"
            );
        }

        let device = MTLCreateSystemDefaultDevice().expect("Metal device for shader lifetime test");
        // SAFETY: device owns this live Metal device until the build completes.
        let device_handle = unsafe { MetalHandle::new(Retained::as_ptr(&device) as u64) };
        let source = "#include <metal_stdlib>\nusing namespace metal;\nvertex float4 probe(uint id [[vertex_id]]) { return float4(float(id), 0, 0, 1); }";
        let handles = super::compile_stage_library(
            device_handle,
            StageTag::Vertex,
            source,
            "probe",
            &mut ShaderTimings::new(),
        )
        .expect("native shader build succeeds");
        // The build's inner pool has drained; only canonical output retains remain.
        let library = handles
            .library
            .into_retained()
            .expect("live library output");
        let function = handles.func.into_retained().expect("live function output");
        assert_eq!(library.label().expect("library label").to_string(), "probe");
        assert_eq!(function.name().to_string(), "probe");
        drop(function);
        drop(library);
        destroy_resources_bulk(DestroyKind::ShaderFunction, &[handles.func.raw()]);
        destroy_resources_bulk(DestroyKind::ShaderLibrary, &[handles.library.raw()]);
    });
}
