use std::sync::mpsc;

use mtld3d_core::{
    pipeline_state::{
        ExtraColorAttachments, PipelineAttachFlags, PipelineRsBits, PipelineSnapshot, StreamLayout,
    },
    shader_cache::ShaderRecordRef,
    shader_key::CachedKind,
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

use super::{
    BufferGpuState, DestroyKind, StageLibHandles, WarmCache, cached_buffer_handles,
    destroy_resources_bulk, take_released_buffer,
};

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

#[test]
fn payload_rotation_reuses_warm_scratch_chunks_across_the_submit_thread() {
    use std::collections::BTreeSet;

    use mtld3d_core::{passes::PassState, scratch::ScratchArena};

    use super::{FramePayload, SUBMIT_PAYLOAD_CAP};

    const FRAMES: usize = 48;
    const WARM_FRAMES: usize = 8;
    let (work_tx, work_rx) = mpsc::sync_channel::<FramePayload>(1);
    let (return_tx, return_rx) = mpsc::channel::<FramePayload>();
    // Stands in for the submit thread: it reads a payload and hands the whole of it back.
    let submit = std::thread::spawn(move || {
        while let Ok(payload) = work_rx.recv() {
            assert!(
                payload.scratch.bytes_used() > 0,
                "the frame's snapshots rode along"
            );
            if return_tx.send(payload).is_err() {
                break;
            }
        }
    });
    let mut live = ScratchArena::new();
    let mut blits = Vec::new();
    let mut pass_state = PassState::new();
    let mut pool: Vec<FramePayload> = Vec::new();
    let mut created = 0_u32;
    let mut seen = BTreeSet::new();
    let snapshot = vec![0x5a_u8; 30 * 1024];
    for frame in 0..FRAMES {
        // `begin_frame`, then two snapshot decodes' worth of storage and a constant block,
        // which together need two chunks.
        live.clear();
        for _ in 0..3 {
            live.alloc(&snapshot);
        }
        let chunks = live
            .allocation_ranges()
            .map(|(address, _)| address)
            .collect::<Vec<_>>();
        assert_eq!(chunks.len(), 2);
        if frame >= WARM_FRAMES {
            assert!(
                chunks.iter().all(|address| seen.contains(address)),
                "frame {frame} recorded into a chunk no earlier frame had"
            );
        }
        seen.extend(chunks);
        // `acquire_clean_payload`: reclaim what came back, reuse, create up to the cap, wait.
        while let Ok(mut returned) = return_rx.try_recv() {
            returned.clear(&mut pass_state);
            pool.push(returned);
        }
        let mut payload = pool.pop().unwrap_or_else(|| {
            if created < SUBMIT_PAYLOAD_CAP {
                created += 1;
                FramePayload::default()
            } else {
                let mut returned = return_rx.recv().unwrap();
                returned.clear(&mut pass_state);
                returned
            }
        });
        payload.adopt_frame_buffers(&mut live, &mut blits);
        work_tx.send(payload).unwrap();
    }
    drop(work_tx);
    submit.join().unwrap();
    while let Ok(mut returned) = return_rx.try_recv() {
        returned.clear(&mut pass_state);
        pool.push(returned);
    }
    // The encoder's arena and at most two payloads' arenas exist, each warmed to two chunks.
    let arenas = pool.iter().map(|payload| &payload.scratch).chain([&live]);
    let chunk_total = arenas.map(ScratchArena::chunk_count).sum::<u32>();
    assert_eq!(pool.len(), usize::try_from(created).unwrap());
    assert_eq!(chunk_total, 2 * (created + 1));
    assert_eq!(seen.len(), usize::try_from(chunk_total).unwrap());
}

/// An opaque buffer handle, never dereferenced.
fn buffer(raw: u64) -> MetalHandle<mtld3d_shared::mtl_handle::MTLBufferKind> {
    // SAFETY: tests; opaque values never dereferenced.
    unsafe { MetalHandle::new(raw) }
}

/// A `Staged` entry whose CPU backing was released, the shape a `WRITEONLY` DEFAULT buffer reaches.
fn staged(device_buffer: u64, last_submit_seq: u64) -> BufferGpuState {
    BufferGpuState {
        mtl_buffer: MetalHandle::NULL,
        device_buffer: buffer(device_buffer),
        is_staged: true,
        backing_ptr: 0,
        length: 4096,
        backing_generation: 0,
        last_submit_seq,
    }
}

/// A `Direct` entry wrapping a CPU backing.
fn direct(wrapper: u64) -> BufferGpuState {
    BufferGpuState {
        mtl_buffer: buffer(wrapper),
        device_buffer: MetalHandle::NULL,
        is_staged: false,
        backing_ptr: 0x10_0000,
        length: 4096,
        backing_generation: 1,
        last_submit_seq: 3,
    }
}

/// A released `Staged` buffer with no backing hands its device buffer to the retention queue.
///
/// A `D3DPOOL_DEFAULT` `D3DUSAGE_WRITEONLY` buffer drops its CPU copy after
/// its upload, so its release sends no backing through the retention intake,
/// the one path that took the entry out before; its Private device buffer
/// leaked on every release, and every DEFAULT buffer is recreated around a
/// `Reset`. The destroy waits for the later of the frame that released the
/// buffer and the last frame that drew with it.
#[test]
fn a_released_staged_buffer_retires_its_device_buffer() {
    use mtld3d_core::ids::BufferId;

    let mut cache = rustc_hash::FxHashMap::default();
    let (released, kept, wrapped) = (
        BufferId::from_raw(1),
        BufferId::from_raw(2),
        BufferId::from_raw(3),
    );
    cache.insert(released, staged(0xD100, 7));
    cache.insert(kept, staged(0xD200, 7));
    cache.insert(wrapped, direct(0xC300));

    let entry = take_released_buffer(&mut cache, released, 5).expect("the device buffer retires");
    assert!(matches!(entry.kind, DestroyKind::Buffer));
    assert_eq!(entry.handle, 0xD100);
    assert_eq!(entry.seq, 7, "gated on the last frame that drew with it");
    assert!(entry.page_box.is_none(), "no CPU backing rides along");
    assert!(!cache.contains_key(&released));
    assert!(
        take_released_buffer(&mut cache, released, 9).is_none(),
        "a second destroy finds nothing"
    );
    assert_eq!(
        take_released_buffer(&mut cache, kept, 9).map(|entry| entry.seq),
        Some(9),
        "a release after the last draw waits for its own frame"
    );
    assert!(
        take_released_buffer(&mut cache, wrapped, 9).is_none(),
        "a Direct wrapper retires with its backing, not here"
    );
    assert!(cache.contains_key(&wrapped));
}

/// Shutdown collects the `Staged` device buffers as well as the `Direct` wrappers.
#[test]
fn shutdown_collects_staged_device_buffers_and_direct_wrappers() {
    use mtld3d_core::ids::BufferId;

    let mut cache = rustc_hash::FxHashMap::default();
    cache.insert(BufferId::from_raw(1), staged(0xD100, 1));
    cache.insert(BufferId::from_raw(2), direct(0xC200));
    // A staged entry whose warmup create failed holds no buffer yet.
    cache.insert(BufferId::from_raw(3), staged(0, 1));

    let mut handles = cached_buffer_handles(&cache);
    handles.sort_unstable();
    assert_eq!(handles, [0xC200, 0xD100]);
}
