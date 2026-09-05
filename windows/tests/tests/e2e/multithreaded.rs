//! A device created with `D3DCREATE_MULTITHREADED`, called from two threads.
//!
//! The flag is D3D9's promise that the device and its resources may be called
//! from any thread. Without it a call from a second thread is undefined, as it
//! is on native, so no test here drives an unflagged device from two threads.

use std::sync::atomic::{AtomicBool, Ordering};

use mtld3d_tests::{Harness, HarnessConfig, Vertex, assert_pixel_eq};
use mtld3d_types::{
    D3D_OK, D3DCREATE_HARDWARE_VERTEXPROCESSING, D3DCREATE_MULTITHREADED, D3DCULL_CCW, D3DCULL_CW,
    D3DCULL_NONE, D3DFMT_A8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DGETDATA_FLUSH, D3DISSUE_BEGIN,
    D3DISSUE_END, D3DLOCK_DISCARD, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPT_TRIANGLELIST,
    D3DQUERYTYPE_OCCLUSION, D3DRS_CULLMODE, D3DRS_LIGHTING, D3DSBT_ALL, D3DUSAGE_DYNAMIC,
    D3DUSAGE_WRITEONLY,
};

const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE;
const BLUE: u32 = 0xFF00_00FF;
const GREEN: u32 = 0xFF00_FF00;

fn stride() -> u32 {
    u32::try_from(size_of::<Vertex>()).expect("vertex stride fits u32")
}

const fn solid_triangle(color: u32) -> [Vertex; 3] {
    [
        Vertex {
            x: 0.0,
            y: 0.5,
            z: 0.5,
            color,
        },
        Vertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            color,
        },
        Vertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color,
        },
    ]
}

fn multithreaded_harness() -> Harness {
    Harness::create(&HarnessConfig {
        behavior_flags: D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED,
        ..HarnessConfig::default()
    })
}

/// Drive the fixed-function pipeline so a draw shows the vertex diffuse colour.
fn arm_diffuse(h: &Harness) {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture");
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(FVF), 0, "SetFVF");
}

/// A second thread sets state, refills a dynamic buffer and flushes while the first draws.
///
/// The worker's `GetData(D3DGETDATA_FLUSH)` and `Present` each submit the
/// frame the main thread is in the middle of recording, and its
/// `SetRenderState` and `Unlock` push ops into it, so every device entry point
/// the two threads share has to be serialised for the process to survive 200
/// frames of it. Two presenters are legal under the flag.
#[test]
fn two_threads_drive_one_multithreaded_device() {
    let h = multithreaded_harness();
    arm_diffuse(&h);
    let tri = solid_triangle(GREEN);
    let vb_a = h.create_vertex_buffer(stride() * 3, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    vb_a.lock(0, 0, 0).write(&tri);
    let vb_b = h.create_vertex_buffer(
        stride() * 3,
        D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
        FVF,
        D3DPOOL_DEFAULT,
    );
    let query = h
        .create_query(D3DQUERYTYPE_OCCLUSION)
        .expect("OCCLUSION query is supported");

    let shared = h.shared();
    let shared_vb = shared.share_vertex_buffer(&vb_b);
    let shared_query = shared.share_query(&query);
    let stop = AtomicBool::new(false);
    let words = [0u32; 12];

    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let mut results = Vec::new();
            let mut clockwise = false;
            while !stop.load(Ordering::Acquire) {
                let cull = if clockwise { D3DCULL_CW } else { D3DCULL_CCW };
                clockwise = !clockwise;
                results.push((
                    "SetRenderState",
                    shared.set_render_state(D3DRS_CULLMODE, cull),
                ));
                results.push(("Lock/Unlock", shared_vb.fill_u32(&words, D3DLOCK_DISCARD)));
                results.push(("Issue(BEGIN)", shared_query.issue(D3DISSUE_BEGIN)));
                results.push(("Issue(END)", shared_query.issue(D3DISSUE_END)));
                results.push(("GetData(FLUSH)", shared_query.data_u32(D3DGETDATA_FLUSH).0));
                results.push(("Present", shared.present()));
            }
            results
        });

        for frame in 0..200 {
            assert!(h.pump(), "WM_QUIT during frame {frame}");
            assert_eq!(h.begin_scene(), D3D_OK, "BeginScene, frame {frame}");
            assert_eq!(h.clear_target(BLUE), D3D_OK, "Clear, frame {frame}");
            assert_eq!(
                h.set_stream_source(0, &vb_a, 0, stride()),
                D3D_OK,
                "SetStreamSource, frame {frame}"
            );
            for draw in 0..50 {
                assert_eq!(
                    h.draw_primitive(D3DPT_TRIANGLELIST, 0, 1),
                    D3D_OK,
                    "DrawPrimitive {draw}, frame {frame}"
                );
            }
            assert_eq!(h.end_scene(), D3D_OK, "EndScene, frame {frame}");
            assert_eq!(h.present(), D3D_OK, "Present, frame {frame}");
        }
        stop.store(true, Ordering::Release);

        let results = worker.join().expect("the worker thread panicked");
        assert!(!results.is_empty(), "the worker made no calls");
        // `GetData` may answer `S_FALSE` (1) for a query the GPU has not
        // retired; every other call answers `D3D_OK`, and no call fails.
        for (call, hr) in &results {
            assert!(*hr >= 0, "{call} on the worker thread failed: 0x{hr:08X}");
        }
    });

    assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), D3D_OK);
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1),
            D3D_OK,
            "DrawPrimitive after the worker stopped"
        );
    });
    assert_pixel_eq(
        h.read_pixel(320, 280),
        GREEN,
        "the device still renders after two threads drove it",
    );
}

/// The paths that re-enter the device from inside an entry point do not deadlock.
///
/// `SetTexture` takes the texture's `AddRef` thunk from inside the device's
/// own, a state-block `Apply` goes through the setters an application calls,
/// `Reset` reapplies state the same way, and `GetDevice` on a child hands
/// back a reference whose release re-enters the device. Each holds the lock
/// twice on one thread; a lock that is not reentrant wedges here and the
/// runner's timeout fails the test.
#[test]
fn multithreaded_device_reenters_its_lock_on_one_thread() {
    let h = multithreaded_harness();
    let texture = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    texture.lock_rect(0, 0).write_u32_rect(2, 2, &[GREEN; 4]);
    assert_eq!(h.set_texture(0, &texture), D3D_OK, "SetTexture");

    let block = h.create_state_block(D3DSBT_ALL);
    assert_eq!(block.capture(), D3D_OK, "Capture");
    assert_eq!(block.apply(), D3D_OK, "Apply");
    assert_eq!(h.clear_texture(0), D3D_OK, "clear the stage before Reset");
    drop(block);

    let (width, height) = h.dims();
    assert_eq!(h.reset(width, height), D3D_OK, "same-size Reset");

    arm_diffuse(&h);
    let tri = solid_triangle(GREEN);
    let vb = h.create_vertex_buffer(stride() * 3, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    vb.lock(0, 0, 0).write(&tri);
    let (hr, device) = vb.get_device();
    assert_eq!(hr, D3D_OK, "GetDevice");
    // SAFETY: `device` is the reference `GetDevice` handed out just above.
    let count = unsafe { h.release_device_ref(device) };
    assert!(count >= 1, "the harness still holds the device");

    assert_eq!(h.set_stream_source(0, &vb, 0, stride()), D3D_OK);
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), D3D_OK);
    });
    assert_pixel_eq(
        h.read_pixel(320, 280),
        GREEN,
        "the device renders after every re-entrant path",
    );
}
