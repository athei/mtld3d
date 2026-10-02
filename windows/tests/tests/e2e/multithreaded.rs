//! A device created with `D3DCREATE_MULTITHREADED`, called from two threads.
//!
//! The flag is D3D9's promise that the device and its resources may be called
//! from any thread. Without it a call from a second thread is undefined, as it
//! is on native, so no test here drives an unflagged device from two threads.

use core::ffi::c_void;
use std::{
    sync::atomic::{AtomicBool, AtomicU32, Ordering},
    time::{Duration, Instant},
};

use mtld3d_tests::{Harness, HarnessConfig, Vertex, assert_pixel_eq, spawn_scoped};
use mtld3d_types::{
    D3D_OK, D3DCREATE_HARDWARE_VERTEXPROCESSING, D3DCREATE_MULTITHREADED, D3DCULL_CCW, D3DCULL_CW,
    D3DCULL_NONE, D3DFMT_A8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DGETDATA_FLUSH, D3DISSUE_BEGIN,
    D3DISSUE_END, D3DLOCK_DISCARD, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPT_TRIANGLELIST,
    D3DQUERYTYPE_OCCLUSION, D3DRS_CULLMODE, D3DRS_LIGHTING, D3DSBT_ALL, D3DUSAGE_DYNAMIC,
    D3DUSAGE_WRITEONLY, E_NOINTERFACE, Guid,
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
        let worker = spawn_scoped(scope, || {
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

/// The private-data key the teardown probe is stored under.
const TEARDOWN_PROBE_KEY: Guid = Guid {
    data1: 0x6d74_6c64,
    data2: 0x0942,
    data3: 0x0001,
    data4: *b"teardown",
};

/// How long the probe holds the teardown open for a call that should be waiting.
const TEARDOWN_HOLD: Duration = Duration::from_millis(500);

/// How long a worker waits for a teardown that never starts before it gives up.
const TEARDOWN_START_LIMIT: Duration = Duration::from_secs(20);

/// The `IUnknown` head of [`TeardownProbe`].
#[repr(C)]
struct ProbeVtbl {
    query_interface: extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    add_ref: extern "system" fn(*mut c_void) -> u32,
    release: extern "system" fn(*mut c_void) -> u32,
}

static PROBE_VTBL: ProbeVtbl = ProbeVtbl {
    query_interface: probe_query_interface,
    add_ref: probe_add_ref,
    release: probe_release,
};

/// A COM object stored as private data, whose last `Release` runs inside the device's teardown.
///
/// An implicit surface's private data is released when the device is
/// destroyed, in the middle of the final `Release`. The probe's last
/// `Release` marks that moment, then holds the teardown open for
/// [`TEARDOWN_HOLD`] and records whether the call another thread made in
/// the meantime came back while the teardown was still running.
#[repr(C)]
struct TeardownProbe {
    vtbl: &'static ProbeVtbl,
    refcount: AtomicU32,
    /// Set by the last `Release`, from inside the device's teardown.
    teardown_started: AtomicBool,
    /// Set by the worker once its call returned.
    call_returned: AtomicBool,
    /// Whether the worker's call had returned before the hold ran out.
    returned_during_teardown: AtomicBool,
}

impl TeardownProbe {
    const fn new() -> Self {
        Self {
            vtbl: &PROBE_VTBL,
            refcount: AtomicU32::new(0),
            teardown_started: AtomicBool::new(false),
            call_returned: AtomicBool::new(false),
            returned_during_teardown: AtomicBool::new(false),
        }
    }

    const fn as_unknown(&self) -> *mut c_void {
        core::ptr::from_ref(self).cast_mut().cast::<c_void>()
    }

    /// Block until the teardown starts; false when it does not within the limit.
    fn await_teardown(&self) -> bool {
        let deadline = Instant::now() + TEARDOWN_START_LIMIT;
        while !self.teardown_started.load(Ordering::Acquire) {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    }
}

fn probe_from(this: *mut c_void) -> &'static TeardownProbe {
    // SAFETY: the runtime only calls the probe through the pointer the test
    // stored, which names a `TeardownProbe` that outlives every call: the
    // test keeps it until the device that holds the reference is gone.
    unsafe { &*this.cast::<TeardownProbe>() }
}

extern "system" fn probe_query_interface(
    _this: *mut c_void,
    _riid: *const Guid,
    ppv: *mut *mut c_void,
) -> i32 {
    if !ppv.is_null() {
        // SAFETY: the caller's out-param, checked non-null.
        unsafe { *ppv = core::ptr::null_mut() };
    }
    E_NOINTERFACE
}

extern "system" fn probe_add_ref(this: *mut c_void) -> u32 {
    probe_from(this).refcount.fetch_add(1, Ordering::AcqRel) + 1
}

extern "system" fn probe_release(this: *mut c_void) -> u32 {
    let probe = probe_from(this);
    let remaining = probe.refcount.fetch_sub(1, Ordering::AcqRel) - 1;
    if remaining == 0 {
        probe.teardown_started.store(true, Ordering::Release);
        let deadline = Instant::now() + TEARDOWN_HOLD;
        while Instant::now() < deadline && !probe.call_returned.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(1));
        }
        probe.returned_during_teardown.store(
            probe.call_returned.load(Ordering::Acquire),
            Ordering::Release,
        );
    }
    remaining
}

/// A managed texture called during the device's final `Release` waits for the teardown to end.
///
/// A `D3DPOOL_MANAGED` texture does not pin its device, so an application
/// may release the device while another thread still holds and uses the
/// texture. The releasing thread holds the device's lock for the whole
/// teardown, and a texture call from the other thread has to wait for it:
/// the teardown detaches the texture and releases the device's own
/// references on it, and a call running beside that races the texture's
/// counts and reads a device that is being freed. The probe stored on the
/// implicit render target is released in the middle of that teardown, after
/// the texture has been detached, and starts the other thread's `LockRect`
/// there; the call has to come back only once the final `Release` returned,
/// and succeed on the texture the device left behind.
#[test]
fn a_managed_texture_waits_out_the_final_device_release() {
    let h = multithreaded_harness();
    let texture = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    texture.lock_rect(0, 0).write_u32_rect(4, 4, &[GREEN; 16]);
    let probe = TeardownProbe::new();
    {
        let back_buffer = h.back_buffer(0);
        // SAFETY: the probe lives on this frame until the end of the test,
        // past the device's destruction, which releases the reference.
        let hr = unsafe {
            back_buffer.set_private_data_unknown(&TEARDOWN_PROBE_KEY, probe.as_unknown())
        };
        assert_eq!(
            hr, D3D_OK,
            "SetPrivateData(D3DSPD_IUNKNOWN) on the back buffer"
        );
    }
    assert_eq!(
        probe.refcount.load(Ordering::Acquire),
        1,
        "the back buffer holds the probe's one reference"
    );

    let shared = h.shared();
    let shared_texture = shared.share_texture(&texture);
    let worker_hr = std::thread::scope(|scope| {
        let worker = spawn_scoped(scope, || {
            if !probe.await_teardown() {
                return None;
            }
            let hr = shared_texture.lock_and_unlock(0);
            probe.call_returned.store(true, Ordering::Release);
            Some(hr)
        });
        assert_eq!(
            h.release_device(),
            0,
            "the harness held the only device reference"
        );
        worker.join().expect("the worker thread panicked")
    });

    assert!(
        probe.teardown_started.load(Ordering::Acquire),
        "the device's destruction released the back buffer's private data"
    );
    let hr = worker_hr.expect("the worker saw the teardown start");
    assert_eq!(
        hr, D3D_OK,
        "LockRect and UnlockRect on the texture the device left behind"
    );
    assert!(
        !probe.returned_during_teardown.load(Ordering::Acquire),
        "a managed texture call ran beside the final Release instead of waiting for its lock"
    );
}
