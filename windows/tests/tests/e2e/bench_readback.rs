//! Frame time when every frame reads a render target back into system memory.
//!
//! Guards the cost of a `GetRenderTargetData` the application reads back
//! later, the pattern of a game that copies a small render target out each
//! frame for hardware or the CPU without stalling on it: it reads frame n into
//! one of three `D3DPOOL_SYSTEMMEM` surfaces in turn and locks the one frame
//! n - 2 was read into. D3D9 owes those pixels only at the lock, so the copy
//! can ride its frame and finish while the next one is recorded; when
//! `GetRenderTargetData` instead submits the frame and waits for the GPU, every
//! frame pays a round trip to the GPU however late it reads the copy.
//!
//! The frame is minimal: a 256x256 A8R8G8B8 render target is cleared to a
//! colour of the frame's own, read back, the surface read two frames before is
//! locked read-only and its rows copied out, the back buffer is cleared and
//! `Present` runs unthrottled. The report times, with the benchmarks' `rdtsc`
//! clock, each `GetRenderTargetData` call, each lock with its copy, and the two
//! together (`blocked`), the time the frame spends waiting on the readback; it
//! checks that every lock reads the colour of the frame copied into it.

use std::time::{Duration, SystemTime};

use mtld3d_tests::{Harness, HarnessConfig, MemorySample, Surface};
use mtld3d_types::{
    D3DCLEAR_TARGET, D3DFMT_A8R8G8B8, D3DLOCK_READONLY, D3DPOOL_SYSTEMMEM,
    D3DPRESENT_INTERVAL_IMMEDIATE,
};

use crate::bench::{
    Class, Direction, FrameClock, FrameStats, LayerLog, Metrics, TscClock, Value, memory_section,
    ok, sample_capacity, write_report,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// The side of the render target read back each frame.
const TARGET_EDGE: u32 = 256;
const TARGET_ROW_BYTES: usize = TARGET_EDGE as usize * 4;
const WARM_UP_FRAMES: u32 = 60;
/// The measured phase is at least this many frames and at least one perf window long.
const MEASURED_FRAMES: usize = 600;
/// The shortest frame the sample buffers are sized for, so they never grow while measuring.
const FRAME_FLOOR: Duration = Duration::from_micros(50);
/// How many frames after its copy a surface is locked: the one read back longest ago.
const LOCK_AFTER: u32 = 2;
/// System-memory surfaces the frames read back into in turn.
const SURFACES: usize = LOCK_AFTER as usize + 1;
/// The back buffer's clear colour.
const BACKGROUND: u32 = 0xFF20_2020;

/// A render target read back every frame and locked two frames later.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn readback_every_frame() {
    // Before the device: its creation logs, so the layer log is written after
    // this mark whatever the benchmark's warm-up logs.
    let tsc = TscClock::calibrated();
    let since = SystemTime::now();
    let before = MemorySample::now();
    let h = Harness::create(&HarnessConfig {
        width: WIDTH,
        height: HEIGHT,
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        ..HarnessConfig::default()
    });
    let started = TscClock::now();
    let mut scene = Scene::new(&h);
    let mut frame = 0;
    for _ in 0..WARM_UP_FRAMES {
        assert!(h.pump(), "WM_QUIT during warm-up");
        scene.frame(frame);
        ok(h.present(), "Present");
        frame += 1;
    }
    let log = LayerLog::find(since);
    let warm = MemorySample::now();

    let start = log.start_span(started, || {
        assert!(h.pump(), "WM_QUIT outside the measured frames");
        scene.frame(frame);
        ok(h.present(), "Present");
        frame += 1;
    });
    let capacity = usize::try_from(start.length().as_micros() / FRAME_FLOOR.as_micros())
        .expect("frame capacity fits usize")
        .max(MEASURED_FRAMES);
    // Allocated before the clock starts, so no measured frame carries the allocation.
    let mut readbacks = Vec::with_capacity(capacity.max(sample_capacity::<Duration>()));
    let mut locks = Vec::with_capacity(capacity.max(sample_capacity::<Duration>()));
    let mut blocked = Vec::with_capacity(capacity.max(sample_capacity::<Duration>()));
    let mut clock = FrameClock::start(capacity);
    while clock.frames() < MEASURED_FRAMES || clock.elapsed() < start.length() {
        assert!(h.pump(), "WM_QUIT during the measured frames");
        let times = scene.frame(frame);
        readbacks.push(TscClock::duration(times.readback));
        locks.push(TscClock::duration(times.lock));
        blocked.push(TscClock::duration(times.readback + times.lock));
        clock.present(&h);
        frame += 1;
    }
    let end = MemorySample::now();
    let span = start.end(&log, || {
        assert!(h.pump(), "WM_QUIT outside the measured frames");
        scene.frame(frame);
        ok(h.present(), "Present");
        frame += 1;
    });

    let stats = clock.stats();
    let work = clock.work_stats();
    let readback = FrameStats::of(&readbacks);
    let lock = FrameStats::of(&locks);
    let blocked = FrameStats::of(&blocked);
    let frames = readbacks.len();
    let body = format!(
        "shape: back buffer {WIDTH}x{HEIGHT}, presented unthrottled; a {TARGET_EDGE}x{TARGET_EDGE} \
         A8R8G8B8 render target read back each frame into one of {SURFACES} SYSTEMMEM surfaces\n\
         per frame: Clear the target, GetRenderTargetData into surface n % {SURFACES}, \
         LockRect(READONLY) of the surface frame n - {LOCK_AFTER} read into and copy its rows, \
         Clear the back buffer, Present\n\
         warm-up: {WARM_UP_FRAMES} frames\n\
         measured: {frames} frames in {elapsed:.2?} (at least {MEASURED_FRAMES} frames \
         and {length:?}, {start})\n\
         frame time (Present to Present): {row}\n\
         API work (Present return to Present call): {work}\n\
         GetRenderTargetData: {readback}\n\
         LockRect, copy and UnlockRect: {lock}\n\
         both, per frame: {blocked}\n\
         {memory}{perf}",
        elapsed = clock.elapsed(),
        row = stats.row(),
        work = work.row(),
        readback = readback.row(),
        lock = lock.row(),
        blocked = blocked.row(),
        memory = memory_section(&before, &warm, &end),
        start = span.start(),
        length = span.length(),
        perf = span.perf_rows(&log).section(),
    );

    let mut metrics = Metrics::new("readback_every_frame", &h, &tsc);
    metrics.frame_rows("frame", &stats);
    metrics.frame_rows("api", &work);
    // The readback's wait can sit in either call: a layer that waits for the
    // copy in `GetRenderTargetData` leaves the lock free, and one that waits
    // at the lock leaves the call free. Each half alone moves whenever the
    // wait moves, so only the two together, and the frame, are judged.
    metrics.context_rows("readback", &readback);
    metrics.context_rows("lock", &lock);
    metrics.frame_rows("blocked", &blocked);
    metrics.metric(
        "measured.frames",
        Value::Count(u64::try_from(frames).expect("a count fits u64")),
        Direction::Higher,
        Class::Info,
    );
    metrics.metric(
        "measured.ms",
        Value::Ms(clock.elapsed()),
        Direction::Lower,
        Class::Info,
    );
    metrics.memory(&before, &warm, &end);
    metrics.meta("window_s", &span.window_s());
    // No `perf.*` metrics. The layer's perf windows count a frame per
    // submission, not per `Present`, and a layer that submits the frame to read
    // it back submits twice per presented frame, so each of its per-frame
    // counts and times covers half a frame. Against a layer that submits once,
    // every `perf.*` row compares unlike units, whether or not the work behind
    // it moved (here it does: the copy's GPU time joins the frame and the wait
    // moves from `GetRenderTargetData` into the lock). The benchmark's own
    // timers measure per presented frame, so they are what it compares; the
    // perf summary stays in the report's text.
    metrics.meta(
        "perf_metrics",
        "omitted: perf windows count submissions as frames",
    );
    write_report(&metrics, &log, &body);
}

/// How long one frame's readback and lock took, in `rdtsc` ticks.
struct FrameTimes {
    readback: u64,
    lock: u64,
}

/// The render target, the surfaces it is read back into, and the rows copied out of them.
struct Scene<'h> {
    h: &'h Harness,
    back_buffer: Surface<'h>,
    target: Surface<'h>,
    surfaces: [Surface<'h>; SURFACES],
    rows: Vec<u8>,
}

impl<'h> Scene<'h> {
    fn new(h: &'h Harness) -> Self {
        Self {
            h,
            back_buffer: h.render_target(0),
            target: h.create_render_target(TARGET_EDGE, TARGET_EDGE, D3DFMT_A8R8G8B8),
            surfaces: [0; SURFACES].map(|_| {
                h.create_offscreen_plain_surface(
                    TARGET_EDGE,
                    TARGET_EDGE,
                    D3DFMT_A8R8G8B8,
                    D3DPOOL_SYSTEMMEM,
                )
            }),
            rows: vec![0; TARGET_ROW_BYTES * TARGET_EDGE as usize],
        }
    }

    /// The colour frame `frame` clears its target to.
    const fn colour(frame: u32) -> u32 {
        0xFF00_0000 | frame.wrapping_mul(0x0001_0307) & 0x00FF_FFFF
    }

    /// Frame `frame`, up to its `Present`.
    fn frame(&mut self, frame: u32) -> FrameTimes {
        let h = self.h;
        let into = frame as usize % SURFACES;
        ok(
            h.set_render_target(0, &self.target),
            "SetRenderTarget(target)",
        );
        ok(
            h.clear(D3DCLEAR_TARGET, Self::colour(frame), 1.0, 0),
            "Clear(target)",
        );
        let before = TscClock::now();
        ok(
            h.get_render_target_data_hr(&self.target, &self.surfaces[into]),
            "GetRenderTargetData",
        );
        let read = TscClock::now();
        let mut lock = 0;
        if let Some(previous) = frame.checked_sub(LOCK_AFTER) {
            let locked = self.surfaces[previous as usize % SURFACES].lock_rect(D3DLOCK_READONLY);
            let pitch = usize::try_from(locked.pitch()).expect("a positive pitch");
            for row in 0..TARGET_EDGE as usize {
                // SAFETY: the locked surface holds TARGET_EDGE rows of `pitch`
                // bytes, so this row's start lies inside it.
                let start = unsafe { locked.bits_ptr().add(row * pitch) };
                // SAFETY: each row is at least TARGET_ROW_BYTES long.
                let from = unsafe { core::slice::from_raw_parts(start, TARGET_ROW_BYTES) };
                self.rows[row * TARGET_ROW_BYTES..(row + 1) * TARGET_ROW_BYTES]
                    .copy_from_slice(from);
            }
            drop(locked);
            lock = TscClock::now() - read;
            let first = u32::from_le_bytes(self.rows[..4].try_into().expect("four bytes"));
            assert_eq!(
                first,
                Self::colour(previous),
                "the lock reads frame {previous}'s copy"
            );
        }
        ok(
            h.set_render_target(0, &self.back_buffer),
            "SetRenderTarget(back buffer)",
        );
        ok(
            h.clear(D3DCLEAR_TARGET, BACKGROUND, 1.0, 0),
            "Clear(back buffer)",
        );
        FrameTimes {
            readback: read - before,
            lock,
        }
    }
}
