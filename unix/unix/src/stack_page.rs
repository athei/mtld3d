//! Runs a hot call at a fixed offset within a 4 KiB stack page.
//!
//! The `x86_64` `mtld3d.so` runs under Rosetta, which maps the process with
//! 4 KiB pages. There a call whose callee saves its registers across a page
//! boundary costs several nanoseconds more than the same call a cache line
//! higher or lower. The encoder's draw path makes several such calls per draw,
//! each at a fixed depth below the draw's entry, so which of them straddle a
//! boundary is decided by the page offset the draw path starts at. Without a
//! pin that offset is the sum of every frame above it (the encoder loop, frame
//! replay, op dispatch), and a change to any of them, even to code inlined
//! there that never runs, could move a hot save onto a boundary and slow the
//! encoder by a third. [`run_pinned`] takes the frames above out of
//! that sum. Every submit replay, thousands of native calls per frame, is
//! pinned the same way, on the submit thread and on the encoder thread's
//! synchronous path.

use core::mem::MaybeUninit;

/// The translator's page size for an `x86_64` process.
const PAGE: usize = 4096;

/// The granularity of the stack gap, and so of the pinned offset.
const STEP: usize = 64;

/// How far below [`TARGET`] a frame inside the pinned call may sit and still count as pinned.
///
/// The gap function's own frame and the closure's locals come between the
/// gap and such a frame; they take far less than this.
#[cfg(any(test, perf_tracking))]
const SLACK: usize = 512;

/// The page offset the pinned call starts near.
///
/// High in the page: the draw path's own frame takes about 2 KiB, so the calls
/// it makes for every draw start near the middle of the page, far from either
/// boundary.
const TARGET: usize = 0xfc0;

/// The array of [`gap`] instances for the listed sizes, in [`STEP`]s.
macro_rules! gap_table {
    ($($steps:literal)*) => {
        [$(gap::<F, R, { $steps * STEP + 16 }>,)*]
    };
}

/// Run `f` with its stack starting at the same 4 KiB page offset on every call.
///
/// The offset depends only on this function's frame and on `f`, never on the
/// caller's depth: a gap of 16 to `PAGE - STEP + 16` bytes is reserved below
/// this frame, chosen from where this frame sits. The gap is never written.
#[inline(never)]
pub fn run_pinned<F: FnOnce() -> R, R>(f: F) -> R {
    let anchor = 0u8;
    let here = core::ptr::from_ref(core::hint::black_box(&anchor)) as usize;
    let index = here.wrapping_sub(TARGET) % PAGE / STEP;
    Gaps::<F, R>::TABLE[index](f)
}

/// Whether the calling frame sits where [`run_pinned`] starts the call it runs.
///
/// From inside such a call this is false only when the pin did not take, for
/// instance when a toolchain change folded the gap away; the PERF builds count
/// those draws (`draw_unpinned_total`). Out of line on purpose: its frame then
/// sits where the pinned call's callees do, right below the gap, wherever the
/// caller's own locals ended up in its frame.
#[cfg(any(test, perf_tracking))]
#[inline(never)]
pub fn at_pin() -> bool {
    let probe = 0u8;
    let here = core::ptr::from_ref(core::hint::black_box(&probe)) as usize;
    (TARGET + STEP).wrapping_sub(here) % PAGE < SLACK
}

/// One instance of [`gap`] per gap size, indexed by size in [`STEP`]s.
struct Gaps<F, R>(core::marker::PhantomData<(F, R)>);

impl<F: FnOnce() -> R, R> Gaps<F, R> {
    const TABLE: [fn(F) -> R; PAGE / STEP] = gap_table!(
        0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
        16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
        32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
        48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
    );
}

/// Call `f` from a frame `BYTES` larger than it would be otherwise.
#[inline(never)]
fn gap<F: FnOnce() -> R, R, const BYTES: usize>(f: F) -> R {
    let reserved = [const { MaybeUninit::<u8>::uninit() }; BYTES];
    core::hint::black_box(&reserved);
    let result = f();
    // Using the reservation after the call keeps `f` from being a tail call,
    // which would release this frame before `f` runs.
    core::hint::black_box(&reserved);
    result
}

#[cfg(test)]
mod tests;
