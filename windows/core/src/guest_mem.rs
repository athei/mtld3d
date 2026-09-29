//! The C memory routines of the `x86_64` DLL, and the route that picks where each call runs.
//!
//! The compiler lowers every copy, fill and comparison it does not inline to a
//! call to `memcpy`, `memmove`, `memset` or `memcmp`, and without a definition
//! in the image those resolve to `vcruntime140.dll`. On an arm64 Wine that DLL
//! is ARM64X, native code, so an `x86_64` guest leaves the x86 emulator and
//! re-enters it on every such call: about 25 ns each under FEX, more than a
//! small copy costs. `d3d9.dll` therefore defines the four itself and sends
//! them through a [`MemRoute`].
//!
//! The first call decides the route for the process. Under the x64 emulator
//! the routines here run, and only a copy longer than [`LOCAL_COPY_MAX`] goes to
//! the CRT, where one crossing costs less than the copy and the CRT's copy runs
//! natively. Everywhere else, Rosetta included, every call goes to the CRT:
//! there is no crossing to save, and Rosetta runs the CRT's x86 routines faster
//! than these at mid sizes.
//!
//! The routines move data with 16- and 32-byte unaligned loads and stores, SSE
//! registers under the no-AVX baseline, and never with the string instructions
//! (`rep movsb`), which Rosetta runs as a byte loop. A range of up to 64 bytes
//! moves as its first and last `K` bytes for the largest `K` that fits, every
//! load before any store, which makes the move correct for overlapping ranges
//! too; a longer one as 64-byte blocks and then that small move for the rest.
//!
//! None of it may compile to a call to the routine it implements, or the call
//! would recurse. Every load and store has a constant size of at most 32
//! bytes, which LLVM expands inline in an optimised build (the DLLs are only
//! built optimised), and the block loops step by a stride the optimiser cannot
//! see (`block_stride`), so its loop idiom pass cannot turn a loop back into a
//! call. The Makefile's `MEM_ROUTINE_GATE` checks that this holds: every build
//! of the `x86_64` DLL fails if any function of this module, or any of the
//! four exports, calls or jumps straight to one of the four.

use core::{
    ffi::c_void,
    ptr::NonNull,
    sync::atomic::{AtomicPtr, AtomicU8, Ordering},
};

/// The longest copy the routines here make under the emulator; a longer one goes to the CRT.
///
/// Under FEX a call into the CRT costs about 25 ns and the CRT's copy then
/// runs natively, which beats the translated block loop somewhere past a few
/// hundred bytes. A 256-byte shader-constant upload stays below it.
pub const LOCAL_COPY_MAX: usize = 512;

/// No call has reached the route yet.
const UNDECIDED: u8 = 0;
/// One call is asking the host; every other call meanwhile takes the routines here.
const DECIDING: u8 = 1;
/// Under an emulator: the routines here, and the CRT's for copies past [`LOCAL_COPY_MAX`].
const EMULATED: u8 = 2;
/// The CRT's routines were not found: the routines here for every call.
const NO_CRT: u8 = 3;
/// The CRT's routines for every call.
const CRT: u8 = 4;

/// The slot of the CRT's `memcpy` in [`MemRoute`]'s table.
const MEMCPY: usize = 0;
/// The slot of the CRT's `memmove`.
const MEMMOVE: usize = 1;
/// The slot of the CRT's `memset`.
const MEMSET: usize = 2;
/// The slot of the CRT's `memcmp`.
const MEMCMP: usize = 3;

/// The addresses of the CRT's `memcpy`, `memmove`, `memset` and `memcmp`.
pub struct CrtMem {
    routines: [NonNull<c_void>; 4],
}

impl CrtMem {
    /// Names the four routines by their entry points.
    ///
    /// # Safety
    ///
    /// Each address must be the entry point of the C routine its parameter is
    /// named after, with that routine's C signature, and stay valid for as
    /// long as a [`MemRoute`] it is given to routes calls.
    #[must_use]
    pub const unsafe fn new(
        memcpy: NonNull<c_void>,
        memmove: NonNull<c_void>,
        memset: NonNull<c_void>,
        memcmp: NonNull<c_void>,
    ) -> Self {
        Self {
            routines: [memcpy, memmove, memset, memcmp],
        }
    }
}

/// What the host answers a [`MemRoute`] on the first call.
pub struct MemHost {
    emulated: bool,
    crt: Option<CrtMem>,
}

impl MemHost {
    /// Whether an x86 emulator on another architecture runs this process, and the CRT's routines.
    #[must_use]
    pub const fn new(emulated: bool, crt: Option<CrtMem>) -> Self {
        Self { emulated, crt }
    }
}

/// The routines a [`MemRoute`] latched, for the log.
#[derive(Debug, PartialEq, Eq)]
pub enum LatchedRoute {
    /// No call has decided the route yet, or one is deciding it now.
    Undecided,
    /// Under an emulator: the routines here, and the CRT's for copies past [`LOCAL_COPY_MAX`].
    Emulated,
    /// The CRT's routines were not found: the routines here for every call.
    NoCrt,
    /// The CRT's routines for every call.
    Crt,
}

/// Where one image's `memcpy`, `memmove`, `memset` and `memcmp` go, decided by the first call.
///
/// The latch never blocks. The first call to find it undecided claims the
/// decision and asks the host; any call that finds the decision claimed but
/// not yet made takes the routines here, which are correct on every route.
/// That covers another thread racing the first call and the host query
/// itself, which may copy memory on the deciding thread, so no call waits and
/// none recurses. The CRT's addresses are stored before the route is
/// published with release ordering and read after it is loaded with acquire
/// ordering; neither changes after that.
pub struct MemRoute {
    state: AtomicU8,
    crt: [AtomicPtr<c_void>; 4],
}

impl Default for MemRoute {
    fn default() -> Self {
        Self::new()
    }
}

impl MemRoute {
    /// A route no call has decided yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: AtomicU8::new(UNDECIDED),
            crt: [const { AtomicPtr::new(core::ptr::null_mut()) }; 4],
        }
    }

    /// Decides the route if no call has yet, and names the routines it latched.
    #[must_use]
    pub fn latch(&self, host: impl FnOnce() -> MemHost) -> LatchedRoute {
        self.settle(host);
        match self.state.load(Ordering::Acquire) {
            EMULATED => LatchedRoute::Emulated,
            NO_CRT => LatchedRoute::NoCrt,
            CRT => LatchedRoute::Crt,
            // `DECIDING`, the one other value left: a call is asking the host.
            _ => LatchedRoute::Undecided,
        }
    }

    /// The C `memcpy`, on the latched route.
    ///
    /// `host` is asked once, by the first call to reach the route through any
    /// of the four routines. A call the route forwards to the CRT is a tail
    /// jump from here; everything else happens in `memcpy_in_image`, which is
    /// kept out of line, so this part needs no stack frame.
    ///
    /// # Safety
    ///
    /// As for C's `memcpy`: `src` must be valid for `n` bytes of reads, `dst`
    /// for `n` bytes of writes, and the two ranges must not overlap.
    #[inline]
    pub unsafe fn memcpy(
        &self,
        host: impl FnOnce() -> MemHost,
        dst: *mut c_void,
        src: *const c_void,
        n: usize,
    ) -> *mut c_void {
        if self.forwards_copy(n) {
            // SAFETY: both forwarding routes store the CRT's addresses before they are published.
            let crt = unsafe { self.copy_routine(MEMCPY) };
            // SAFETY: the caller's contract is `memcpy`'s.
            return unsafe { crt(dst, src, n) };
        }
        // SAFETY: the caller's contract.
        unsafe { self.memcpy_in_image(host, dst, src, n) }
    }

    /// The C `memmove`, on the latched route, shaped like [`Self::memcpy`].
    ///
    /// # Safety
    ///
    /// As for C's `memmove`: `src` must be valid for `n` bytes of reads and
    /// `dst` for `n` bytes of writes. The ranges may overlap.
    #[inline]
    pub unsafe fn memmove(
        &self,
        host: impl FnOnce() -> MemHost,
        dst: *mut c_void,
        src: *const c_void,
        n: usize,
    ) -> *mut c_void {
        if self.forwards_copy(n) {
            // SAFETY: both forwarding routes store the CRT's addresses before they are published.
            let crt = unsafe { self.copy_routine(MEMMOVE) };
            // SAFETY: the caller's contract is `memmove`'s.
            return unsafe { crt(dst, src, n) };
        }
        // SAFETY: the caller's contract.
        unsafe { self.memmove_in_image(host, dst, src, n) }
    }

    /// The C `memset`, on the latched route, shaped like [`Self::memcpy`].
    ///
    /// # Safety
    ///
    /// As for C's `memset`: `dst` must be valid for `n` bytes of writes.
    #[inline]
    pub unsafe fn memset(
        &self,
        host: impl FnOnce() -> MemHost,
        dst: *mut c_void,
        value: i32,
        n: usize,
    ) -> *mut c_void {
        if self.forwards_all() {
            // SAFETY: the forwarding route stores the CRT's addresses before it is published.
            let crt = unsafe { self.fill_routine() };
            // SAFETY: the caller's contract is `memset`'s.
            return unsafe { crt(dst, value, n) };
        }
        // SAFETY: the caller's contract.
        unsafe { self.memset_in_image(host, dst, value, n) }
    }

    /// The C `memcmp`, on the latched route, shaped like [`Self::memcpy`].
    ///
    /// # Safety
    ///
    /// As for C's `memcmp`: `a` and `b` must each be valid for `n` bytes of reads.
    #[inline]
    pub unsafe fn memcmp(
        &self,
        host: impl FnOnce() -> MemHost,
        a: *const c_void,
        b: *const c_void,
        n: usize,
    ) -> i32 {
        if self.forwards_all() {
            // SAFETY: the forwarding route stores the CRT's addresses before it is published.
            let crt = unsafe { self.compare_routine() };
            // SAFETY: the caller's contract is `memcmp`'s.
            return unsafe { crt(a, b, n) };
        }
        // SAFETY: the caller's contract.
        unsafe { self.memcmp_in_image(host, a, b, n) }
    }

    /// Whether the route latched so far sends a copy or move of `n` bytes to the CRT.
    #[inline]
    fn forwards_copy(&self, n: usize) -> bool {
        let state = self.state.load(Ordering::Acquire);
        state == CRT || (state == EMULATED && n > LOCAL_COPY_MAX)
    }

    /// Whether the route latched so far sends every call to the CRT.
    #[inline]
    fn forwards_all(&self) -> bool {
        self.state.load(Ordering::Acquire) == CRT
    }

    /// The part of [`Self::memcpy`] that does not forward: the first decision, then the copy.
    ///
    /// The call that decides the route forwards from here if the route it
    /// latched forwards.
    ///
    /// # Safety
    ///
    /// As for [`Self::memcpy`].
    #[inline(never)]
    unsafe fn memcpy_in_image(
        &self,
        host: impl FnOnce() -> MemHost,
        dst: *mut c_void,
        src: *const c_void,
        n: usize,
    ) -> *mut c_void {
        self.settle(host);
        if self.forwards_copy(n) {
            // SAFETY: both forwarding routes store the CRT's addresses before they are published.
            let crt = unsafe { self.copy_routine(MEMCPY) };
            // SAFETY: the caller's contract is `memcpy`'s.
            return unsafe { crt(dst, src, n) };
        }
        // SAFETY: the caller's contract; the ranges do not overlap.
        unsafe { copy(dst.cast(), src.cast(), n) };
        dst
    }

    /// The part of [`Self::memmove`] that does not forward, as [`Self::memcpy_in_image`] is.
    ///
    /// # Safety
    ///
    /// As for [`Self::memmove`].
    #[inline(never)]
    unsafe fn memmove_in_image(
        &self,
        host: impl FnOnce() -> MemHost,
        dst: *mut c_void,
        src: *const c_void,
        n: usize,
    ) -> *mut c_void {
        self.settle(host);
        if self.forwards_copy(n) {
            // SAFETY: both forwarding routes store the CRT's addresses before they are published.
            let crt = unsafe { self.copy_routine(MEMMOVE) };
            // SAFETY: the caller's contract is `memmove`'s.
            return unsafe { crt(dst, src, n) };
        }
        // SAFETY: the caller's contract.
        unsafe { copy_overlapping(dst.cast(), src.cast(), n) };
        dst
    }

    /// The part of [`Self::memset`] that does not forward, as [`Self::memcpy_in_image`] is.
    ///
    /// # Safety
    ///
    /// As for [`Self::memset`].
    #[inline(never)]
    unsafe fn memset_in_image(
        &self,
        host: impl FnOnce() -> MemHost,
        dst: *mut c_void,
        value: i32,
        n: usize,
    ) -> *mut c_void {
        self.settle(host);
        if self.forwards_all() {
            // SAFETY: the forwarding route stores the CRT's addresses before it is published.
            let crt = unsafe { self.fill_routine() };
            // SAFETY: the caller's contract is `memset`'s.
            return unsafe { crt(dst, value, n) };
        }
        // C fills with the low byte of the `int` it is passed.
        let [byte, ..] = value.to_le_bytes();
        // SAFETY: the caller's contract.
        unsafe { fill(dst.cast(), byte, n) };
        dst
    }

    /// The part of [`Self::memcmp`] that does not forward, as [`Self::memcpy_in_image`] is.
    ///
    /// # Safety
    ///
    /// As for [`Self::memcmp`].
    #[inline(never)]
    unsafe fn memcmp_in_image(
        &self,
        host: impl FnOnce() -> MemHost,
        a: *const c_void,
        b: *const c_void,
        n: usize,
    ) -> i32 {
        self.settle(host);
        if self.forwards_all() {
            // SAFETY: the forwarding route stores the CRT's addresses before it is published.
            let crt = unsafe { self.compare_routine() };
            // SAFETY: the caller's contract is `memcmp`'s.
            return unsafe { crt(a, b, n) };
        }
        // SAFETY: the caller's contract.
        unsafe { compare(a.cast(), b.cast(), n) }
    }

    /// Decides the route if no call has claimed the decision yet.
    #[inline]
    fn settle(&self, host: impl FnOnce() -> MemHost) {
        if self.state.load(Ordering::Relaxed) == UNDECIDED {
            self.decide(host);
        }
    }

    /// Claims the decision and makes it, unless another call claimed it first.
    ///
    /// A call that finds the decision claimed, on another thread or from
    /// inside the host query, returns at once and reads the state as
    /// [`DECIDING`], which forwards nothing.
    #[cold]
    fn decide(&self, host: impl FnOnce() -> MemHost) {
        let claimed = self
            .state
            .compare_exchange(UNDECIDED, DECIDING, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok();
        if !claimed {
            return;
        }
        let host = host();
        let route = match host.crt {
            Some(crt) => {
                for (slot, address) in self.crt.iter().zip(crt.routines) {
                    slot.store(address.as_ptr(), Ordering::Relaxed);
                }
                if host.emulated { EMULATED } else { CRT }
            }
            None => NO_CRT,
        };
        self.state.store(route, Ordering::Release);
    }

    /// The CRT's `memcpy` or `memmove`, from `slot`.
    ///
    /// # Safety
    ///
    /// `slot` must be [`MEMCPY`] or [`MEMMOVE`], and the route must have latched
    /// [`EMULATED`] or [`CRT`], which store the addresses.
    #[inline]
    unsafe fn copy_routine(
        &self,
        slot: usize,
    ) -> unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> *mut c_void {
        let address = self.crt[slot].load(Ordering::Relaxed);
        // SAFETY: the slot holds the non-null entry point of the routine with
        // this C signature, per `CrtMem::new`'s contract.
        unsafe {
            core::mem::transmute::<
                *mut c_void,
                unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> *mut c_void,
            >(address)
        }
    }

    /// The CRT's `memset`.
    ///
    /// # Safety
    ///
    /// The route must have latched [`CRT`], which stores the addresses.
    #[inline]
    unsafe fn fill_routine(&self) -> unsafe extern "C" fn(*mut c_void, i32, usize) -> *mut c_void {
        let address = self.crt[MEMSET].load(Ordering::Relaxed);
        // SAFETY: the slot holds the non-null entry point of `memset`, per
        // `CrtMem::new`'s contract.
        unsafe {
            core::mem::transmute::<
                *mut c_void,
                unsafe extern "C" fn(*mut c_void, i32, usize) -> *mut c_void,
            >(address)
        }
    }

    /// The CRT's `memcmp`.
    ///
    /// # Safety
    ///
    /// The route must have latched [`CRT`], which stores the addresses.
    #[inline]
    unsafe fn compare_routine(
        &self,
    ) -> unsafe extern "C" fn(*const c_void, *const c_void, usize) -> i32 {
        let address = self.crt[MEMCMP].load(Ordering::Relaxed);
        // SAFETY: the slot holds the non-null entry point of `memcmp`, per
        // `CrtMem::new`'s contract.
        unsafe {
            core::mem::transmute::<
                *mut c_void,
                unsafe extern "C" fn(*const c_void, *const c_void, usize) -> i32,
            >(address)
        }
    }
}

/// Copies `n` bytes from `src` to `dst`, for ranges that do not overlap.
///
/// # Safety
///
/// `src` must be valid for `n` bytes of reads and `dst` for `n` bytes of writes.
#[inline]
const unsafe fn copy(dst: *mut u8, src: *const u8, n: usize) {
    if n <= 64 {
        // SAFETY: the caller's contract.
        unsafe { copy_small(dst, src, n) };
    } else {
        // SAFETY: the caller's contract; with no overlap either direction is correct.
        unsafe { copy_up(dst, src, n) };
    }
}

/// Copies `n` bytes from `src` to `dst`, which may overlap, as if through a buffer.
///
/// # Safety
///
/// `src` must be valid for `n` bytes of reads and `dst` for `n` bytes of writes.
#[inline]
unsafe fn copy_overlapping(dst: *mut u8, src: *const u8, n: usize) {
    if n <= 64 {
        // SAFETY: the caller's contract; the small copy loads every byte first.
        unsafe { copy_small(dst, src, n) };
    } else if dst.addr().wrapping_sub(src.addr()) >= n {
        // SAFETY: the caller's contract; `dst` is below `src` or past its end,
        // which copying upward allows.
        unsafe { copy_up(dst, src, n) };
    } else {
        // SAFETY: the caller's contract; `dst` is inside `src`'s range, which
        // copying downward allows.
        unsafe { copy_down(dst, src, n) };
    }
}

/// Copies `n <= 64` bytes, loading every one before storing any.
///
/// # Safety
///
/// `src` must be valid for `n` bytes of reads and `dst` for `n` bytes of writes.
#[inline]
const unsafe fn copy_small(dst: *mut u8, src: *const u8, n: usize) {
    if n >= 32 {
        // SAFETY: `32 <= n`, within the caller's ranges.
        unsafe { copy_ends::<32>(dst, src, n) };
    } else if n >= 16 {
        // SAFETY: `16 <= n`, within the caller's ranges.
        unsafe { copy_ends::<16>(dst, src, n) };
    } else if n >= 8 {
        // SAFETY: `8 <= n`, within the caller's ranges.
        unsafe { copy_ends::<8>(dst, src, n) };
    } else if n >= 4 {
        // SAFETY: `4 <= n`, within the caller's ranges.
        unsafe { copy_ends::<4>(dst, src, n) };
    } else if n >= 2 {
        // SAFETY: `2 <= n`, within the caller's ranges.
        unsafe { copy_ends::<2>(dst, src, n) };
    } else if n == 1 {
        // SAFETY: `1 <= n`, within the caller's ranges.
        unsafe { copy_ends::<1>(dst, src, n) };
    }
}

/// Copies the first and the last `K` of `n` bytes, loading both before storing either.
///
/// That covers all `n` bytes when `n <= 2 K`, and loading first makes the copy
/// correct for overlapping ranges.
///
/// # Safety
///
/// `K <= n`, and `src` must be valid for `n` bytes of reads and `dst` for `n`
/// bytes of writes.
#[inline]
const unsafe fn copy_ends<const K: usize>(dst: *mut u8, src: *const u8, n: usize) {
    let tail = n - K;
    let (src_tail, dst_tail) = (src.wrapping_add(tail), dst.wrapping_add(tail));
    // SAFETY: `[0, K)` lies in the caller's `n` readable bytes.
    let head_bytes = unsafe { src.cast::<[u8; K]>().read_unaligned() };
    // SAFETY: `[n - K, n)` lies in them too.
    let tail_bytes = unsafe { src_tail.cast::<[u8; K]>().read_unaligned() };
    // SAFETY: `[0, K)` lies in the caller's `n` writable bytes.
    unsafe { dst.cast::<[u8; K]>().write_unaligned(head_bytes) };
    // SAFETY: `[n - K, n)` lies in them too.
    unsafe { dst_tail.cast::<[u8; K]>().write_unaligned(tail_bytes) };
}

/// Copies `n > 64` bytes as 64-byte blocks from the lowest address up, then the rest.
///
/// Correct for overlapping ranges when `dst` is below `src`: each block is
/// loaded whole before it is stored, and every store lands below the source
/// bytes still to be read.
///
/// # Safety
///
/// `src` must be valid for `n` bytes of reads and `dst` for `n` bytes of writes.
#[inline]
const unsafe fn copy_up(dst: *mut u8, src: *const u8, n: usize) {
    let (step, end) = (block_stride(), n - n % 64);
    let mut at = 0;
    while at < end {
        // SAFETY: `at` is a multiple of 64 below `end`, so `[at, at + 64)` lies below `end <= n`.
        unsafe { copy_ends::<32>(dst.wrapping_add(at), src.wrapping_add(at), 64) };
        at += step;
    }
    // SAFETY: the rest, `[end, n)`, is under 64 bytes.
    unsafe { copy_small(dst.wrapping_add(end), src.wrapping_add(end), n - end) };
}

/// Copies `n > 64` bytes as 64-byte blocks from the highest address down, then the rest.
///
/// Correct for overlapping ranges when `dst` is above `src`: each block is
/// loaded whole before it is stored, and every store lands above the source
/// bytes still to be read.
///
/// # Safety
///
/// `src` must be valid for `n` bytes of reads and `dst` for `n` bytes of writes.
#[inline]
const unsafe fn copy_down(dst: *mut u8, src: *const u8, n: usize) {
    let (step, rest) = (block_stride(), n % 64);
    let mut at = n;
    while at > rest {
        at -= step;
        // SAFETY: `at` is `rest` plus a multiple of 64, so `[at, at + 64)` ends at or below `n`.
        unsafe { copy_ends::<32>(dst.wrapping_add(at), src.wrapping_add(at), 64) };
    }
    // SAFETY: the rest, `[0, rest)`, is under 64 bytes.
    unsafe { copy_small(dst, src, rest) };
}

/// Sets `n` bytes at `dst` to `byte`.
///
/// # Safety
///
/// `dst` must be valid for `n` bytes of writes.
#[inline]
const unsafe fn fill(dst: *mut u8, byte: u8, n: usize) {
    let end = if n > 64 { n - n % 64 } else { 0 };
    if end > 0 {
        let (step, mut at) = (block_stride(), 0);
        while at < end {
            // SAFETY: `at` is a multiple of 64 below `end`, so `[at, at + 64)` lies below `end <= n`.
            unsafe { fill_ends::<32>(dst.wrapping_add(at), byte, 64) };
            at += step;
        }
    }
    // SAFETY: the rest, `[end, n)`, is at most 64 bytes.
    unsafe { fill_small(dst.wrapping_add(end), byte, n - end) };
}

/// The 64-byte step of the block loops, as a value the optimiser cannot see.
///
/// A loop whose stores advance by a constant stride equal to what it stores
/// per step is the shape LLVM's loop idiom pass turns into a call to `memcpy`,
/// `memmove` or `memset`, which here would be a call to the routine itself:
/// the two 32-byte fills of a block merge into one 64-byte `memset`, and the
/// pass then replaces the whole loop with one. The pass needs the stride as a
/// constant, and a stride read through `black_box` is not one. The cost is a
/// stack store and load per call that reaches a block loop.
#[inline]
const fn block_stride() -> usize {
    core::hint::black_box(64)
}

/// Sets `n <= 64` bytes at `dst` to `byte`.
///
/// # Safety
///
/// `dst` must be valid for `n` bytes of writes.
#[inline]
const unsafe fn fill_small(dst: *mut u8, byte: u8, n: usize) {
    if n >= 32 {
        // SAFETY: `32 <= n`, within the caller's range.
        unsafe { fill_ends::<32>(dst, byte, n) };
    } else if n >= 16 {
        // SAFETY: `16 <= n`, within the caller's range.
        unsafe { fill_ends::<16>(dst, byte, n) };
    } else if n >= 8 {
        // SAFETY: `8 <= n`, within the caller's range.
        unsafe { fill_ends::<8>(dst, byte, n) };
    } else if n >= 4 {
        // SAFETY: `4 <= n`, within the caller's range.
        unsafe { fill_ends::<4>(dst, byte, n) };
    } else if n >= 2 {
        // SAFETY: `2 <= n`, within the caller's range.
        unsafe { fill_ends::<2>(dst, byte, n) };
    } else if n == 1 {
        // SAFETY: `1 <= n`, within the caller's range.
        unsafe { fill_ends::<1>(dst, byte, n) };
    }
}

/// Sets the first and the last `K` of `n` bytes at `dst` to `byte`.
///
/// # Safety
///
/// `K <= n`, and `dst` must be valid for `n` bytes of writes.
#[inline]
const unsafe fn fill_ends<const K: usize>(dst: *mut u8, byte: u8, n: usize) {
    let bytes = [byte; K];
    let dst_tail = dst.wrapping_add(n - K);
    // SAFETY: `[0, K)` lies in the caller's `n` writable bytes.
    unsafe { dst.cast::<[u8; K]>().write_unaligned(bytes) };
    // SAFETY: `[n - K, n)` lies in them too.
    unsafe { dst_tail.cast::<[u8; K]>().write_unaligned(bytes) };
}

/// Compares `n` bytes as unsigned, the sign of the first difference, or 0 for none.
///
/// # Safety
///
/// `a` and `b` must each be valid for `n` bytes of reads.
#[inline]
unsafe fn compare(a: *const u8, b: *const u8, n: usize) -> i32 {
    let mut at = 0;
    while n - at >= 16 {
        // SAFETY: `[at, at + 16)` lies below `n`.
        let line_a = unsafe { a.wrapping_add(at).cast::<[u8; 16]>().read_unaligned() };
        // SAFETY: the same range of `b`.
        let line_b = unsafe { b.wrapping_add(at).cast::<[u8; 16]>().read_unaligned() };
        if u128::from_ne_bytes(line_a) != u128::from_ne_bytes(line_b) {
            break;
        }
        at += 16;
    }
    while at < n {
        // SAFETY: `at < n`.
        let byte_a = unsafe { a.wrapping_add(at).read() };
        // SAFETY: the same byte of `b`.
        let byte_b = unsafe { b.wrapping_add(at).read() };
        if byte_a != byte_b {
            return i32::from(byte_a) - i32::from(byte_b);
        }
        at += 1;
    }
    0
}

#[cfg(test)]
mod tests;
