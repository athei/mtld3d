//! A thread that exits hands its allocator back for the next thread to take.
//!
//! `d3d9.dll` allocates through snmalloc, which gives every thread that
//! allocates an allocator of its own and keeps it until the thread exits. Then
//! the allocator is torn down (what it caches goes back, the frees it owes other
//! threads are posted) and returned to a process-wide pool, from which the next
//! thread that allocates takes it. That happens only if the DLL hears about the
//! exit: the x86 DLLs hear through MSVC's CRT, the ARM64X ones through the TLS
//! callback in `windows/d3d9/src/arm64_crt.rs`. A DLL that does not hear
//! leaks the allocator of every thread that ever called into it, with every
//! slab the allocator holds.
//!
//! The pool is what makes the teardown visible from outside. Threads run one
//! after another, each creating an `IDirect3D9` and releasing it before it
//! exits. With the teardown, each thread takes up an allocator an earlier one
//! returned, whose free lists hand back the memory the earlier interfaces
//! occupied, so the interfaces land on a few addresses over and over. Without
//! it, each thread builds a new allocator on memory no earlier thread has
//! used, and every interface lands at an address of its own.
//!
//! A binary of its own, because the pool is process-wide: the threads of other
//! tests would take and return allocators in between.

use std::collections::BTreeSet;

use mtld3d_tests::{Harness, spawn_scoped};

/// How many threads create and release an interface, one after another.
const THREADS: usize = 32;

/// The most distinct interface addresses the threads may see between them.
///
/// Leaked allocators give every thread a new address, `THREADS` in all. The
/// returned ones settle on two addresses that alternate, one per allocator in
/// the pool, since the logging thread every interface starts takes and returns
/// one too; a few more appear while the pool fills and when a slab changes,
/// five in all over 32 threads and over 64.
const MAX_DISTINCT_ADDRESSES: usize = THREADS / 4;

#[test]
fn an_exiting_thread_returns_its_allocator() {
    let addresses: BTreeSet<usize> = (0..THREADS)
        .map(|_| {
            std::thread::scope(|scope| {
                spawn_scoped(scope, || Harness::factory_only().factory().addr())
                    .join()
                    .expect("the worker creating and releasing an IDirect3D9 panicked")
            })
        })
        .collect();
    // Shown with `--nocapture`: the count is the measurement, even on a pass.
    eprintln!(
        "{THREADS} threads got IDirect3D9 at {} distinct addresses: {addresses:x?}",
        addresses.len()
    );
    assert!(
        addresses.len() <= MAX_DISTINCT_ADDRESSES,
        "{THREADS} threads, one after another, got IDirect3D9 at {} distinct addresses \
         (at most {MAX_DISTINCT_ADDRESSES} expected): {addresses:x?}; a thread's allocator \
         is not returned when the thread exits",
        addresses.len()
    );
}
