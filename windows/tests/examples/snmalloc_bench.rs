//! Measure allocation batches with local and remote frees under the active allocator.
//!
//! Build with the production profile for each PE target and run under Wine.
//! CSV rows report requested, padded and usable bytes plus nanoseconds per
//! allocation/free pair. Remote rows include channel handoff and scheduling.
//! Payloads are uninitialized; timings measure allocation rather than writes.

use std::{
    hint::black_box,
    mem::MaybeUninit,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use mtld3d_core::page_box::{PAGE_SIZE, PageBox};
use snmalloc_rs::SnMalloc;

#[global_allocator]
static ALLOCATOR: SnMalloc = SnMalloc;

const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;
const MAX_BATCH_BYTES: usize = 64 * MIB;
const WARMUP: Duration = Duration::from_millis(100);
const SAMPLE: Duration = Duration::from_millis(250);

/// Time complete batches without allocating bookkeeping in the measured loop.
fn measure(mut batch: impl FnMut(), count: u32) -> f64 {
    let start = Instant::now();
    while start.elapsed() < WARMUP {
        batch();
    }
    let start = Instant::now();
    let mut batches = 0_u32;
    while start.elapsed() < SAMPLE {
        batch();
        batches += 1;
    }
    start.elapsed().as_secs_f64() * 1e9 / (f64::from(batches) * f64::from(count))
}

/// Compare identical batches freed on their allocating thread or one worker.
fn compare<T: Send>(
    kind: &str,
    requested: usize,
    padded: usize,
    usable: usize,
    alloc: impl Fn() -> T,
) {
    // Keep batch counts identical across allocators with different size classes.
    let count = u32::try_from((MAX_BATCH_BYTES / padded.next_power_of_two()).clamp(1, 256))
        .expect("batch holds at most 256 allocations");
    let mut values = Vec::with_capacity(usize::try_from(count).expect("batch fits usize"));
    let local = measure(
        || {
            for _ in 0..count {
                values.push(black_box(alloc()));
            }
            values.clear();
        },
        count,
    );
    let remote = thread::scope(|scope| {
        let (send, recv) = mpsc::sync_channel::<Vec<T>>(1);
        let (return_send, return_recv) = mpsc::sync_channel(1);
        let worker = scope.spawn(move || {
            // Closing the input channel is the normal shutdown signal.
            while let Ok(mut batch) = recv.recv() {
                batch.clear();
                return_send.send(batch).expect("allocating thread is alive");
            }
        });
        let elapsed = measure(
            || {
                for _ in 0..count {
                    values.push(black_box(alloc()));
                }
                send.send(core::mem::take(&mut values))
                    .expect("freeing thread is alive");
                values = return_recv.recv().expect("freeing thread returns storage");
            },
            count,
        );
        drop(send);
        worker.join().expect("freeing thread completed");
        elapsed
    });
    println!("{kind},{requested},{padded},{usable},{count},local,{local:.2}");
    println!("{kind},{requested},{padded},{usable},{count},remote,{remote:.2}");
}

fn main() {
    println!("kind,requested,padded,usable,batch,free_thread,ns_per_pair");
    for size in [16, 256, 4096, 64 * KIB + 1, 128 * KIB + 1, 256 * KIB + 1] {
        let alloc = || vec![MaybeUninit::<u8>::uninit(); black_box(size)].into_boxed_slice();
        let probe = alloc();
        let usable = SnMalloc
            .usable_size(probe.as_ptr().cast())
            .expect("live allocation");
        drop(probe);
        compare("byte", size, size, usable, alloc);
    }
    for size in [
        PAGE_SIZE,
        64 * KIB,
        64 * KIB + 1,
        128 * KIB,
        128 * KIB + 1,
        256 * KIB,
        256 * KIB + 1,
        512 * KIB,
        512 * KIB + 1,
        MIB,
        MIB + 1,
        176 * PAGE_SIZE,
        4 * MIB,
    ] {
        let probe = PageBox::new_uninit(size);
        let usable = SnMalloc.usable_size(probe.as_ptr()).expect("live PageBox");
        drop(probe);
        compare("page", size, PageBox::padded_len(size), usable, || {
            PageBox::new_uninit(black_box(size))
        });
    }
}
