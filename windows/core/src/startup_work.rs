//! Bounded startup batches that finish before their borrowed inputs are released.

use std::{
    io,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, Scope},
};

/// Maximum concurrent jobs in one device's startup batch.
///
/// Startup compilation benefits from eight callers on a many-core host. The
/// available CPU count and batch length further bound smaller machines and batches.
const MAX_WORKERS: usize = 8;

/// Run each admitted job once, returning results in input order.
///
/// Cancellation stops admission between jobs; already admitted work finishes before
/// return. Each batch owns its workers, so simultaneous devices have independent
/// cancellation and result ownership. Failure to create a worker reduces concurrency
/// without dropping jobs. This is startup work, never a background gameplay queue.
///
/// # Panics
///
/// Panics if a job panics. All scoped job closures finish before unwinding returns.
pub fn map<T: Sync, R: Send>(
    items: &[T],
    stop: &AtomicBool,
    work: impl Fn(&T) -> R + Sync,
) -> Vec<(usize, R)> {
    let available = thread::available_parallelism().map_or_else(
        |error| {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "startup: CPU count unavailable, using one worker: {error}"
            );
            1
        },
        std::num::NonZero::get,
    );
    map_with_spawner(items, stop, available, &work, &mut NativeSpawner)
}

trait Spawner {
    fn spawn<'scope, 'env: 'scope>(
        &mut self,
        scope: &'scope Scope<'scope, 'env>,
        work: impl FnOnce() + Send + 'scope,
    ) -> io::Result<()>;
}

struct NativeSpawner;

impl Spawner for NativeSpawner {
    fn spawn<'scope, 'env: 'scope>(
        &mut self,
        scope: &'scope Scope<'scope, 'env>,
        work: impl FnOnce() + Send + 'scope,
    ) -> io::Result<()> {
        // Dropping a scoped handle leaves completion to the scope's counter/park
        // wait. Explicit join would wait on a Win32 thread handle, which Wine can
        // invalidate before a delayed join.
        thread::Builder::new()
            .name("mtld3d-startup".into())
            .spawn_scoped(scope, work)
            .map(drop)
    }
}

fn map_with_spawner<T: Sync, R: Send>(
    items: &[T],
    stop: &AtomicBool,
    available: usize,
    work: &(impl Fn(&T) -> R + Sync),
    spawner: &mut impl Spawner,
) -> Vec<(usize, R)> {
    if items.is_empty() || stop.load(Ordering::Acquire) {
        return Vec::new();
    }
    let workers = available.clamp(1, MAX_WORKERS).min(items.len());
    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::with_capacity(items.len()));
    let run = || {
        let mut completed = Vec::new();
        while !stop.load(Ordering::Acquire) {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(item) = items.get(index) else {
                break;
            };
            completed.push((index, work(item)));
        }
        results
            .lock()
            .expect("startup result lock poisoned")
            .append(&mut completed);
    };
    thread::scope(|scope| {
        for _ in 1..workers {
            if stop.load(Ordering::Acquire) {
                break;
            }
            if let Err(error) = spawner.spawn(scope, run) {
                mtld3d_shared::log_once_warn!(
                    target: crate::LOG_TARGET,
                    "startup: worker creation failed, continuing with fewer workers: {error}"
                );
                break;
            }
        }
        run();
    });
    let mut results = results.into_inner().expect("startup result lock poisoned");
    results.sort_unstable_by_key(|(index, _)| *index);
    results
}

#[cfg(test)]
mod tests;
