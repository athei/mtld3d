//! Per-queue presentation state.
//!
//! One record per command queue, keyed by the queue's raw address, created
//! with the queue and retired with it. It carries what presentation needs
//! beyond the frame itself: for now the test seam, a gate file the presenting
//! thread waits on before acquiring a drawable while the file exists.

use std::{
    path::PathBuf,
    sync::{Arc, LazyLock, Mutex},
    thread,
    time::Duration,
};

use mtld3d_shared::mtl_handle::{MTLCommandQueueKind, MetalHandle};
use rustc_hash::FxHashMap;

use crate::LOG_TARGET;

/// How often a parked presenter looks for the gate file to be gone.
const GATE_POLL: Duration = Duration::from_millis(1);

/// The live records, keyed by the raw `MTLCommandQueue*` address.
///
/// `LazyLock` because the map's constructor is not `const`.
static PRESENTERS: LazyLock<Mutex<FxHashMap<u64, Arc<PresentState>>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

/// What one queue's presentation keeps between frames.
pub struct PresentState {
    /// The gate file `debug.presentGateFile` named, `None` = no gate.
    gate: Option<PathBuf>,
}

/// Create the record for `queue`.
///
/// A record already registered under the address is replaced with a warning:
/// the address belongs to one live queue at a time, so a stale record means
/// a retire that never ran.
pub fn register(queue: MetalHandle<MTLCommandQueueKind>, gate: Option<PathBuf>) {
    if let Some(path) = &gate {
        log::info!(
            target: LOG_TARGET,
            "presenter: gated at {} (parks before each drawable while it exists)",
            path.display(),
        );
    }
    let state = Arc::new(PresentState { gate });
    let mut map = PRESENTERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if map.insert(queue.raw(), state).is_some() {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "presenter: queue {:#x} registered twice; the earlier record was never retired",
            queue.raw(),
        );
    }
}

/// The record for `queue`, if it is registered.
pub fn find(queue: MetalHandle<MTLCommandQueueKind>) -> Option<Arc<PresentState>> {
    PRESENTERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&queue.raw())
        .cloned()
}

/// Retire the record for `queue`.
pub fn unregister(queue: MetalHandle<MTLCommandQueueKind>) {
    let removed = PRESENTERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&queue.raw());
    if removed.is_none() {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "presenter: queue {:#x} retired without a record",
            queue.raw(),
        );
    }
}

/// Park while the gate file exists.
///
/// The seam the test suite uses to hold presentation still: nothing else
/// blocks here, and a record without a gate returns at once.
pub fn hold_at_gate(state: &PresentState) {
    let Some(gate) = &state.gate else {
        return;
    };
    while gate.exists() {
        thread::sleep(GATE_POLL);
    }
}
