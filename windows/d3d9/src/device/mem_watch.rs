//! Address-space watch for 32-bit games.
//!
//! A large-address-aware i386 process has 4 GiB of virtual address space
//! and every texture streamed, every shader compiled and every one of our
//! staging copies lives inside it. When it runs out, allocations fail and
//! the game usually follows a garbage pointer a few frames later, far from
//! the cause. This watch samples `GlobalMemoryStatusEx` every few presents
//! and logs one line per threshold crossed on the way down, with the sizes
//! of the pools mtld3d itself holds, so the log says how close the process
//! was and who owned the space.

use core::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use log::warn;

use super::DeviceInner;
use crate::{LOG_TARGET, crash::avail_virtual_mib};

/// Presents between two samples; the query is a syscall, this keeps it off the frame time.
const SAMPLE_EVERY: u32 = 120;

/// Free-address-space thresholds, in MiB, each logged once when crossed downwards.
const THRESHOLDS_MIB: [u64; 6] = [1536, 1024, 768, 512, 256, 128];

static PRESENTS: AtomicU32 = AtomicU32::new(0);
/// Index of the next threshold to report; thresholds above it were already logged.
static NEXT_THRESHOLD: AtomicU8 = AtomicU8::new(0);

impl DeviceInner {
    /// Count and total mip bytes of every live texture, whatever its pool.
    fn live_texture_footprint(&self) -> (usize, u64) {
        let live = self
            .live_textures
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let bytes = live
            .iter()
            .map(|&t| {
                // SAFETY: the registry holds every live texture until its
                // release deregisters it under the same lock.
                unsafe { (*t).allocated_bytes() }
            })
            .sum();
        (live.len(), bytes)
    }

    /// Sample the free virtual address space and log threshold crossings.
    pub fn mem_watch_present(&self) {
        if !PRESENTS
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(SAMPLE_EVERY)
        {
            return;
        }
        let Some(avail) = avail_virtual_mib() else {
            return;
        };
        let mut next = NEXT_THRESHOLD.load(Ordering::Relaxed);
        while let Some(&threshold) = THRESHOLDS_MIB.get(usize::from(next))
            && avail < threshold
        {
            next += 1;
            NEXT_THRESHOLD.store(next, Ordering::Relaxed);
            let (texture_count, texture_bytes) = self.live_texture_footprint();
            warn!(
                target: LOG_TARGET,
                "address space: {avail} MiB free (below {threshold} MiB); mtld3d holds \
                 {texture_count} textures with {} MiB of mip data ({} MiB default pool), \
                 retained vertex/index buffers {} MiB",
                texture_bytes >> 20,
                self.vram_bytes_used.load(Ordering::Relaxed) >> 20,
                self.vbib_retained_bytes.load(Ordering::Relaxed) >> 20
            );
        }
    }
}
