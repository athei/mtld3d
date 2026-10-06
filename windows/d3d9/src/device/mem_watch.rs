//! Address-space watch for 32-bit games.
//!
//! A large-address-aware i386 process has 4 GiB of virtual address space
//! and every texture streamed, every shader compiled and every one of our
//! staging copies lives inside it. When it runs out, allocations fail and
//! the game usually follows a garbage pointer a few frames later, far from
//! the cause. This watch walks the address space every few presents, sums its
//! usable free regions (`crash::free_space`), and logs one warning per
//! threshold crossed on the way down, of the free total or of the largest free
//! block, with the region map and the page boxes mtld3d holds, so the log says
//! how close the process was and who owned the space.
//!
//! The page boxes are reported by holder (texture staging, surfaces,
//! vertex/index backing, encoder leases, the recycle pool) with the rest as
//! `other`, and the texture staging and vertex/index backing are split again
//! by the class that decides whether the copy can be released at all, so the
//! line names which holder keeps the space rather than leaving it to a guess.
//!
//! The periodic breakdown logs at debug on its own target, so
//! `RUST_LOG=mtld3d::d3d9::mem_watch=debug` turns it on without the rest of
//! the layer's debug output; the threshold warnings log at warn on the same
//! target and show by default. A 64-bit process has no address space to run
//! out of and a slow walk, so it skips the walk and the thresholds and keeps
//! the breakdown, which still says what the layer holds in memory.

use core::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use log::{debug, info, warn};
use mtld3d_core::address_space::{
    FREE_THRESHOLDS_MIB, FreeSpace, LARGEST_THRESHOLDS_MIB, PageBoxHolders, ThresholdReport,
    UNSAMPLED, cycles_to_micros, threshold_step,
};
use mtld3d_shared::tsc::{rdtsc, tsc_hz};

use super::DeviceInner;
use crate::crash::{address_space_map, free_space};

const LOG_TARGET: &str = "mtld3d::d3d9::mem_watch";

/// Presents between two samples, about five seconds at 120 presents a second.
///
/// The walk costs a `VirtualQuery` per region, about 0.75 microseconds each
/// in a test process under Wine, so a game with a few thousand regions would
/// pay a few milliseconds a sample; the debug line reports the real cost.
const SAMPLE_EVERY: u32 = 600;

/// Samples between two unconditional log lines: a time series of the space at ~10 s.
const REPORT_EVERY_SAMPLES: u32 = 2;

/// Whether this build's process can run out of address space: a 32-bit one can.
const WALKS: bool = cfg!(target_pointer_width = "32");

/// Index of the next free-total threshold to report, or `UNSAMPLED` before the first sample.
///
/// Process-wide: free address space is a property of the process, so a
/// threshold is crossed once however many devices are live.
static NEXT_FREE_THRESHOLD: AtomicU8 = AtomicU8::new(UNSAMPLED);

/// Index of the next largest-free-block threshold to report, or `UNSAMPLED` before the first.
///
/// Process-wide on the same argument as [`NEXT_FREE_THRESHOLD`]: the
/// largest free block is a property of the one address space.
static NEXT_LARGEST_THRESHOLD: AtomicU8 = AtomicU8::new(UNSAMPLED);

/// The watch's per-device state, embedded in `DeviceInner`.
///
/// The sampling counter is per device because the cadence is: on one counter,
/// two presenting devices reach `SAMPLE_EVERY` twice as fast as one does, and
/// each samples on whichever of its presents happened to land on the multiple.
pub struct MemWatchState {
    presents: AtomicU32,
}

impl MemWatchState {
    pub const fn new() -> Self {
        Self {
            presents: AtomicU32::new(0),
        }
    }
}

/// What the live textures hold in the 32-bit address space.
///
/// Staging is in padded bytes, split by class; `staging_requested` is the
/// same staging at the lengths the levels asked for, before page rounding.
struct TextureFootprint {
    count: usize,
    mip_bytes: u64,
    staging_requested: u64,
    staging_default_static: u64,
    staging_default_dynamic: u64,
    staging_other: u64,
}

impl TextureFootprint {
    const fn staging(&self) -> u64 {
        self.staging_default_static + self.staging_default_dynamic + self.staging_other
    }
}

/// One sample's walk, with what it cost.
struct Walk {
    space: FreeSpace,
    cycles: u64,
}

impl DeviceInner {
    /// Count, total mip bytes, and resident staging bytes of every live texture.
    ///
    /// The staging split names who still holds a system copy: default-pool
    /// static (droppable after upload), default-pool dynamic, and the
    /// lockable pools.
    fn live_texture_footprint(&self) -> TextureFootprint {
        let live = self
            .live_textures
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut fp = TextureFootprint {
            count: live.len(),
            mip_bytes: 0,
            staging_requested: 0,
            staging_default_static: 0,
            staging_default_dynamic: 0,
            staging_other: 0,
        };
        for &t in live.values() {
            // SAFETY: the registry holds every live texture until its
            // release deregisters it under the same lock.
            let ti = unsafe { &*t };
            fp.mip_bytes += ti.allocated_bytes();
            let resident = ti.resident_staging();
            fp.staging_requested += resident.requested;
            if ti.d3d_pool() == mtld3d_types::D3DPOOL_DEFAULT {
                if ti.d3d_usage() & mtld3d_types::D3DUSAGE_DYNAMIC == 0 {
                    fp.staging_default_static += resident.padded;
                } else {
                    fp.staging_default_dynamic += resident.padded;
                }
            } else {
                fp.staging_other += resident.padded;
            }
        }
        drop(live);
        fp
    }

    /// The live page boxes split by holder, with this device's textures as the staging holder.
    fn page_box_holders(textures: &TextureFootprint) -> PageBoxHolders {
        PageBoxHolders {
            total: mtld3d_core::page_box::live_bytes(),
            texture_staging: textures.staging(),
            surfaces: mtld3d_core::held_pages::live_surface_bytes(),
            vertex_index_backing: mtld3d_core::buffer_backing::live_backing_bytes().total(),
            encoder_leases: mtld3d_core::held_pages::live_encoder_lease_bytes(),
            pool_parked: crate::page_box_pool::PAGEBOX_POOL.pooled_bytes() as u64,
        }
    }

    /// Sample the free virtual address space and log threshold crossings.
    ///
    /// Called from `present` after its stall timer has stopped, so the walk
    /// is charged to the API time between presents, not to the present block.
    pub fn mem_watch_present(&self) {
        let present = self.mem_watch.presents.fetch_add(1, Ordering::Relaxed);
        if !present.is_multiple_of(SAMPLE_EVERY) {
            return;
        }
        let walk = WALKS.then(|| {
            let start = rdtsc();
            let space = free_space();
            Walk {
                space,
                cycles: rdtsc().wrapping_sub(start),
            }
        });
        if (present / SAMPLE_EVERY).is_multiple_of(REPORT_EVERY_SAMPLES)
            && log::log_enabled!(target: LOG_TARGET, log::Level::Debug)
        {
            self.log_breakdown(walk.as_ref());
        }
        if let Some(walk) = walk {
            self.report_thresholds(&walk.space);
        }
    }

    /// The periodic debug line: the free space, what the walk cost, and every holder.
    fn log_breakdown(&self, walk: Option<&Walk>) {
        let fp = self.live_texture_footprint();
        let vbib = mtld3d_core::buffer_backing::live_backing_bytes();
        let space = walk.map_or_else(
            || "not walked in a 64-bit process".to_owned(),
            |walk| {
                format!(
                    "{} MiB free, largest free block {} MiB, walked {} regions in {} us",
                    walk.space.total_mib(),
                    walk.space.largest_mib(),
                    walk.space.regions(),
                    cycles_to_micros(walk.cycles, tsc_hz())
                )
            },
        );
        debug!(
            target: LOG_TARGET,
            "address space: {space}; mtld3d holds {} textures with {} MiB of mip data; {}; \
             texture staging split default static {} / default dynamic {} / other {}, \
             {} MiB before page rounding; vertex/index backing split writeonly static {} / \
             dynamic {} / other {}; locks on static default textures {}",
            fp.count,
            fp.mip_bytes >> 20,
            Self::page_box_holders(&fp),
            fp.staging_default_static >> 20,
            fp.staging_default_dynamic >> 20,
            fp.staging_other >> 20,
            fp.staging_requested >> 20,
            vbib.write_only_static >> 20,
            vbib.dynamic >> 20,
            vbib.other >> 20,
            crate::texture::default_static_lock_count()
        );
    }

    /// Advance both threshold latches on this sample and warn about what they crossed.
    ///
    /// One region map follows the warnings of a sample, however many
    /// thresholds it crossed.
    fn report_thresholds(&self, space: &FreeSpace) {
        let free = space.total_mib();
        let largest = space.largest_mib();
        let free_report = advance(&NEXT_FREE_THRESHOLD, &FREE_THRESHOLDS_MIB, free);
        let largest_report = advance(&NEXT_LARGEST_THRESHOLD, &LARGEST_THRESHOLDS_MIB, largest);
        let mut crossed = false;
        for (what, value, report) in [
            ("free", free, free_report),
            ("largest free block", largest, largest_report),
        ] {
            match report {
                None | Some(ThresholdReport::StartsAbove) => {}
                Some(ThresholdReport::StartsBelow(threshold)) => info!(
                    target: LOG_TARGET,
                    "address space: {what} {value} MiB at the first sample, already below \
                     {threshold} MiB; reporting only lower thresholds"
                ),
                Some(ThresholdReport::Crossed(threshold)) => {
                    crossed = true;
                    let fp = self.live_texture_footprint();
                    warn!(
                        target: LOG_TARGET,
                        "address space: {what} {value} MiB (below {threshold} MiB); {free} MiB \
                         free, largest free block {largest} MiB; mtld3d holds {} textures with \
                         {} MiB of mip data; {}",
                        fp.count,
                        fp.mip_bytes >> 20,
                        Self::page_box_holders(&fp)
                    );
                }
            }
        }
        if crossed {
            warn!(target: LOG_TARGET, "address space map: {}", address_space_map());
        }
    }
}

/// Step `latch` over `thresholds` with this sample's `value_mib`, once per process.
///
/// `None` when nothing new is reported, including when another device's
/// sample advanced the latch first.
fn advance(latch: &AtomicU8, thresholds: &[u64], value_mib: u64) -> Option<ThresholdReport> {
    let next = latch.load(Ordering::Relaxed);
    let step = threshold_step(thresholds, next, value_mib)?;
    latch
        .compare_exchange(next, step.next, Ordering::Relaxed, Ordering::Relaxed)
        .ok()?;
    Some(step.report)
}
