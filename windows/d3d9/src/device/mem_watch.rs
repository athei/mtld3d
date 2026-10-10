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
//! vertex/index backing, encoder leases, upload leases, the recycle pool)
//! with the rest as `other`, and the texture staging and vertex/index backing
//! are split again by the class that decides whether the copy can be released
//! at all, so the line names which holder keeps the space rather than leaving
//! it to a guess. Beside them goes what `d3d9.dll`'s heap has committed, page
//! boxes included, which bounds everything else the image allocates.
//!
//! The walk runs on a thread of the device's own (`mtld3d-mem-watch`), never
//! on the presenting thread, which pays a counter bump and an empty-channel
//! check per present and hands the thread a sample index every
//! `SAMPLE_EVERY` presents. The thread walks, advances the threshold latches,
//! reads the process-wide holder figures, formats and logs. Two figures only
//! the API thread can read safely: the live textures' footprint (staging the
//! API thread changes without a lock) and the upload leases (behind the
//! encoder's retirement lock). The thread asks for them only when a line is
//! due, and the next present answers, so a line is logged a present or two
//! after its walk.
//!
//! The periodic breakdown logs at debug on its own target, so
//! `RUST_LOG=mtld3d::d3d9::mem_watch=debug` turns it on without the rest of
//! the layer's debug output; the threshold warnings log at warn on the same
//! target and show by default. A 64-bit process has no address space to run
//! out of and a slow walk, so it starts no thread, skips the walk and the
//! thresholds, and keeps the breakdown, logged from the present, which still
//! says what the layer holds in memory.

use core::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::{
    sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
    thread::{self, JoinHandle},
};

use log::{debug, info, warn};
use mtld3d_core::{
    address_space::{
        FREE_THRESHOLDS_MIB, FreeSpace, LARGEST_THRESHOLDS_MIB, PageBoxHolders, ThresholdReport,
        UNSAMPLED, cycles_to_micros, threshold_step,
    },
    thread_wait::wait_until_finished,
};
use mtld3d_shared::tsc::{rdtsc, tsc_hz};

use super::DeviceInner;
use crate::crash::{address_space_map, free_space};

const LOG_TARGET: &str = "mtld3d::d3d9::mem_watch";

/// Presents between two samples, about five seconds at 120 presents a second.
///
/// The walk costs a `VirtualQuery` per region, about 0.75 microseconds each
/// in a test process under Wine, so a game with a few thousand regions takes
/// a few milliseconds a sample; the debug line reports the real cost. The
/// watch thread pays it, so the cadence sets how fresh the figures are, not
/// what a present costs.
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
/// The watch thread is per device for the same reason, and because the
/// figures it asks for are the device's.
pub struct MemWatchState {
    presents: AtomicU32,
    /// The device's watch thread; `None` in a 64-bit build, or when the thread did not start.
    ///
    /// Without a thread the present logs the breakdown itself, saying the
    /// space was not walked, and no threshold is watched for this device.
    worker: Option<MemWatchWorker>,
}

impl MemWatchState {
    /// Start the watch, with its thread in a 32-bit build.
    pub fn new() -> Self {
        Self {
            presents: AtomicU32::new(0),
            worker: if WALKS { MemWatchWorker::spawn() } else { None },
        }
    }

    /// The channels to the watch thread; `None` when the device has no thread.
    ///
    /// The worker lets its link go only in its own drop, which no present
    /// can overlap, so a device with a thread always finds its link here.
    fn link(&self) -> Option<&ApiLink> {
        self.worker.as_ref().and_then(|worker| worker.link.as_ref())
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
    staging_render_target: u64,
    staging_default_dynamic: u64,
    staging_other: u64,
}

impl TextureFootprint {
    const fn staging(&self) -> u64 {
        self.staging_default_static
            + self.staging_render_target
            + self.staging_default_dynamic
            + self.staging_other
    }
}

/// The device's share of a line, computed on the API thread when the watch thread asks.
///
/// `live_texture_footprint` reads texture staging the API thread changes
/// without a lock, and `upload_lease_bytes` takes the encoder's retirement
/// lock, which the API thread's own maintenance contends for, so the watch
/// thread gets both as plain figures rather than reading them itself.
struct DeviceFigures {
    footprint: TextureFootprint,
    upload_leases: u64,
}

/// One sample's walk, with what it cost.
struct Walk {
    space: FreeSpace,
    cycles: u64,
}

impl Walk {
    /// Walk the address space now, timing the walk.
    fn now() -> Self {
        let start = rdtsc();
        let space = free_space();
        Self {
            space,
            cycles: rdtsc().wrapping_sub(start),
        }
    }

    /// The breakdown's free-space clause: what the walk found and what it cost.
    fn describe(&self) -> String {
        format!(
            "{} MiB free, largest free block {} MiB, walked {} regions in {} us",
            self.space.total_mib(),
            self.space.largest_mib(),
            self.space.regions(),
            cycles_to_micros(self.cycles, tsc_hz())
        )
    }
}

/// One sample as the watch thread took it: the walk, the latch reports, and the lines due.
struct Sample {
    walk: Walk,
    free: Option<ThresholdReport>,
    largest: Option<ThresholdReport>,
    /// Whether this sample logs the debug breakdown.
    breakdown: bool,
}

impl Sample {
    /// Walk for sample `index` and advance both threshold latches on what the walk found.
    fn take(index: u32) -> Self {
        let walk = Walk::now();
        let free = advance(
            &NEXT_FREE_THRESHOLD,
            &FREE_THRESHOLDS_MIB,
            walk.space.total_mib(),
        );
        let largest = advance(
            &NEXT_LARGEST_THRESHOLD,
            &LARGEST_THRESHOLDS_MIB,
            walk.space.largest_mib(),
        );
        Self {
            walk,
            free,
            largest,
            breakdown: breakdown_due(index),
        }
    }

    /// Whether the sample logs anything, and so needs the device figures.
    ///
    /// The first-sample info line needs none, but it comes once per process
    /// and is asked for like the rest, so the lines keep their order.
    fn lines_due(&self) -> bool {
        self.breakdown
            || [&self.free, &self.largest].into_iter().any(|report| {
                matches!(
                    report,
                    Some(ThresholdReport::StartsBelow(_) | ThresholdReport::Crossed(_))
                )
            })
    }

    /// Log the breakdown, then the threshold lines, with the device's figures.
    ///
    /// One region map follows the warnings of a sample, however many
    /// thresholds it crossed.
    fn log(&self, figures: &DeviceFigures) {
        if self.breakdown {
            log_breakdown(&self.walk.describe(), figures);
        }
        let free = self.walk.space.total_mib();
        let largest = self.walk.space.largest_mib();
        let mut crossed = false;
        for (what, value, report) in [
            ("free", free, &self.free),
            ("largest free block", largest, &self.largest),
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
                    let fp = &figures.footprint;
                    warn!(
                        target: LOG_TARGET,
                        "address space: {what} {value} MiB (below {threshold} MiB); {free} MiB \
                         free, largest free block {largest} MiB; mtld3d holds {} textures with \
                         {} MiB of mip data; {}; d3d9.dll heap {} MiB committed",
                        fp.count,
                        fp.mip_bytes >> 20,
                        page_box_holders(figures),
                        heap_committed_bytes() >> 20
                    );
                }
            }
        }
        if crossed {
            warn!(target: LOG_TARGET, "address space map: {}", address_space_map());
        }
    }
}

/// The device's watch thread and the API thread's ends of the channels to it.
///
/// Its drop lets the channel ends go before it waits for the thread: a watch
/// thread blocked on a sample or on the device figures wakes to the
/// disconnect and ends, and one still walking ends at its next send or
/// receive. A receive still hands over a sample queued before the
/// disconnect, so the wait is bounded by two walks.
struct MemWatchWorker {
    /// The channel ends; `None` only inside `drop`, which lets them go first.
    link: Option<ApiLink>,
    /// The thread, waited for in `drop` and never joined; `None` only there.
    thread: Option<JoinHandle<()>>,
}

impl MemWatchWorker {
    /// Start the device's watch thread; `None`, logged once, when it cannot start.
    fn spawn() -> Option<Self> {
        let (samples, sample_rx) = mpsc::sync_channel(1);
        let (request_tx, figure_requests) = mpsc::sync_channel(1);
        let (figures, figures_rx) = mpsc::sync_channel(1);
        let link = WatchLink {
            samples: sample_rx,
            figure_requests: request_tx,
            figures: figures_rx,
        };
        match thread::Builder::new()
            .name("mtld3d-mem-watch".into())
            .spawn(move || link.run())
        {
            Ok(handle) => Some(Self {
                link: Some(ApiLink {
                    samples,
                    figure_requests,
                    figures,
                }),
                thread: Some(handle),
            }),
            Err(error) => {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "address space: the watch thread did not start ({error}); the device's \
                     watch logs its breakdown without walking and reports no threshold"
                );
                None
            }
        }
    }
}

impl Drop for MemWatchWorker {
    fn drop(&mut self) {
        drop(self.link.take());
        if let Some(thread) = self.thread.take() {
            wait_until_finished(thread);
        }
    }
}

/// The API thread's ends of the channels to the device's watch thread.
struct ApiLink {
    /// Sample indices for the thread, at most one waiting; sent without waiting.
    samples: SyncSender<u32>,
    /// The thread's request for the device figures, at most one outstanding.
    figure_requests: Receiver<()>,
    /// The answer to a request; the thread has taken the previous one before it asks again.
    figures: SyncSender<DeviceFigures>,
}

impl ApiLink {
    /// Hand the thread sample `index`, or skip it when the thread is still behind.
    fn offer_sample(&self, index: u32) {
        match self.samples.try_send(index) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "address space: sample {index} skipped, the watch thread has not taken the \
                 one before it yet"
            ),
            Err(TrySendError::Disconnected(_)) => mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "address space: the watch thread has ended; sample {index} and the device's \
                 later samples are dropped"
            ),
        }
    }

    /// Whether the thread waits for the device figures.
    fn figures_requested(&self) -> bool {
        match self.figure_requests.try_recv() {
            Ok(()) => true,
            Err(TryRecvError::Empty) => false,
            Err(TryRecvError::Disconnected) => {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "address space: the watch thread has ended; the device answers no more \
                     figure requests"
                );
                false
            }
        }
    }

    /// Hand the thread the figures it asked for.
    fn answer(&self, figures: DeviceFigures) {
        if let Err(error) = self.figures.try_send(figures) {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "address space: the watch thread did not take the device figures ({error}); \
                 that sample logs nothing"
            );
        }
    }
}

/// The watch thread's ends of the channels whose other ends the device holds.
struct WatchLink {
    samples: Receiver<u32>,
    figure_requests: SyncSender<()>,
    figures: Receiver<DeviceFigures>,
}

impl WatchLink {
    /// One walk per sample and the lines it makes due, until the device lets its ends go.
    ///
    /// A failed send or receive means the device is being released, which is
    /// how the thread ends; the sample in progress logs nothing then.
    fn run(self) {
        while let Ok(index) = self.samples.recv() {
            let sample = Sample::take(index);
            if !sample.lines_due() {
                continue;
            }
            if self.figure_requests.send(()).is_err() {
                break;
            }
            let Ok(figures) = self.figures.recv() else {
                break;
            };
            sample.log(&figures);
        }
    }
}

impl DeviceInner {
    /// Count, total mip bytes, and resident staging bytes of every live texture.
    ///
    /// The staging split names who still holds a system copy: default-pool
    /// static (droppable after upload), default-pool render targets (a 2D
    /// level holds staging only once a CPU path has used it, a cube face
    /// always), default-pool dynamic, and the lockable pools.
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
            staging_render_target: 0,
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
            let usage = ti.d3d_usage();
            if ti.d3d_pool() != mtld3d_types::D3DPOOL_DEFAULT {
                fp.staging_other += resident.padded;
            } else if usage & mtld3d_types::D3DUSAGE_DYNAMIC != 0 {
                fp.staging_default_dynamic += resident.padded;
            } else if usage & mtld3d_types::D3DUSAGE_RENDERTARGET != 0 {
                fp.staging_render_target += resident.padded;
            } else {
                fp.staging_default_static += resident.padded;
            }
        }
        drop(live);
        fp
    }

    /// The figures of this device that only the API thread reads.
    fn device_figures(&self) -> DeviceFigures {
        DeviceFigures {
            footprint: self.live_texture_footprint(),
            upload_leases: self.encoder.upload_lease_bytes(),
        }
    }

    /// Count the present, hand the watch thread a sample when one is due, and answer its request.
    ///
    /// Called from `present` after its stall timer has stopped. Nothing here
    /// waits for the watch thread. A device without one logs the breakdown
    /// here instead, on the same cadence.
    pub fn mem_watch_present(&self) {
        let present = self.mem_watch.presents.fetch_add(1, Ordering::Relaxed);
        let sample = present
            .is_multiple_of(SAMPLE_EVERY)
            .then_some(present / SAMPLE_EVERY);
        let Some(link) = self.mem_watch.link() else {
            if let Some(index) = sample
                && breakdown_due(index)
            {
                let space = if WALKS {
                    "not walked, the watch thread did not start"
                } else {
                    "not walked in a 64-bit process"
                };
                log_breakdown(space, &self.device_figures());
            }
            return;
        };
        if link.figures_requested() {
            link.answer(self.device_figures());
        }
        if let Some(index) = sample {
            link.offer_sample(index);
        }
    }
}

/// Whether sample `index` logs the debug breakdown: every second one, with debug on.
fn breakdown_due(index: u32) -> bool {
    index.is_multiple_of(REPORT_EVERY_SAMPLES)
        && log::log_enabled!(target: LOG_TARGET, log::Level::Debug)
}

/// The live page boxes split by holder, with the device's textures and leases as holders.
///
/// The process-wide figures are read here, right after the device's
/// arrived, so the two sets are as close in time as the hand-off allows.
fn page_box_holders(figures: &DeviceFigures) -> PageBoxHolders {
    PageBoxHolders {
        total: mtld3d_core::page_box::live_bytes(),
        texture_staging: figures.footprint.staging(),
        surfaces: mtld3d_core::held_pages::live_surface_bytes(),
        vertex_index_backing: mtld3d_core::buffer_backing::live_backing_bytes().total(),
        encoder_leases: mtld3d_core::held_pages::live_encoder_lease_bytes(),
        upload_leases: figures.upload_leases,
        pool_parked: crate::page_box_pool::PAGEBOX_POOL.pooled_bytes() as u64,
    }
}

/// The periodic debug line: the free space as `space` gives it, and every holder.
fn log_breakdown(space: &str, figures: &DeviceFigures) {
    let fp = &figures.footprint;
    let vbib = mtld3d_core::buffer_backing::live_backing_bytes();
    debug!(
        target: LOG_TARGET,
        "address space: {space}; mtld3d holds {} textures with {} MiB of mip data; {}; \
         d3d9.dll heap {} MiB committed; texture staging split default static {} / \
         render target {} / default dynamic {} / other {}, {} MiB before page rounding; \
         vertex/index backing split writeonly static {} / dynamic {} / other {}; \
         locks on static default textures {}",
        fp.count,
        fp.mip_bytes >> 20,
        page_box_holders(figures),
        heap_committed_bytes() >> 20,
        fp.staging_default_static >> 20,
        fp.staging_render_target >> 20,
        fp.staging_default_dynamic >> 20,
        fp.staging_other >> 20,
        fp.staging_requested >> 20,
        vbib.write_only_static >> 20,
        vbib.dynamic >> 20,
        vbib.other >> 20,
        crate::texture::default_static_lock_count()
    );
}

/// Bytes `d3d9.dll`'s snmalloc holds committed, its page boxes included.
///
/// snmalloc's backend counts the chunks it has committed and handed to its
/// allocators, two atomic loads: every heap block of this image, the page
/// boxes, and what the per-thread caches keep for reuse. It is not the
/// image's share of the address space, which reserves ahead of commit.
fn heap_committed_bytes() -> u64 {
    snmalloc_rs::SnMalloc::memory_stats().current_memory_usage as u64
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
