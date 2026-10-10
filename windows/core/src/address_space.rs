//! Free address space and the page boxes that hold it, for the address-space watch.
//!
//! The PE side walks its regions with `VirtualQuery` and feeds each one to
//! [`FreeSpace::add_region`]; the walk itself is Win32 and stays in `d3d9`,
//! the tally is here. `GlobalMemoryStatusEx` is no source for the free figure
//! under Wine: its `ullAvailVirtual` is the total minus the process working
//! set, and on macOS the working set is the resident size of the whole
//! process, the 64-bit host side included, so it falls with resident growth
//! while the 32-bit space stays free. The sum of the free regions is the space
//! an allocation can still be served from.
//!
//! [`PageBoxHolders`] splits the live page-box bytes by the holder that keeps
//! them, with what no holder claims left as `other`, so the watch's line
//! accounts for every page box rather than leaving the gap to a guess.

use core::fmt;

/// Free-address-space thresholds, in MiB, each reported once when crossed downwards.
pub const FREE_THRESHOLDS_MIB: [u64; 6] = [1536, 1024, 768, 512, 256, 128];

/// Largest-free-block thresholds, in MiB, each reported once when crossed downwards.
///
/// A fragmented space can hold plenty in total and still fail one large
/// allocation, which these catch before the free total would.
pub const LARGEST_THRESHOLDS_MIB: [u64; 3] = [512, 256, 128];

/// The threshold latch's value before the process's first sample.
pub const UNSAMPLED: u8 = u8::MAX;

/// `VirtualAlloc`'s allocation granularity: a reservation starts on a 64 KiB boundary.
pub const ALLOCATION_GRANULARITY: u64 = 64 * 1024;

/// Free address space, tallied over the regions of one walk.
pub struct FreeSpace {
    total: u64,
    largest: u64,
    regions: u32,
}

impl FreeSpace {
    /// An empty tally, for a walk to fill.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            total: 0,
            largest: 0,
            regions: 0,
        }
    }

    /// Count one region of `size` bytes at `base`; only a free one adds to the tally.
    ///
    /// A free region counts only the part an allocation can start in: from
    /// `base` rounded up to the allocation granularity to its end rounded
    /// down. A free sliver between two allocations that no reservation can
    /// use adds nothing.
    pub fn add_region(&mut self, free: bool, base: u64, size: u64) {
        self.regions = self.regions.saturating_add(1);
        if !free {
            return;
        }
        let granule = ALLOCATION_GRANULARITY;
        let start = base.div_ceil(granule).saturating_mul(granule);
        let end = base.saturating_add(size) / granule * granule;
        let usable = end.saturating_sub(start);
        self.total = self.total.saturating_add(usable);
        self.largest = self.largest.max(usable);
    }

    /// Usable free bytes summed over every free region, in MiB.
    #[must_use]
    pub const fn total_mib(&self) -> u64 {
        self.total >> 20
    }

    /// The largest single usable free region, in MiB.
    ///
    /// What a DLL load or a large streaming block needs, which runs out
    /// before the total does once the space is fragmented.
    #[must_use]
    pub const fn largest_mib(&self) -> u64 {
        self.largest >> 20
    }

    /// Regions the walk visited, free or not: what its cost scales with.
    #[must_use]
    pub const fn regions(&self) -> u32 {
        self.regions
    }
}

impl Default for FreeSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// What one sample does to a threshold latch.
#[derive(Debug, PartialEq, Eq)]
pub struct ThresholdStep {
    /// The latch's new value: the index of the next threshold still to report.
    pub next: u8,
    /// What the step reports.
    pub report: ThresholdReport,
}

/// What a [`ThresholdStep`] reports.
#[derive(Debug, PartialEq, Eq)]
pub enum ThresholdReport {
    /// The process's first sample, already below this threshold; nothing above it is reported.
    ///
    /// A process that starts with less space than the higher thresholds,
    /// such as one without large-address-aware and its 2 GiB, would
    /// otherwise warn about all of them at its first present.
    StartsBelow(u64),
    /// The process's first sample, above every threshold.
    StartsAbove,
    /// A later sample fell below this threshold, the lowest it crossed.
    Crossed(u64),
}

/// The step a sample of `value_mib` takes on a latch at `next` over `thresholds`.
///
/// `None` when a sample after the first crosses nothing new. A fall past
/// several thresholds between two samples is one report, naming the lowest.
/// Once a threshold is reported it stays reported, so a recovery and a
/// second fall report nothing.
#[must_use]
pub fn threshold_step(thresholds: &[u64], next: u8, value_mib: u64) -> Option<ThresholdStep> {
    let first = next == UNSAMPLED;
    let from = if first { 0 } else { usize::from(next) };
    let mut index = from;
    while thresholds
        .get(index)
        .is_some_and(|&threshold| value_mib < threshold)
    {
        index += 1;
    }
    let new_next = u8::try_from(index).ok()?;
    let lowest = index
        .checked_sub(1)
        .and_then(|i| thresholds.get(i))
        .copied();
    let report = match (first, lowest) {
        (true, None) => ThresholdReport::StartsAbove,
        (true, Some(threshold)) => ThresholdReport::StartsBelow(threshold),
        (false, Some(threshold)) if index > from => ThresholdReport::Crossed(threshold),
        (false, _) => return None,
    };
    Some(ThresholdStep {
        next: new_next,
        report,
    })
}

/// Whole microseconds in `cycles` of a counter running at `hz`.
///
/// Zero for a counter whose rate is not known yet, and saturating past
/// what a `u64` holds.
#[must_use]
pub fn cycles_to_micros(cycles: u64, hz: u64) -> u64 {
    if hz == 0 {
        return 0;
    }
    u64::try_from(u128::from(cycles) * 1_000_000 / u128::from(hz)).unwrap_or(u64::MAX)
}

/// Live page-box bytes split by the holder that keeps them.
///
/// Every figure is padded bytes (`PageBox::len`) held on the side that owns
/// the 32-bit address space, the unit and the side `total` counts, so
/// `other` is what none of the named holders accounts for: read-back pages
/// of a call in progress, and any holder not counted yet. Pages the native
/// encoder allocates live outside the 32-bit space and are in none of these.
pub struct PageBoxHolders {
    /// Every live page box (`page_box::live_bytes`).
    pub total: u64,
    /// Staging the live textures hold, every level not yet dropped.
    pub texture_staging: u64,
    /// System-memory and lockable render-target surfaces, and lock read-back pages.
    pub surfaces: u64,
    /// The CPU backing of live vertex and index buffers.
    pub vertex_index_backing: u64,
    /// Renamed backings and upload snapshots the encoder still reads.
    ///
    /// Kept until the encoder acknowledges its last use, which follows the
    /// GPU retiring the frame that read them.
    pub encoder_leases: u64,
    /// Texture staging only upload leases still keep, after the texture let go of it.
    ///
    /// Kept until native code drops its last owner of the pages (the upload
    /// read, a cached wrapper) and the PE side sees the acknowledgment.
    /// Staging a texture still holds counts as texture staging, not here.
    pub upload_leases: u64,
    /// Chunks of the device's texture upload-snapshot arena, read or not.
    ///
    /// Copies of the box a partial upload carries, for the levels whose
    /// partial locks would otherwise rename their staging
    /// (`crate::upload_snapshot`). Held for the device's life, at most 16 MiB.
    pub texture_snapshots: u64,
    /// Retired boxes parked in the recycle pool for reuse.
    pub pool_parked: u64,
}

impl PageBoxHolders {
    /// The bytes no named holder accounts for.
    #[must_use]
    pub const fn other(&self) -> u64 {
        self.total.saturating_sub(
            self.texture_staging
                .saturating_add(self.surfaces)
                .saturating_add(self.vertex_index_backing)
                .saturating_add(self.encoder_leases)
                .saturating_add(self.upload_leases)
                .saturating_add(self.texture_snapshots)
                .saturating_add(self.pool_parked),
        )
    }
}

impl fmt::Display for PageBoxHolders {
    /// One clause for the log line, every figure in MiB.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "page boxes {} MiB: texture staging {}, surfaces {}, vertex/index backing {}, \
             encoder leases {}, upload leases {}, texture snapshots {}, pool parked {}, other {}",
            self.total >> 20,
            self.texture_staging >> 20,
            self.surfaces >> 20,
            self.vertex_index_backing >> 20,
            self.encoder_leases >> 20,
            self.upload_leases >> 20,
            self.texture_snapshots >> 20,
            self.pool_parked >> 20,
            self.other() >> 20
        )
    }
}

#[cfg(test)]
mod tests;
