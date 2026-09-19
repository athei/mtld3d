//! Cumulative pool outcomes for the performance summary.

/// Mutually exclusive outcomes of one acquire attempt.
pub enum Acquire {
    Hit,
    Empty,
    Oversize,
    Disabled,
}

/// Mutually exclusive outcomes of one recycle attempt.
pub enum Recycle {
    Parked,
    Full,
    Oversize,
    Disabled,
}

/// Totals guarded by the owning pool's mutex.
pub struct Diagnostics {
    acquire: [u64; 4],
    recycle: [u64; 4],
    oversize_bytes: u64,
    largest_oversize: usize,
}

impl Diagnostics {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            acquire: [0; 4],
            recycle: [0; 4],
            oversize_bytes: 0,
            largest_oversize: 0,
        }
    }

    pub fn acquire(&mut self, outcome: Acquire, logical_len: usize) {
        if matches!(outcome, Acquire::Oversize) {
            self.oversize_bytes = self.oversize_bytes.saturating_add(logical_len as u64);
            self.largest_oversize = self.largest_oversize.max(logical_len);
        }
        let count = &mut self.acquire[outcome as usize];
        *count = count.saturating_add(1);
    }

    pub const fn recycle(&mut self, outcome: Recycle) {
        let count = &mut self.recycle[outcome as usize];
        *count = count.saturating_add(1);
    }

    #[must_use]
    pub fn summary(&self) -> String {
        let [hit, empty, oversize, disabled] = self.acquire;
        let [parked, full, recycle_oversize, recycle_disabled] = self.recycle;
        format!(
            "pagebox-pool cumulative: hit={hit} empty={empty} oversize={oversize} \
             disabled={disabled} oversize_requested_bytes={} largest_oversize_request={} \
             recycle_parked={parked} recycle_cap={full} recycle_oversize={recycle_oversize} \
             recycle_disabled={recycle_disabled}",
            self.oversize_bytes, self.largest_oversize,
        )
    }
}
