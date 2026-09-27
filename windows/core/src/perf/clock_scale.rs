//! Convert elapsed source clock ticks after both device clock domains are calibrated.

use super::{FrameCounters, FrameTiming};

impl FrameCounters {
    pub(super) fn rescale_durations(&mut self, source_hz: u64, target_hz: u64) {
        self.api_cycles_by_category.rescale(source_hz, target_hz);
        self.query_wait_cycles.rescale(source_hz, target_hz);
        self.device_sub_cycles.rescale(source_hz, target_hz);
        self.bind_sub_cycles.rescale(source_hz, target_hz);
        self.surface_sub_cycles.rescale(source_hz, target_hz);
        self.draw_snapshot_cycles.rescale(source_hz, target_hz);
        self.draw_snapshot_stages_cycles
            .rescale(source_hz, target_hz);
        self.draw_snapshot_c_ff_cycles.rescale(source_hz, target_hz);
        self.draw_snapshot_c_pr_cycles.rescale(source_hz, target_hz);
        self.draw_snapshot_keys_cycles.rescale(source_hz, target_hz);
        self.draw_snapshot_bumps_cycles
            .rescale(source_hz, target_hz);
        self.draw_push_op_cycles.rescale(source_hz, target_hz);
    }
}

impl FrameTiming {
    pub(super) fn rescale_durations(&mut self, source_hz: u64, target_hz: u64) {
        self.present_block_cycles.rescale(source_hz, target_hz);
        self.frame_total_cycles.rescale(source_hz, target_hz);
    }
}

trait Rescale {
    fn rescale(&mut self, source_hz: u64, target_hz: u64);
}
impl Rescale for u64 {
    fn rescale(&mut self, source_hz: u64, target_hz: u64) {
        *self = scale_ticks(*self, source_hz, target_hz);
    }
}
impl<const N: usize> Rescale for [u64; N] {
    fn rescale(&mut self, source_hz: u64, target_hz: u64) {
        for value in self {
            value.rescale(source_hz, target_hz);
        }
    }
}
pub(super) fn scale_ticks(ticks: u64, source_hz: u64, target_hz: u64) -> u64 {
    debug_assert!(source_hz != 0);
    u64::try_from(u128::from(ticks) * u128::from(target_hz) / u128::from(source_hz))
        .unwrap_or(u64::MAX)
}
#[cfg(test)]
mod tests;
