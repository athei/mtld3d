//! Architecture-independent API telemetry for native frame replay.
//!
//! PERF durations are elapsed source-counter ticks, tagged separately from nanoseconds.
//! The device calibration mailbox supplies their frequency asynchronously; aggregation
//! converts them only after both clock domains are ready. Counts and byte volumes stay exact.
//! No timestamp or Rust representation crosses the runtime boundary.

use mtld3d_shared::encoder_wire::{WireError, WireReader, WireWriter};

use super::FramePerfPayload;
#[cfg(perf_tracking)]
use super::{FrameCounters, FrameTiming};
use crate::encoder_value::WireValue;

#[repr(u8)]
enum DurationEncoding {
    Disabled = 0,
    SourceElapsedTicks = 2,
}

const DURATION_ENCODING: DurationEncoding = if cfg!(perf_tracking) {
    DurationEncoding::SourceElapsedTicks
} else {
    DurationEncoding::Disabled
};

impl WireValue for FramePerfPayload {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u8(DURATION_ENCODING as u8)?;
        #[cfg(perf_tracking)]
        {
            self.counters.write_wire(writer)?;
            self.timing.write_wire(writer)?;
        }
        Ok(())
    }

    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        if reader.u8()? != DURATION_ENCODING as u8 {
            // Matching PE/Unix builds agree on whether telemetry is present.
            return Err(WireError::InvalidValue);
        }
        #[cfg(perf_tracking)]
        {
            Ok(Self {
                counters: FrameCounters::read_wire(reader)?,
                timing: FrameTiming::read_wire(reader)?,
            })
        }
        #[cfg(not(perf_tracking))]
        {
            Ok(Self)
        }
    }
}

#[cfg(perf_tracking)]
macro_rules! telemetry_codec {
    ($value:ident { values: [$($value_field:ident),* $(,)?],
                       durations: [$($duration_field:ident),* $(,)?] }) => {
        impl $value {
            pub(super) fn rescale_durations(&mut self, source_hz: u64, target_hz: u64) {
                $(self.$duration_field.rescale(source_hz, target_hz);)*
            }
        }
        impl WireValue for $value {
            fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
                // Exhaustive destructuring makes a newly added counter require a codec entry.
                let Self { $($value_field,)* $($duration_field,)* } = self;
                $($value_field.write_wire(writer)?;)*
                $($duration_field.write_ticks(writer)?;)*
                Ok(())
            }

            fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
                Ok(Self {
                    $($value_field: WireValue::read_wire(reader)?,)*
                    $($duration_field: WireDuration::read_ticks(reader)?,)*
                })
            }
        }
    };
}

#[cfg(perf_tracking)]
telemetry_codec!(FrameCounters {
    values: [
        reset_epoch,
        reset_epoch_saturated,
        inverse_view,
        inverse_view_saturated,
        api_call_counts_by_category,
        vb_rename,
        ib_rename,
        vbib_rename_bytes,
        vbib_pool_hits,
        vbib_pool_misses,
        vb_discards,
        ib_discards,
        vbib_preserve_cpu,
        vbib_write_in_place_contended,
        retention_cap_drain,
        retention_cap_submit,
        texture_renames,
        texture_discards,
        texture_preserve_cpu,
        texture_write_in_place_contended,
        texture_add_dirty_calls,
        texture_add_dirty_partial,
        texture_add_dirty_area_bp,
        device_sub_calls,
        bind_sub_calls,
        surface_sub_calls,
        keys_gate_calls,
        keys_gate_skips,
    ],
    durations: [
        api_cycles_by_category,
        query_wait_cycles,
        device_sub_cycles,
        bind_sub_cycles,
        surface_sub_cycles,
        draw_snapshot_cycles,
        draw_snapshot_stages_cycles,
        draw_snapshot_c_ff_cycles,
        draw_snapshot_c_pr_cycles,
        draw_snapshot_keys_cycles,
        draw_snapshot_bumps_cycles,
        draw_push_op_cycles,
    ]
});

#[cfg(perf_tracking)]
telemetry_codec!(FrameTiming {
    values: [op_vec_capacity_bytes, op_vec_realloc_bytes],
    durations: [present_block_cycles, frame_total_cycles]
});

#[cfg(perf_tracking)]
trait WireDuration: Sized {
    fn rescale(&mut self, source_hz: u64, target_hz: u64);
    fn write_ticks(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError>;
    fn read_ticks(reader: &mut WireReader<'_>) -> Result<Self, WireError>;
}

#[cfg(perf_tracking)]
impl WireDuration for u64 {
    fn rescale(&mut self, source_hz: u64, target_hz: u64) {
        *self = scale_ticks(*self, source_hz, target_hz);
    }
    fn write_ticks(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u64(*self)
    }

    fn read_ticks(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        reader.u64()
    }
}

#[cfg(perf_tracking)]
impl<const N: usize> WireDuration for [u64; N] {
    fn rescale(&mut self, source_hz: u64, target_hz: u64) {
        for value in self {
            value.rescale(source_hz, target_hz);
        }
    }
    fn write_ticks(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        for duration in self {
            duration.write_ticks(writer)?;
        }
        Ok(())
    }

    fn read_ticks(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        let mut values = [0; N];
        for value in &mut values {
            *value = u64::read_ticks(reader)?;
        }
        Ok(values)
    }
}

#[cfg(perf_tracking)]
pub(super) fn scale_ticks(ticks: u64, source_hz: u64, target_hz: u64) -> u64 {
    debug_assert!(source_hz != 0);
    u64::try_from(u128::from(ticks) * u128::from(target_hz) / u128::from(source_hz))
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests;
