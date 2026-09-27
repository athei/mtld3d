//! Architecture-independent API telemetry for native frame replay.
//!
//! Every duration is serialized as elapsed nanoseconds, then converted once
//! into the reader's local counter units. Counts and byte volumes stay exact.
//! No timestamp or Rust representation crosses the runtime boundary.

use mtld3d_shared::encoder_wire::{WireError, WireReader, WireWriter};
#[cfg(perf_tracking)]
use mtld3d_shared::tsc::ns_to_cycles;

use super::FramePerfPayload;
#[cfg(perf_tracking)]
use super::{FrameCounters, FrameTiming, compilation::cycles_to_ns};
use crate::encoder_value::WireValue;

impl WireValue for FramePerfPayload {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u8(u8::from(cfg!(perf_tracking)))?;
        #[cfg(perf_tracking)]
        {
            self.counters.write_wire(writer)?;
            self.timing.write_wire(writer)?;
        }
        Ok(())
    }

    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        if reader.u8()? != u8::from(cfg!(perf_tracking)) {
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
        impl WireValue for $value {
            fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
                // Exhaustive destructuring makes a newly added counter require a codec entry.
                let Self { $($value_field,)* $($duration_field,)* } = self;
                $($value_field.write_wire(writer)?;)*
                $($duration_field.write_nanos(writer)?;)*
                Ok(())
            }

            fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
                Ok(Self {
                    $($value_field: WireValue::read_wire(reader)?,)*
                    $($duration_field: WireDuration::read_nanos(reader)?,)*
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
    fn write_nanos(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError>;
    fn read_nanos(reader: &mut WireReader<'_>) -> Result<Self, WireError>;
}

#[cfg(perf_tracking)]
impl WireDuration for u64 {
    fn write_nanos(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        // Zero duration does not require initializing the sender's timer calibration.
        writer.u64(if *self == 0 { 0 } else { cycles_to_ns(*self) })
    }

    fn read_nanos(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        reader.u64().map(ns_to_cycles)
    }
}

#[cfg(perf_tracking)]
impl<const N: usize> WireDuration for [u64; N] {
    fn write_nanos(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        for duration in self {
            duration.write_nanos(writer)?;
        }
        Ok(())
    }

    fn read_nanos(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        let mut values = [0; N];
        for value in &mut values {
            *value = u64::read_nanos(reader)?;
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests;
