use mtld3d_shared::encoder_wire::FrameSlab;
#[cfg(perf_tracking)]
use strum::EnumCount;

#[cfg(perf_tracking)]
use super::super::{ApiCategory, BindSubCategory, DeviceSubCategory, KeysGate, SurfaceSubCategory};
use super::*;

fn encoded(payload: &FramePerfPayload) -> FrameSlab {
    let mut slab = FrameSlab::new();
    slab.push_record(1, |writer| payload.write_wire(writer))
        .unwrap();
    slab
}

#[test]
fn tracking_marker_must_match_build() {
    let marker = u8::from(!cfg!(perf_tracking));
    assert!(matches!(
        FramePerfPayload::read_wire(&mut WireReader::new(&[marker])),
        Err(WireError::InvalidValue)
    ));
    assert!(matches!(
        FramePerfPayload::read_wire(&mut WireReader::new(&[3])),
        Err(WireError::InvalidValue)
    ));
}

#[test]
fn default_payload_and_all_truncations() {
    let slab = encoded(&FramePerfPayload::new());
    let mut reader = WireReader::new(slab.as_bytes());
    let mut record = reader.next_record().unwrap().unwrap();
    let bytes = record
        .payload
        .bytes(u32::try_from(record.payload.remaining_len()).unwrap())
        .unwrap();
    for length in 0..bytes.len() {
        assert!(matches!(
            FramePerfPayload::read_wire(&mut WireReader::new(&bytes[..length])),
            Err(WireError::Truncated)
        ));
    }
    let mut complete = WireReader::new(bytes);
    FramePerfPayload::read_wire(&mut complete).unwrap();
    assert!(complete.is_empty());
    assert!(reader.is_empty());
    #[cfg(not(perf_tracking))]
    assert_eq!(bytes, &[0]);
}

#[cfg(perf_tracking)]
#[test]
fn wire_durations_preserve_source_ticks_without_calibration() {
    let frequency = 123_456_789;
    let payload = FramePerfPayload {
        counters: FrameCounters {
            reset_epoch: 1,
            reset_epoch_saturated: true,
            inverse_view: [3; 3],
            inverse_view_saturated: true,
            api_cycles_by_category: [5 * frequency; ApiCategory::COUNT],
            api_call_counts_by_category: [6; ApiCategory::COUNT],
            vb_rename: 7,
            ib_rename: 8,
            vbib_rename_bytes: 9,
            vbib_pool_hits: 10,
            vbib_pool_misses: 11,
            vb_discards: 12,
            ib_discards: 13,
            vbib_preserve_cpu: 14,
            vbib_write_in_place_contended: 15,
            retention_cap_drain: 16,
            retention_cap_submit: 17,
            texture_renames: 18,
            texture_discards: 19,
            texture_preserve_cpu: 20,
            texture_write_in_place_contended: 21,
            texture_add_dirty_calls: 22,
            texture_add_dirty_partial: 23,
            texture_add_dirty_area_bp: 24,
            query_wait_cycles: 25 * frequency,
            device_sub_cycles: [26 * frequency; DeviceSubCategory::COUNT],
            device_sub_calls: [27; DeviceSubCategory::COUNT],
            bind_sub_cycles: [28 * frequency; BindSubCategory::COUNT],
            bind_sub_calls: [29; BindSubCategory::COUNT],
            surface_sub_cycles: [30 * frequency; SurfaceSubCategory::COUNT],
            surface_sub_calls: [31; SurfaceSubCategory::COUNT],
            keys_gate_calls: [32; KeysGate::COUNT],
            keys_gate_skips: [33; KeysGate::COUNT],
            draw_snapshot_cycles: 34 * frequency,
            draw_snapshot_stages_cycles: 35 * frequency,
            draw_snapshot_c_ff_cycles: 36 * frequency,
            draw_snapshot_c_pr_cycles: 37 * frequency,
            draw_snapshot_keys_cycles: 38 * frequency,
            draw_snapshot_bumps_cycles: 39 * frequency,
            draw_push_op_cycles: 40 * frequency,
        },
        timing: FrameTiming {
            present_block_cycles: 41 * frequency,
            frame_total_cycles: 42 * frequency,
            op_vec_capacity_bytes: 43,
            op_vec_realloc_bytes: 44,
        },
    };
    let mut expected_bytes = vec![2_u8];
    expected_bytes.extend_from_slice(&(1_u64).to_le_bytes());
    expected_bytes.extend_from_slice(&(1_u8).to_le_bytes());
    for _ in 0..3 {
        expected_bytes.extend_from_slice(&(3_u64).to_le_bytes());
    }
    expected_bytes.extend_from_slice(&(1_u8).to_le_bytes());
    for _ in 0..ApiCategory::COUNT {
        expected_bytes.extend_from_slice(&(6_u32).to_le_bytes());
    }
    expected_bytes.extend_from_slice(&(7_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(8_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(9_u64).to_le_bytes());
    expected_bytes.extend_from_slice(&(10_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(11_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(12_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(13_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(14_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(15_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(16_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(17_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(18_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(19_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(20_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(21_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(22_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(23_u32).to_le_bytes());
    expected_bytes.extend_from_slice(&(24_u32).to_le_bytes());
    for _ in 0..DeviceSubCategory::COUNT {
        expected_bytes.extend_from_slice(&(27_u32).to_le_bytes());
    }
    for _ in 0..BindSubCategory::COUNT {
        expected_bytes.extend_from_slice(&(29_u32).to_le_bytes());
    }
    for _ in 0..SurfaceSubCategory::COUNT {
        expected_bytes.extend_from_slice(&(31_u32).to_le_bytes());
    }
    for _ in 0..KeysGate::COUNT {
        expected_bytes.extend_from_slice(&(32_u32).to_le_bytes());
    }
    for _ in 0..KeysGate::COUNT {
        expected_bytes.extend_from_slice(&(33_u32).to_le_bytes());
    }
    for _ in 0..ApiCategory::COUNT {
        expected_bytes.extend_from_slice(&(5_u64 * frequency).to_le_bytes());
    }
    expected_bytes.extend_from_slice(&(25_u64 * frequency).to_le_bytes());
    for _ in 0..DeviceSubCategory::COUNT {
        expected_bytes.extend_from_slice(&(26_u64 * frequency).to_le_bytes());
    }
    for _ in 0..BindSubCategory::COUNT {
        expected_bytes.extend_from_slice(&(28_u64 * frequency).to_le_bytes());
    }
    for _ in 0..SurfaceSubCategory::COUNT {
        expected_bytes.extend_from_slice(&(30_u64 * frequency).to_le_bytes());
    }
    expected_bytes.extend_from_slice(&(34_u64 * frequency).to_le_bytes());
    expected_bytes.extend_from_slice(&(35_u64 * frequency).to_le_bytes());
    expected_bytes.extend_from_slice(&(36_u64 * frequency).to_le_bytes());
    expected_bytes.extend_from_slice(&(37_u64 * frequency).to_le_bytes());
    expected_bytes.extend_from_slice(&(38_u64 * frequency).to_le_bytes());
    expected_bytes.extend_from_slice(&(39_u64 * frequency).to_le_bytes());
    expected_bytes.extend_from_slice(&(40_u64 * frequency).to_le_bytes());
    expected_bytes.extend_from_slice(&(43_u64).to_le_bytes());
    expected_bytes.extend_from_slice(&(44_u64).to_le_bytes());
    expected_bytes.extend_from_slice(&(41_u64 * frequency).to_le_bytes());
    expected_bytes.extend_from_slice(&(42_u64 * frequency).to_le_bytes());
    let slab = encoded(&payload);
    let mut reader = WireReader::new(slab.as_bytes());
    let mut record = reader.next_record().unwrap().unwrap();
    assert_eq!(
        record
            .payload
            .bytes(u32::try_from(expected_bytes.len()).unwrap())
            .unwrap(),
        expected_bytes
    );
    assert!(record.payload.is_empty());
    let mut wire = WireReader::new(&expected_bytes);
    let decoded = FramePerfPayload::read_wire(&mut wire).unwrap();
    assert!(wire.is_empty());
    assert_eq!(decoded.counters.reset_epoch, 1);
    assert!(decoded.counters.reset_epoch_saturated);
    assert_eq!(decoded.counters.inverse_view, [3; 3]);
    assert!(decoded.counters.inverse_view_saturated);
    assert_eq!(
        decoded.counters.api_cycles_by_category,
        [5 * frequency; ApiCategory::COUNT]
    );
    assert_eq!(
        decoded.counters.api_call_counts_by_category,
        [6; ApiCategory::COUNT]
    );
    assert_eq!(decoded.counters.vb_rename, 7);
    assert_eq!(decoded.counters.ib_rename, 8);
    assert_eq!(decoded.counters.vbib_rename_bytes, 9);
    assert_eq!(decoded.counters.vbib_pool_hits, 10);
    assert_eq!(decoded.counters.vbib_pool_misses, 11);
    assert_eq!(decoded.counters.vb_discards, 12);
    assert_eq!(decoded.counters.ib_discards, 13);
    assert_eq!(decoded.counters.vbib_preserve_cpu, 14);
    assert_eq!(decoded.counters.vbib_write_in_place_contended, 15);
    assert_eq!(decoded.counters.retention_cap_drain, 16);
    assert_eq!(decoded.counters.retention_cap_submit, 17);
    assert_eq!(decoded.counters.texture_renames, 18);
    assert_eq!(decoded.counters.texture_discards, 19);
    assert_eq!(decoded.counters.texture_preserve_cpu, 20);
    assert_eq!(decoded.counters.texture_write_in_place_contended, 21);
    assert_eq!(decoded.counters.texture_add_dirty_calls, 22);
    assert_eq!(decoded.counters.texture_add_dirty_partial, 23);
    assert_eq!(decoded.counters.texture_add_dirty_area_bp, 24);
    assert_eq!(decoded.counters.query_wait_cycles, 25 * frequency);
    assert_eq!(
        decoded.counters.device_sub_cycles,
        [26 * frequency; DeviceSubCategory::COUNT]
    );
    assert_eq!(
        decoded.counters.device_sub_calls,
        [27; DeviceSubCategory::COUNT]
    );
    assert_eq!(
        decoded.counters.bind_sub_cycles,
        [28 * frequency; BindSubCategory::COUNT]
    );
    assert_eq!(
        decoded.counters.bind_sub_calls,
        [29; BindSubCategory::COUNT]
    );
    assert_eq!(
        decoded.counters.surface_sub_cycles,
        [30 * frequency; SurfaceSubCategory::COUNT]
    );
    assert_eq!(
        decoded.counters.surface_sub_calls,
        [31; SurfaceSubCategory::COUNT]
    );
    assert_eq!(decoded.counters.keys_gate_calls, [32; KeysGate::COUNT]);
    assert_eq!(decoded.counters.keys_gate_skips, [33; KeysGate::COUNT]);
    assert_eq!(decoded.counters.draw_snapshot_cycles, 34 * frequency);
    assert_eq!(decoded.counters.draw_snapshot_stages_cycles, 35 * frequency);
    assert_eq!(decoded.counters.draw_snapshot_c_ff_cycles, 36 * frequency);
    assert_eq!(decoded.counters.draw_snapshot_c_pr_cycles, 37 * frequency);
    assert_eq!(decoded.counters.draw_snapshot_keys_cycles, 38 * frequency);
    assert_eq!(decoded.counters.draw_snapshot_bumps_cycles, 39 * frequency);
    assert_eq!(decoded.counters.draw_push_op_cycles, 40 * frequency);
    assert_eq!(decoded.timing.present_block_cycles, 41 * frequency);
    assert_eq!(decoded.timing.frame_total_cycles, 42 * frequency);
    assert_eq!(decoded.timing.op_vec_capacity_bytes, 43);
    assert_eq!(decoded.timing.op_vec_realloc_bytes, 44);
}
