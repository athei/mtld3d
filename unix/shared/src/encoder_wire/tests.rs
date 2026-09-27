use std::{
    sync::atomic::{AtomicU64, Ordering},
    thread,
};

use super::{FrameSlab, LeaseCompletion, ReplayMailbox, WireError, WireReader};

#[test]
fn scalar_records_round_trip_without_payload_copy() {
    let mut slab = FrameSlab::new();
    slab.push_record(0x1234, |writer| {
        writer.u8(0xa5)?;
        writer.u16(0x5678)?;
        writer.u32(0x1234_5678)?;
        writer.u64(0x0123_4567_89ab_cdef)?;
        writer.i32(-123)?;
        writer.f32(f32::from_bits(0x7fc0_0042))?;
        writer.bytes(&[11, 22, 33])
    })
    .unwrap();
    slab.push_record(2, |_| Ok(())).unwrap();
    assert_eq!(&slab.as_bytes()[..6], &[0x34, 0x12, 26, 0, 0, 0]);
    let mut reader = WireReader::new(slab.as_bytes());
    let record = reader.next_record().unwrap().unwrap();
    assert_eq!(record.tag, 0x1234);
    let mut payload = record.payload;
    assert_eq!(payload.u8(), Ok(0xa5));
    assert_eq!(payload.u16(), Ok(0x5678));
    assert_eq!(payload.u32(), Ok(0x1234_5678));
    assert_eq!(payload.u64(), Ok(0x0123_4567_89ab_cdef));
    assert_eq!(payload.i32(), Ok(-123));
    assert_eq!(payload.f32().unwrap().to_bits(), 0x7fc0_0042);
    let tail = payload.bytes(3).unwrap();
    assert_eq!(tail, &[11, 22, 33]);
    assert_eq!(tail.as_ptr(), slab.as_bytes()[29..].as_ptr());
    assert!(payload.is_empty());
    let empty = reader.next_record().unwrap().unwrap();
    assert_eq!(empty.tag, 2);
    assert!(empty.payload.is_empty());
    assert!(reader.next_record().unwrap().is_none());
}

#[test]
fn every_truncated_record_prefix_is_rejected_without_advancing() {
    let mut slab = FrameSlab::new();
    slab.push_record(1, |writer| writer.u64(u64::MAX)).unwrap();
    for length in 1..slab.as_bytes().len() {
        let mut reader = WireReader::new(&slab.as_bytes()[..length]);
        assert!(matches!(reader.next_record(), Err(WireError::Truncated)));
        assert!(matches!(reader.next_record(), Err(WireError::Truncated)));
    }
    let malformed = [1, 0, 255, 255, 255, 255];
    let mut reader = WireReader::new(&malformed);
    assert!(matches!(reader.next_record(), Err(WireError::Truncated)));
}

#[test]
fn fields_cannot_consume_the_next_record() {
    let mut slab = FrameSlab::new();
    slab.push_record(1, |writer| writer.u8(42)).unwrap();
    slab.push_record(2, |writer| writer.u64(3)).unwrap();
    let mut reader = WireReader::new(slab.as_bytes());
    let mut first = reader.next_record().unwrap().unwrap().payload;
    assert_eq!(first.u64(), Err(WireError::Truncated));
    assert_eq!(first.bytes(u32::MAX), Err(WireError::Truncated));
    assert_eq!(first.u8(), Ok(42));
    assert_eq!(first.u8(), Err(WireError::Truncated));
    assert_eq!(reader.next_record().unwrap().unwrap().payload.u64(), Ok(3));
}

#[test]
fn failed_record_rolls_back_and_clear_reuses_allocation() {
    let mut slab = FrameSlab::new();
    slab.push_record(7, |writer| writer.u32(9)).unwrap();
    let original = slab.as_bytes().to_vec();
    let result = slab.push_record(8, |writer| {
        writer.u64(999)?;
        Err(WireError::TooLarge)
    });
    assert_eq!(result, Err(WireError::TooLarge));
    assert_eq!(slab.as_bytes(), original);
    let capacity = slab.bytes.capacity();
    let address = slab.as_bytes().as_ptr();
    slab.clear();
    slab.push_record(10, |writer| writer.u8(11)).unwrap();
    assert_eq!(slab.bytes.capacity(), capacity);
    assert_eq!(slab.as_bytes().as_ptr(), address);
}

#[test]
fn lease_acknowledgment_publishes_prior_work() {
    let completion = LeaseCompletion::new();
    let prior_work = AtomicU64::new(0);
    assert!(!completion.is_complete());
    thread::scope(|scope| {
        scope.spawn(|| {
            prior_work.store(42, Ordering::Relaxed);
            completion.publish();
        });
        while !completion.is_complete() {
            thread::yield_now();
        }
        assert_eq!(prior_work.load(Ordering::Relaxed), 42);
    });
}

#[test]
fn replay_completion_orders_and_isolates_device_progress() {
    let first = ReplayMailbox::new();
    let second = ReplayMailbox::new();
    let prior_work = AtomicU64::new(0);
    thread::scope(|scope| {
        scope.spawn(|| {
            for sequence in 1..=100 {
                prior_work.store(sequence, Ordering::Relaxed);
                first.publish(sequence);
            }
        });
        while first.completed() < 100 {
            thread::yield_now();
        }
        assert_eq!(prior_work.load(Ordering::Relaxed), 100);
        assert_eq!(second.completed(), 0);
    });
}

#[test]
fn retained_ranges_reject_escape_and_survive_record_nesting() {
    let mut slab = FrameSlab::new();
    slab.push_record(1, |writer| writer.u32(7)).unwrap();
    let ranges = [(0x1000, 0x100), (0x2000, 0x20), (u64::MAX - 2, 8)];
    // SAFETY: this test queries integer membership only and never decodes an address.
    let mut reader = unsafe { WireReader::new_trusted_with_ranges(slab.as_bytes(), &ranges) };
    assert!(reader.permits_range(0x1000, 0x100));
    assert!(reader.permits_range(0x1010, 0x10));
    assert!(!reader.permits_range(0x10ff, 2));
    assert!(!reader.permits_range(0x1100, 1));
    assert!(!reader.permits_range(0x0fff, 2));
    assert!(!reader.permits_range(0, 1));
    assert!(!reader.permits_range(u64::MAX - 1, 3));
    assert!(!reader.permits_range(u64::MAX - 2, 1));
    let record = reader.next_record().unwrap().unwrap();
    assert!(record.payload.permits_range(0x2000, 0x20));
    assert!(!record.payload.permits_range(0x2000, 0x21));
    assert!(!WireReader::new(&[]).permits_range(0x1000, 1));
}
