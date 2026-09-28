//! Unit tests for the per-frame bump arena.
//!
//! The arena hands raw pointers to another thread, so these pin the invariants
//! that make that sound: an earlier pointer stays valid and readable after later
//! allocations, payload allocations are 16-byte aligned, command storage is
//! eight-byte aligned, and an oversized request gets its own chunk without
//! displacing the hot cursor. The `clear` cases pin
//! high-water retention, the reason steady-state frames never call the allocator.

use super::*;

const TEST_CHUNK: usize = 256;

#[test]
fn alloc_returns_stable_pointer_across_subsequent_allocs() {
    let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
    let payloads: Vec<Vec<u8>> = (0u8..5)
        .map(|i| (0u8..32).map(move |b| i * 32 + b).collect())
        .collect();
    let ptrs: Vec<u64> = payloads.iter().map(|p| arena.alloc(p)).collect();

    for _ in 0..64 {
        let filler = [0xABu8; 48];
        arena.alloc(&filler);
    }

    for (ptr, expected) in ptrs.iter().zip(payloads.iter()) {
        // SAFETY: `*ptr` was just returned by `arena.alloc(expected)` and
        // covers `expected.len()` bytes within the arena slab.
        let slice = unsafe { std::slice::from_raw_parts(*ptr as *const u8, expected.len()) };
        assert_eq!(slice, expected.as_slice());
    }
}

#[test]
fn oversized_request_gets_own_chunk_and_preserves_hot_chunk() {
    let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
    let small_a = arena.alloc(&[1u8; 32]);
    assert_eq!(arena.chunk_count(), 1);

    let huge = vec![0xCDu8; TEST_CHUNK * 3];
    let _big = arena.alloc(&huge);
    assert_eq!(arena.chunk_count(), 2);

    let small_b = arena.alloc(&[2u8; 32]);
    assert_eq!(arena.chunk_count(), 2, "oversized must not reset hot chunk");
    assert_eq!(small_b, small_a + 32, "next small alloc follows small_a");
}

#[test]
fn clear_retains_high_water() {
    let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
    for _ in 0..20 {
        arena.alloc(&[0u8; 64]);
    }
    let peak = arena.small_chunk_count();
    assert!(peak >= 2);

    arena.clear();
    // High-water mark of small chunks survives clear; bytes_used
    // resets to 0 because the cursor is back at the start of chunk 0.
    assert_eq!(arena.small_chunk_count(), peak);
    assert_eq!(arena.bytes_used(), 0);

    // First allocation after clear lands at the start of chunk 0,
    // not in a new chunk past the peak.
    let post_clear = arena.alloc(&[0u8; 16]);
    let chunk0_start = arena
        .chunks
        .first()
        .map(|c| c.as_ptr() as u64)
        .expect("chunk 0 retained");
    assert_eq!(post_clear, chunk0_start);
    assert_eq!(arena.small_chunk_count(), peak);
}

/// After clear, the cursor walks forward through retained chunks.
///
/// It does so before any new allocation hits the heap. Validates the
/// high-water promise: steady-state frames touch the allocator zero
/// times.
#[test]
fn reserve_walks_existing_chunks_after_clear() {
    let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
    // Fill enough to span 3 chunks: TEST_CHUNK / 64 = 4 slots per
    // chunk; 9 allocs fill 2 chunks and bleed into a third.
    for _ in 0..9 {
        arena.alloc(&[0u8; 64]);
    }
    let peak = arena.small_chunk_count();
    assert!(
        peak >= 3,
        "test setup expects at least 3 chunks, got {peak}"
    );
    let chunk_ptrs: Vec<u64> = arena.chunks.iter().map(|c| c.as_ptr() as u64).collect();

    arena.clear();
    assert_eq!(
        arena.small_chunk_count(),
        peak,
        "clear retains the chunk vec",
    );

    // Fill the same shape again; each chunk's start address should
    // match the pre-clear pointers — no new chunk pushed.
    for i in 0..9 {
        let p = arena.alloc(&[0u8; 64]);
        let chunk_idx = i / 4;
        let slot_in_chunk = (i % 4) as u64 * 64;
        assert_eq!(
            p,
            chunk_ptrs[chunk_idx] + slot_in_chunk,
            "alloc {i} should land in the same chunk slot as pre-clear",
        );
    }
    assert_eq!(
        arena.small_chunk_count(),
        peak,
        "no new chunk allocated when walking retained ones",
    );
}

#[test]
fn alignment_padding_is_16_bytes() {
    let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
    let a = arena.alloc(&[0u8; 1]);
    let b = arena.alloc(&[0u8; 1]);
    assert_eq!(b - a, ALIGN as u64);
}

#[test]
fn empty_arena_reports_zero() {
    let arena = ScratchArena::new();
    assert_eq!(arena.chunk_count(), 0);
    assert_eq!(arena.capacity_bytes(), 0);
    assert_eq!(arena.bytes_used(), 0);
}

#[test]
fn alloc_uninit_slice_round_trips_when_written() {
    const COUNT: usize = 5;
    let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
    let ptr: *mut [f32; 4] = arena.alloc_uninit_slice::<[f32; 4]>(COUNT);
    // SAFETY: `alloc_uninit_slice` reserved `COUNT` consecutive
    // `[f32; 4]` slots starting at `ptr`; viewing them as a `&mut`
    // slice of `MaybeUninit` lets every write be a safe assignment.
    let slots: &mut [core::mem::MaybeUninit<[f32; 4]>] = unsafe {
        core::slice::from_raw_parts_mut(ptr.cast::<core::mem::MaybeUninit<[f32; 4]>>(), COUNT)
    };
    let payload: [[f32; 4]; COUNT] = [
        [0.0, 0.5, 1.0, -1.0],
        [1.0, 1.5, 1.0, -1.0],
        [2.0, 2.5, 1.0, -1.0],
        [3.0, 3.5, 1.0, -1.0],
        [4.0, 4.5, 1.0, -1.0],
    ];
    for (slot, row) in slots.iter_mut().zip(payload.iter()) {
        *slot = core::mem::MaybeUninit::new(*row);
    }
    // SAFETY: every slot was just initialised; reinterpret as the
    // concrete `[f32; 4]` slice for read-back.
    let read: &[[f32; 4]] = unsafe { core::slice::from_raw_parts(ptr.cast_const(), COUNT) };
    for (row, expected) in read.iter().zip(payload.iter()) {
        for lane in 0..4 {
            assert_eq!(row[lane].to_bits(), expected[lane].to_bits());
        }
    }
}

#[test]
fn alloc_uninit_slice_is_16_byte_aligned() {
    let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
    arena.alloc(&[0u8; 1]);
    let ptr: *mut [f32; 4] = arena.alloc_uninit_slice::<[f32; 4]>(3);
    assert_eq!(ptr.addr() % ALIGN, 0);
}

#[test]
fn bytes_used_tracks_cursor() {
    let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
    arena.alloc(&[0u8; 20]);
    assert_eq!(arena.bytes_used(), ALIGN as u64 * 2);
    arena.alloc(&[0u8; 16]);
    assert_eq!(arena.bytes_used(), ALIGN as u64 * 3);
}

#[test]
fn command_regions_ignore_interleaved_payload_allocations_and_reuse_capacity() {
    use mtld3d_shared::command_header::CommandHeader;

    let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
    let first = arena
        .write_command(1, 0, 1, |payload| {
            payload[0] = 3;
            Ok(1)
        })
        .unwrap();
    let payload = arena.alloc(&[9; 16]);
    let second = arena
        .write_command(2, 0, 4, |payload| {
            payload.copy_from_slice(&7u32.to_le_bytes());
            Ok(4)
        })
        .unwrap();
    assert_eq!(second.address, first.address + 16);
    assert_eq!(second.region_address, first.region_address);
    assert_eq!((first.record_bytes, second.record_bytes), (9, 12));
    assert_eq!(payload % 16, 0);
    assert_eq!(arena.chunk_count(), 2);
    let capacity = arena.capacity_bytes();
    // SAFETY: committed headers are aligned and retained by the arena.
    let header = unsafe { &*(first.address as *const CommandHeader) };
    assert_eq!(header.record_bytes, 9);
    arena.clear();
    let reused = arena
        .write_command(3, 0, 1, |payload| {
            payload[0] = 5;
            Ok(1)
        })
        .unwrap();
    assert_eq!(reused.address, first.address);
    arena.alloc(&[2; 16]);
    assert_eq!(arena.capacity_bytes(), capacity);
}

#[test]
fn flat_command_commit_preserves_previous_region_on_failure_and_rollover() {
    use mtld3d_shared::{command_header::CommandHeader, encoder_wire::WireError};

    let mut arena = ScratchArena::with_chunk_size(64);
    let first = arena
        .write_command(7, 9, 3, |payload| {
            payload.copy_from_slice(&[1, 2, 3]);
            Ok(3)
        })
        .unwrap();
    assert_eq!(first.address % 16, 0);
    assert_eq!(first.record_bytes, 11);
    let used = arena.bytes_used();
    assert!(matches!(
        arena.write_command(8, 0, 16, |_| Err(WireError::InvalidValue)),
        Err(WireError::InvalidValue)
    ));
    assert_eq!(arena.bytes_used(), used);
    // SAFETY: the successful command is aligned and retained by this arena.
    let header = unsafe { &*(first.address as *const CommandHeader) };
    assert_eq!(
        (header.opcode, header.operand, header.record_bytes),
        (7, 9, 11)
    );
    let second = arena
        .write_command(8, 0, 40, |payload| {
            payload.fill(0x5a);
            Ok(40)
        })
        .unwrap();
    assert_eq!(second.address, first.address + 16);
    assert_eq!(second.region_bytes, 64);
    let third = arena
        .write_command(9, 0, 1, |payload| {
            payload[0] = 6;
            Ok(1)
        })
        .unwrap();
    assert_ne!(third.region_address, first.region_address);
    assert_eq!(third.address % 16, 0);
    assert_eq!(third.region_bytes, 16);
    // SAFETY: the entire first region stays alive after rollover.
    let bytes = unsafe { core::slice::from_raw_parts(first.address as *const u8, 16) };
    assert_eq!(&bytes[8..11], &[1, 2, 3]);
    assert!(bytes[11..].iter().all(|&byte| byte == 0));
}

#[test]
fn reused_command_padding_is_zero_for_every_alignment_residue() {
    use mtld3d_shared::command_header::{COMMAND_ALIGNMENT, COMMAND_HEADER_BYTES, CommandHeader};

    let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
    for payload_bytes in 0..COMMAND_ALIGNMENT {
        arena.clear();
        let dirty = arena
            .write_command(1, 0, TEST_CHUNK - COMMAND_HEADER_BYTES, |payload| {
                payload.fill(0xff);
                Ok(payload.len())
            })
            .unwrap();
        arena.clear();
        let first = arena
            .write_command(2, 3, payload_bytes, |payload| {
                payload.fill(0xa5);
                Ok(payload.len())
            })
            .unwrap();
        assert_eq!(first.address, dirty.address);
        let used = COMMAND_HEADER_BYTES + payload_bytes;
        let aligned = (used + COMMAND_ALIGNMENT - 1) & !(COMMAND_ALIGNMENT - 1);
        // SAFETY: this retained chunk was fully initialized above. The committed
        // record occupies its prefix, and the next sixteen bytes retain the sentinel.
        let bytes =
            unsafe { core::slice::from_raw_parts(first.address as *const u8, aligned + 16) };
        assert!(
            bytes[COMMAND_HEADER_BYTES..used]
                .iter()
                .all(|&byte| byte == 0xa5)
        );
        assert!(bytes[used..aligned].iter().all(|&byte| byte == 0));
        assert!(bytes[aligned..].iter().all(|&byte| byte == 0xff));
        // SAFETY: the command begins at an aligned address and has an initialized header.
        let header = unsafe { &*(first.address as *const CommandHeader) };
        assert_eq!((header.opcode, header.operand), (2, 3));
        assert_eq!(header.record_bytes as usize, used);
        let next = arena
            .write_command(4, 0, 1, |payload| {
                payload[0] = 0x5a;
                Ok(1)
            })
            .unwrap();
        assert_eq!(next.address, first.address + aligned as u64);
        assert_eq!(next.address % COMMAND_ALIGNMENT as u64, 0);
    }
}

#[test]
fn command_slot_claims_exact_room_and_commits_only_when_written() {
    use mtld3d_shared::command_header::{COMMAND_HEADER_BYTES, CommandHeader};

    use crate::encoder_records::IdRecord;

    let mut arena = ScratchArena::with_chunk_size(64);
    assert!(arena.command_slot::<IdRecord>().is_none());
    assert!(!arena.has_command_room(16));
    arena.open_command_region(16).unwrap();
    let region = arena.command_base as u64;
    assert_eq!(arena.command_regions.len(), 1);
    assert_eq!(region % 16, 0);
    assert_eq!(arena.open_command_bytes(), 0);
    assert!(arena.command_slot::<IdRecord>().is_some());
    assert_eq!((arena.open_command_bytes(), arena.bytes_used()), (0, 0));
    for id in 0..4_u64 {
        arena
            .command_slot::<IdRecord>()
            .unwrap()
            .write(3, 9, IdRecord { id: 0x100 + id });
    }
    assert_eq!((arena.open_command_bytes(), arena.bytes_used()), (64, 64));
    assert!(arena.command_slot::<IdRecord>().is_none());
    for index in 0..4_u64 {
        let address = region + index * 16;
        // SAFETY: the arena retains its open region, whose four written commands are aligned.
        let header = unsafe { &*(address as *const CommandHeader) };
        assert_eq!(
            (header.opcode, header.operand, header.record_bytes),
            (3, 9, 16)
        );
        // SAFETY: the payload follows its header inside the same retained command.
        let id = unsafe { *((address + COMMAND_HEADER_BYTES as u64) as *const u64) };
        assert_eq!(id, 0x100 + index);
    }
    let next = arena
        .write_command(4, 0, 8, |payload| {
            payload.fill(0x5a);
            Ok(8)
        })
        .unwrap();
    assert_ne!(next.region_address, region);
    assert_eq!((next.region_bytes, arena.bytes_used()), (16, 80));
    arena.clear();
    assert!(arena.command_slot::<IdRecord>().is_none());
    assert_eq!(arena.bytes_used(), 0);
}

#[test]
fn failed_rollover_keeps_committed_region_and_reuses_next_reservation() {
    use mtld3d_shared::encoder_wire::WireError;

    let mut arena = ScratchArena::with_chunk_size(64);
    arena
        .push_fixed_record(1, 0, 56, |bytes| {
            bytes.fill(1);
            Ok(())
        })
        .unwrap();
    arena.publish_command_region().unwrap();
    let first = arena.command_regions[0].address;
    assert_eq!(arena.command_regions[0].used_bytes, 64);
    assert_eq!(
        arena.push_fixed_record(2, 0, 16, |bytes| {
            bytes.fill(2);
            Err(WireError::InvalidValue)
        }),
        Err(WireError::InvalidValue)
    );
    // The rollover names its region before writing, so the failed record leaves it empty.
    arena.publish_command_region().unwrap();
    assert_eq!(arena.command_regions.len(), 2);
    assert_eq!(arena.command_regions[0].address, first);
    assert_eq!(arena.command_regions[0].used_bytes, 64);
    assert_eq!(arena.command_regions[1].used_bytes, 0);
    assert_eq!(arena.bytes_used(), 64);
    arena
        .push_fixed_record(3, 0, 16, |bytes| {
            bytes.fill(3);
            Ok(())
        })
        .unwrap();
    arena.publish_command_region().unwrap();
    assert_eq!(arena.command_regions.len(), 2);
    assert_eq!(arena.command_regions[1].used_bytes, 24);
    assert_ne!(arena.command_regions[1].address, first);
    assert_eq!(arena.chunk_count(), 2);
}

#[test]
fn oversized_payload_and_two_cursors_reuse_one_pool_after_clear() {
    let mut arena = ScratchArena::with_chunk_size(64);
    arena.push_fixed_record(1, 0, 0, |_| Ok(())).unwrap();
    let first = arena.command_regions[0].address;
    let oversized = arena.alloc(&[7; 128]);
    let ordinary = arena.alloc(&[8; 16]);
    assert_ne!(oversized, first);
    assert_ne!(ordinary, first);
    arena.push_fixed_record(2, 0, 0, |_| Ok(())).unwrap();
    arena.publish_command_region().unwrap();
    assert_eq!(arena.command_regions.len(), 1);
    assert_eq!(arena.command_regions[0].used_bytes, 16);
    assert_eq!(arena.oversized_chunk_count(), 1);
    let retained = arena.small_chunk_count();
    arena.clear();
    assert_eq!(arena.oversized_chunk_count(), 0);
    arena.push_fixed_record(3, 0, 0, |_| Ok(())).unwrap();
    arena.alloc(&[9; 16]);
    assert_eq!(arena.command_regions[0].address, first);
    assert_eq!(arena.small_chunk_count(), retained);
}

#[test]
fn write_command_names_every_region_it_opens() {
    let mut arena = ScratchArena::with_chunk_size(32);
    for opcode in 1..=5_u16 {
        arena
            .write_command(opcode, 0, 8, |payload| {
                payload.fill(0x11);
                Ok(8)
            })
            .unwrap();
    }
    let regions: Vec<_> = arena.command_ranges().collect();
    assert_eq!(
        regions.iter().map(|&(_, used)| used).collect::<Vec<_>>(),
        [32, 32, 16]
    );
    assert_eq!(arena.chunk_count(), 3);
    assert_eq!(arena.bytes_used(), 80);
}
