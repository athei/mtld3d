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
        .small_chunks
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
    let chunk_ptrs: Vec<u64> = arena
        .small_chunks
        .iter()
        .map(|c| c.as_ptr() as u64)
        .collect();

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
