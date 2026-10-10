//! Unit tests for the upload-snapshot arena.
//!
//! Offsets land on the 16-byte grid and the bytes `fill` writes are the ones
//! the read sees. A chunk that cannot fit a request moves to the queue, the
//! front chunk starts over only after every read of it has dropped, the arena
//! stops at its chunk cap rather than waiting, and a declined fill, an empty
//! request and an oversized one claim nothing.

use std::sync::Arc;

use super::*;

/// Write `len` bytes of `value` and return the snapshot.
fn snapshot(arena: &mut UploadSnapshots, len: usize, value: u8) -> UploadSnapshot {
    arena
        .write(len, |destination| {
            destination.fill(value);
            true
        })
        .expect("the arena has room")
}

/// The `len` bytes at `offset` of the chunk `read` names.
fn bytes_at(read: &PageBoxRead, offset: u32, len: usize) -> Vec<u8> {
    let start = usize::try_from(offset).expect("offset fits usize");
    read.backing().as_slice()[start..start + len].to_vec()
}

#[test]
fn snapshots_land_on_the_alignment_grid_with_their_bytes() {
    let mut arena = UploadSnapshots::default();
    let first = snapshot(&mut arena, 5, 0xA5);
    let second = snapshot(&mut arena, 3, 0x5A);
    assert_eq!(first.offset, 0);
    assert_eq!(second.offset, 16);
    assert!(Arc::ptr_eq(first.read.backing(), second.read.backing()));
    assert_eq!(bytes_at(&first.read, first.offset, 5), vec![0xA5; 5]);
    assert_eq!(bytes_at(&second.read, second.offset, 3), vec![0x5A; 3]);
    assert!(first.read.backing().has_readers());
}

#[test]
fn a_full_chunk_moves_on_to_a_new_one_while_it_is_read() {
    let mut arena = UploadSnapshots::default();
    let held: Vec<UploadSnapshot> = (0..4)
        .map(|i| snapshot(&mut arena, SNAPSHOT_MAX_BYTES, i))
        .collect();
    let offsets: Vec<u32> = held.iter().map(|s| s.offset).collect();
    assert_eq!(offsets, [0, 65_536, 131_072, 196_608]);
    let next = snapshot(&mut arena, 16, 9);
    assert_eq!(next.offset, 0);
    assert!(!Arc::ptr_eq(held[0].read.backing(), next.read.backing()));
    assert_eq!(arena.chunk_count(), 2);
    assert_eq!(arena.held_bytes(), 2 * CHUNK_BYTES as u64);
}

#[test]
fn the_front_chunk_starts_over_only_once_its_reads_drop() {
    let mut arena = UploadSnapshots::default();
    let first: Vec<UploadSnapshot> = (0..4)
        .map(|i| snapshot(&mut arena, SNAPSHOT_MAX_BYTES, i))
        .collect();
    let first_chunk = Arc::clone(first[0].read.backing());
    // Fill a second chunk while the first is still read.
    let second: Vec<UploadSnapshot> = (0..4)
        .map(|i| snapshot(&mut arena, SNAPSHOT_MAX_BYTES, i))
        .collect();
    assert!(!Arc::ptr_eq(&first_chunk, second[0].read.backing()));
    // One read of the first chunk left keeps it from starting over.
    let mut first = first;
    let last_read = first.pop().expect("four snapshots");
    drop(first);
    let third = snapshot(&mut arena, 16, 1);
    assert!(!Arc::ptr_eq(&first_chunk, third.read.backing()));
    assert_eq!(arena.chunk_count(), 3);
    drop(last_read);
    // Fill the third chunk so the arena moves on again: the first is free now.
    let filler: Vec<UploadSnapshot> = (0..3)
        .map(|i| snapshot(&mut arena, SNAPSHOT_MAX_BYTES, i))
        .collect();
    let reused = snapshot(&mut arena, SNAPSHOT_MAX_BYTES, 7);
    assert!(Arc::ptr_eq(&first_chunk, reused.read.backing()));
    assert_eq!(reused.offset, 0);
    assert_eq!(arena.chunk_count(), 3);
    drop((second, filler));
}

#[test]
fn the_chunk_cap_falls_back_instead_of_waiting() {
    let mut arena = UploadSnapshots::default();
    let held: Vec<UploadSnapshot> = (0..MAX_CHUNKS * 4)
        .map(|_| snapshot(&mut arena, SNAPSHOT_MAX_BYTES, 0))
        .collect();
    assert_eq!(arena.chunk_count(), MAX_CHUNKS);
    assert!(arena.write(16, |_| true).is_none());
    assert_eq!(arena.chunk_count(), MAX_CHUNKS);
    drop(held);
    assert_eq!(snapshot(&mut arena, 16, 0).offset, 0);
    assert_eq!(arena.chunk_count(), MAX_CHUNKS);
}

#[test]
fn a_declined_fill_an_empty_and_an_oversized_request_claim_nothing() {
    let mut arena = UploadSnapshots::default();
    assert!(arena.write(32, |_| false).is_none());
    assert!(arena.write(0, |_| true).is_none());
    assert!(arena.write(SNAPSHOT_MAX_BYTES + 1, |_| true).is_none());
    assert_eq!(snapshot(&mut arena, 8, 1).offset, 0);
}
