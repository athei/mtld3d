//! Unit tests for the on-disk shader-cache binary format.
//!
//! Hand-built files cover the round trips, the damage paths (torn tail, flipped chunk-header
//! bit, scrambled zstd frame, unknown chunk kind), duplicate keys and a header-only file. Every
//! one of those but the scrambled-frame case pins the `needs_compaction` verdict the pre-warm
//! rewrite keys off. Further tests cover header validation, `CachedKind` mapping and
//! `ff_key_hash` stability.

use super::*;

fn write_file(entries_per_chunk: &[Vec<CacheEntry>], bundle_last: bool) -> Vec<u8> {
    let mut buf = Vec::new();
    write_header(&mut buf);
    let last_idx = entries_per_chunk.len().saturating_sub(1);
    for (i, group) in entries_per_chunk.iter().enumerate() {
        let as_bundle = bundle_last && i == last_idx;
        if as_bundle {
            write_bundle(&mut buf, group);
        } else {
            for entry in group {
                write_record(&mut buf, entry);
            }
        }
    }
    buf
}

fn sample_entries() -> Vec<CacheEntry> {
    vec![
        CacheEntry {
            kind: CachedKind::Sm3Vs,
            key: 0xDEAD_BEEF_CAFE_BABE,
            msl: "vertex VsOut vs(Inputs in [[stage_in]]) { /* … */ }".into(),
        },
        CacheEntry {
            kind: CachedKind::FfPs,
            key: 0,
            msl: String::new(),
        },
        CacheEntry {
            kind: CachedKind::Sm2Ps,
            key: 0x0102_0304_0506_0708,
            msl: "fragment float4 ps() { return float4(1); }".into(),
        },
    ]
}

#[test]
fn single_chunk_round_trip() {
    let entries = sample_entries();
    let buf = write_file(std::slice::from_ref(&entries), false);
    assert_eq!(read_header(&buf), Ok(SHADER_CACHE_SCHEMA_VERSION));
    let (read, needs_compaction) = read_records(&buf);
    assert_eq!(read, entries);
    // Singles only, no Bundle ⇒ compact next launch.
    assert!(needs_compaction);
}

#[test]
fn bundle_chunk_round_trip_is_optimal() {
    let entries = sample_entries();
    let buf = write_file(std::slice::from_ref(&entries), true);
    assert_eq!(read_header(&buf), Ok(SHADER_CACHE_SCHEMA_VERSION));
    let (read, needs_compaction) = read_records(&buf);
    assert_eq!(read, entries);
    // Exactly one Bundle, no dupes, EOF clean ⇒ optimal.
    assert!(!needs_compaction);
    // First chunk byte after the file header is the Bundle discriminator.
    assert_eq!(buf[HEADER_LEN], RECORD_KIND_BUNDLE);
}

#[test]
fn mixed_bundle_plus_singles_round_trip() {
    let bundle_entries = sample_entries();
    let later_appends = vec![CacheEntry {
        kind: CachedKind::Sm3Ps,
        key: 0xAAAA_BBBB_CCCC_DDDD,
        msl: "fragment float4 ps_later() { return float4(0,1,0,1); }".into(),
    }];
    let mut buf = Vec::new();
    write_header(&mut buf);
    write_bundle(&mut buf, &bundle_entries);
    for e in &later_appends {
        write_record(&mut buf, e);
    }
    let (read, needs_compaction) = read_records(&buf);
    let mut expected = bundle_entries.clone();
    expected.extend(later_appends);
    assert_eq!(read, expected);
    // Bundle + singles ⇒ not optimal.
    assert!(needs_compaction);
}

#[test]
fn torn_trailing_chunk_dropped_and_flags_compaction() {
    let entries = sample_entries();
    let mut buf = write_file(std::slice::from_ref(&entries), false);
    // Truncate mid-frame of the final chunk.
    buf.truncate(buf.len() - 5);
    let (read, needs_compaction) = read_records(&buf);
    // Dropped the torn last chunk.
    assert_eq!(read.len(), entries.len() - 1);
    assert!(needs_compaction);
}

#[test]
fn corrupt_chunk_header_caught_by_xxh3() {
    let mut buf = Vec::new();
    write_header(&mut buf);
    write_record(
        &mut buf,
        &CacheEntry {
            kind: CachedKind::Sm2Vs,
            key: 0xAABB,
            msl: "ok before".into(),
        },
    );
    let after_first = buf.len();
    write_record(
        &mut buf,
        &CacheEntry {
            kind: CachedKind::Sm2Ps,
            key: 0xCCDD,
            msl: "corrupted in header below".into(),
        },
    );
    // Flip a bit in the second chunk's `frame_len` field. Without
    // the xxh3 this would silently desync every subsequent parse;
    // with it, the chunk is detected as corrupt and we stop here.
    buf[after_first + 12] ^= 0x01;
    // Append one more well-formed chunk; since we can't trust the
    // corrupt frame_len to skip safely, this trailing chunk is
    // intentionally forfeit (recompiled next session).
    write_record(
        &mut buf,
        &CacheEntry {
            kind: CachedKind::Sm3Ps,
            key: 0xEEFF,
            msl: "ok after — forfeit on corruption-stop".into(),
        },
    );
    let (read, needs_compaction) = read_records(&buf);
    // Only the chunk before the corruption survives. The corrupt
    // chunk and everything after it are dropped; compaction rewrites
    // a clean file so the trailing chunk recompiles next launch.
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].key, 0xAABB);
    assert!(needs_compaction);
}

#[test]
fn corrupt_frame_body_caught_and_skipped() {
    let mut buf = Vec::new();
    write_header(&mut buf);
    write_record(
        &mut buf,
        &CacheEntry {
            kind: CachedKind::Sm2Vs,
            key: 0x1111,
            msl: "good".into(),
        },
    );
    let bad_start = buf.len();
    write_record(
        &mut buf,
        &CacheEntry {
            kind: CachedKind::Sm2Ps,
            key: 0x2222,
            msl: "frame body will be scrambled".into(),
        },
    );
    // Scramble a byte inside the second chunk's compressed frame
    // (past the 24-byte chunk header).
    buf[bad_start + CHUNK_HEADER_LEN + 2] ^= 0xFF;
    let (read, _) = read_records(&buf);
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].key, 0x1111);
}

#[test]
fn unknown_chunk_kind_skipped_via_frame_len() {
    let mut buf = Vec::new();
    write_header(&mut buf);
    // Hand-craft a chunk with kind = 0x42 (neither CachedKind nor Bundle),
    // valid xxh3, a tiny zstd frame as payload. Use write_record to
    // build a Single, then patch its kind byte after the fact and
    // recompute the checksum so the parser reaches the unknown-kind
    // arm rather than failing on xxh3.
    let weird_kind: u8 = 0x42;
    write_record(
        &mut buf,
        &CacheEntry {
            kind: CachedKind::FfVs,
            key: 0x9999,
            msl: "irrelevant".into(),
        },
    );
    let chunk_off = HEADER_LEN;
    buf[chunk_off] = weird_kind;
    let frame_len =
        u32::from_le_bytes(buf[chunk_off + 12..chunk_off + 16].try_into().unwrap()) as usize;
    let header16: [u8; 16] = buf[chunk_off..chunk_off + 16].try_into().unwrap();
    let frame_start = chunk_off + CHUNK_HEADER_LEN;
    let frame = &buf[frame_start..frame_start + frame_len];
    let new_checksum = chunk_xxh3(&header16, frame);
    buf[chunk_off + 16..chunk_off + 24].copy_from_slice(&new_checksum.to_le_bytes());
    // Followed by a valid chunk.
    write_record(
        &mut buf,
        &CacheEntry {
            kind: CachedKind::Sm3Ps,
            key: 0x4321,
            msl: "after weird".into(),
        },
    );
    let (read, needs_compaction) = read_records(&buf);
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].key, 0x4321);
    assert!(needs_compaction);
}

#[test]
fn duplicate_keys_flag_compaction() {
    let dup = CacheEntry {
        kind: CachedKind::Sm3Vs,
        key: 0x5555,
        msl: "first copy".into(),
    };
    let mut buf = Vec::new();
    write_header(&mut buf);
    write_bundle(&mut buf, &[dup.clone(), dup]);
    let (read, needs_compaction) = read_records(&buf);
    // Both entries are read; the dedupe is the caller's job.
    assert_eq!(read.len(), 2);
    assert!(needs_compaction);
}

#[test]
fn empty_file_with_just_header_is_not_compacted() {
    let mut buf = Vec::new();
    write_header(&mut buf);
    let (read, needs_compaction) = read_records(&buf);
    assert!(read.is_empty());
    // Nothing to compact ⇒ pre-warm leaves the file alone.
    assert!(!needs_compaction);
}

#[test]
fn read_header_rejects_wrong_magic() {
    let bytes = b"GARBAGE!\x01\x00\x00\x00\x00\x00\x00\x00";
    assert_eq!(read_header(bytes), Err(CacheReadError::WrongMagic));
}

#[test]
fn read_header_rejects_short_input() {
    let bytes = b"MTLD3DSH";
    assert_eq!(read_header(bytes), Err(CacheReadError::WrongMagic));
}

#[test]
fn read_header_returns_schema_for_caller_comparison() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&SHADER_CACHE_MAGIC);
    buf.extend_from_slice(&99u32.to_le_bytes());
    buf.extend_from_slice(&[0u8; 4]);
    assert_eq!(read_header(&buf), Ok(99));
}

#[test]
fn cached_kind_round_trips_via_byte() {
    for k in [
        CachedKind::FfVs,
        CachedKind::FfPs,
        CachedKind::Sm1Vs,
        CachedKind::Sm1Ps,
        CachedKind::Sm2Vs,
        CachedKind::Sm2Ps,
        CachedKind::Sm3Vs,
        CachedKind::Sm3Ps,
    ] {
        assert_eq!(CachedKind::from_byte(k as u8), Some(k));
    }
}

#[test]
fn from_programmable_maps_supported_majors() {
    assert_eq!(
        CachedKind::from_programmable(1, false),
        Some(CachedKind::Sm1Vs)
    );
    assert_eq!(
        CachedKind::from_programmable(2, true),
        Some(CachedKind::Sm2Ps)
    );
    assert_eq!(
        CachedKind::from_programmable(3, false),
        Some(CachedKind::Sm3Vs)
    );
    assert_eq!(CachedKind::from_programmable(0, false), None);
    assert_eq!(CachedKind::from_programmable(4, true), None);
}

#[test]
fn ff_key_hash_is_stable() {
    let a = (1u32, 2u32, 3u32);
    let b = (1u32, 2u32, 3u32);
    assert_eq!(ff_key_hash(&a), ff_key_hash(&b));
}
