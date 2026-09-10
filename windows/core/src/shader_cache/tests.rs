//! Unit tests for the on-disk shader-cache binary format.
//!
//! Hand-built files cover the round trips, the damage paths (torn tail, flipped chunk-header
//! bit, scrambled zstd frame, unknown chunk kind, stray file header), duplicate keys and a
//! header-only file. Every one of those but the scrambled-frame case pins the
//! `needs_compaction` verdict the pre-warm rewrite keys off. The file-level tests write real
//! files under the system temp root and pin that many writers reaching a cold cache together
//! still produce one header. Further tests cover header validation, `CachedKind` mapping and
//! `ff_key_hash` stability.

use super::*;

fn write_file(entries_per_chunk: &[Vec<CacheEntry>], bundle_last: bool) -> Vec<u8> {
    let mut buf = Vec::new();
    write_header(&mut buf);
    let last_idx = entries_per_chunk.len().saturating_sub(1);
    for (i, group) in entries_per_chunk.iter().enumerate() {
        let as_bundle = bundle_last && i == last_idx;
        if as_bundle {
            write_bundle(&mut buf, group, &[]);
        } else {
            for entry in group {
                write_record(&mut buf, entry);
            }
        }
    }
    buf
}

/// Count the occurrences of the file magic in `bytes`.
///
/// A file carrying its header once has exactly one; a duplicate creation
/// leaves a second run of the magic further in.
fn magic_count(bytes: &[u8]) -> usize {
    bytes
        .windows(SHADER_CACHE_MAGIC.len())
        .filter(|w| **w == SHADER_CACHE_MAGIC)
        .count()
}

/// An empty directory under the system temp root, unique to this process and this call.
fn scratch_dir(tag: &str) -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "mtld3d-shader-cache-{tag}-{}-{seq}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Append one record to the cache file at `path`, creating the file when absent.
fn append_one(path: &std::path::Path, entry: &CacheEntry) {
    let writer = CacheWriter::open(path).expect("open cache writer");
    writer.append_shader(entry).expect("append record");
}

fn read_shaders(bytes: &[u8]) -> (Vec<CacheEntry>, bool) {
    let records = read_records(bytes);
    assert!(records.pipelines.is_empty());
    (records.shaders, records.needs_compaction)
}

fn sample_entries() -> Vec<CacheEntry> {
    vec![
        CacheEntry {
            source: None,
            emitter_version: SHADER_EMITTER_VERSION,
            kind: CachedKind::Sm3Vs,
            key: 0xDEAD_BEEF_CAFE_BABE,
            msl: "vertex VsOut vs(Inputs in [[stage_in]]) { /* … */ }".into(),
        },
        CacheEntry {
            source: None,
            emitter_version: SHADER_EMITTER_VERSION,
            kind: CachedKind::FfPs,
            key: 0,
            msl: String::new(),
        },
        CacheEntry {
            source: None,
            emitter_version: SHADER_EMITTER_VERSION,
            kind: CachedKind::Sm2Ps,
            key: 0x0102_0304_0506_0708,
            msl: "fragment float4 ps() { return float4(1); }".into(),
        },
    ]
}

fn sample_recipe() -> PipelineRecipe {
    let entries = sample_entries();
    let mut stream_layouts = [StreamLayout::UNUSED; MAX_STREAMS as usize];
    stream_layouts[0] = StreamLayout {
        stride: 12,
        step: VertexStepFunction::PerVertex,
        step_rate: 1,
    };
    let snapshot = PipelineSnapshot {
        vs_fn: MetalHandle::NULL,
        ps_fn: MetalHandle::NULL,
        vdecl_hash: 0x1234,
        stream_layouts,
        color_format: PixelFormat::Bgra8Unorm,
        attach: PipelineAttachFlags::HAS_COLOR_OUTPUT | PipelineAttachFlags::COLOR_HAS_ALPHA,
        rs: PipelineRsBits {
            color_write_mask: 0x0F,
            ..PipelineRsBits::default()
        },
        extra: ExtraColorAttachments::NONE,
        ps_color_out_mask: 1,
        sample_count: 1,
    };
    PipelineRecipe::from_snapshot(
        ShaderRecordRef::new(entries[0].kind, entries[0].key),
        ShaderRecordRef::new(entries[2].kind, entries[2].key),
        &snapshot,
        &[VertexAttrDesc {
            attr_index: 0,
            buffer_index: 0,
            offset: 0,
            format: VertexFormat::Float3,
        }],
    )
}

#[test]
fn single_chunk_round_trip() {
    let entries = sample_entries();
    let buf = write_file(std::slice::from_ref(&entries), false);
    assert_eq!(read_header(&buf), Ok(CacheHeader::CURRENT));
    let (read, needs_compaction) = read_shaders(&buf);
    assert_eq!(read, entries);
    // Singles only, no Bundle ⇒ compact next launch.
    assert!(needs_compaction);
}

#[test]
fn bundle_chunk_round_trip_is_optimal() {
    let entries = sample_entries();
    let buf = write_file(std::slice::from_ref(&entries), true);
    assert_eq!(read_header(&buf), Ok(CacheHeader::CURRENT));
    let (read, needs_compaction) = read_shaders(&buf);
    assert_eq!(read, entries);
    // Exactly one Bundle, no dupes, EOF clean ⇒ optimal.
    assert!(!needs_compaction);
    // First chunk byte after the file header is the Bundle discriminator.
    assert_eq!(buf[HEADER_LEN], RECORD_KIND_BUNDLE);
}

#[test]
fn pipeline_recipe_round_trips_with_stable_shader_refs() {
    let entries = sample_entries();
    let recipe = sample_recipe();
    let mut buf = Vec::new();
    write_header(&mut buf);
    write_bundle(&mut buf, &entries, std::slice::from_ref(&recipe));
    let records = read_records(&buf);
    assert_eq!(records.shaders, entries);
    assert!(records.pipelines == vec![recipe]);
    assert!(!records.needs_compaction);
}

#[test]
fn attachmentless_recipes_are_removed_without_losing_valid_records() {
    let entries = sample_entries();
    let mut empty = sample_recipe();
    empty.snapshot.attach = PipelineAttachFlags::empty();
    empty.snapshot.rs.color_write_mask = 0;
    let mut depth_only = sample_recipe();
    depth_only.snapshot.attach = PipelineAttachFlags::HAS_DEPTH;
    depth_only.snapshot.rs.color_write_mask = 0;
    let mut extra_only = sample_recipe();
    extra_only.snapshot.attach = PipelineAttachFlags::empty();
    extra_only.snapshot.extra.present_mask = 1;
    let recipes = [sample_recipe(), empty, depth_only, extra_only];

    for bundled in [false, true] {
        let mut bytes = Vec::new();
        write_header(&mut bytes);
        if bundled {
            write_bundle(&mut bytes, &entries, &recipes);
        } else {
            for entry in &entries {
                write_record(&mut bytes, entry);
            }
            for recipe in &recipes {
                write_pipeline_record(&mut bytes, recipe);
            }
        }
        let records = read_records(&bytes);
        assert_eq!(records.shaders, entries);
        assert_eq!(records.pipelines.len(), 3);
        assert!(records.pipelines[0] == recipes[0]);
        assert!(records.pipelines[1] == recipes[2]);
        assert!(records.pipelines[2] == recipes[3]);
        assert!(records.needs_compaction);

        let mut compacted = Vec::new();
        write_header(&mut compacted);
        write_bundle(&mut compacted, &records.shaders, &records.pipelines);
        let reloaded = read_records(&compacted);
        assert_eq!(reloaded.shaders, entries);
        assert!(reloaded.pipelines == records.pipelines);
        assert!(!reloaded.needs_compaction);
    }
}

#[test]
fn pipeline_recipe_without_both_shader_records_is_dropped() {
    let entries = sample_entries();
    let recipe = sample_recipe();
    let mut buf = Vec::new();
    write_header(&mut buf);
    write_bundle(&mut buf, &entries[..1], std::slice::from_ref(&recipe));
    let records = read_records(&buf);
    assert!(records.pipelines.is_empty());
    assert!(records.needs_compaction);
}

#[test]
fn load_repairs_a_torn_tail_before_another_writer_appends() {
    let dir = scratch_dir("load-repair");
    let path = dir.join("mtld3d_shaders.bin");
    let entries = sample_entries();
    let mut bytes = write_file(std::slice::from_ref(&entries), false);
    bytes.truncate(bytes.len() - 5);
    std::fs::write(&path, bytes).expect("write torn cache");

    let CacheLoad::Current(records) = load(&path).expect("load torn cache") else {
        panic!("torn cache lost its valid header");
    };
    assert_eq!(records.shaders, entries[..2]);
    // A live encoder can write while another device compiles its startup
    // snapshot. Its records must remain reachable by the later compactor.
    append_one(&path, &entries[2]);
    compact(&path).expect("compact repaired cache");
    let records = read_records(&std::fs::read(&path).expect("read repaired cache"));
    assert_eq!(records.shaders, entries);
    assert!(!records.needs_compaction);
    std::fs::remove_dir_all(&dir).expect("remove scratch dir");
}

#[test]
fn recipes_preserve_instancing_constant_streams_mrt_and_siblings() {
    let mut recipe = sample_recipe();
    recipe.snapshot.stream_layouts[0].step = VertexStepFunction::Constant;
    recipe.snapshot.stream_layouts[0].step_rate = 0;
    recipe.snapshot.stream_layouts[15] = StreamLayout {
        stride: 32,
        step: VertexStepFunction::PerInstance,
        step_rate: 7,
    };
    recipe.vertex_attrs.push(VertexAttrDesc {
        attr_index: 15,
        buffer_index: 15,
        offset: 16,
        format: VertexFormat::Float4,
    });
    recipe.snapshot.sample_count = 4;
    recipe
        .snapshot
        .attach
        .insert(PipelineAttachFlags::HAS_DEPTH | PipelineAttachFlags::HAS_STENCIL);
    recipe.snapshot.extra = ExtraColorAttachments {
        formats: [
            PixelFormat::Rgba16Float,
            PixelFormat::Bgra8Unorm,
            PixelFormat::R32Float,
        ],
        present_mask: 0b101,
        has_alpha_mask: 0b001,
    };
    recipe.snapshot.ps_color_out_mask = 0b1011;
    recipe.snapshot.rs.flags = PipelineRsFlags::all();
    recipe.snapshot.rs.src_blend_alpha = 5;
    recipe.snapshot.rs.dst_blend_alpha = 6;
    recipe.snapshot.rs.blend_op_alpha = 3;
    recipe.snapshot.rs.color_write_mask = 0;
    recipe.snapshot.rs.color_write_mask_ext = [0; 3];
    let mut sibling_snapshot = recipe.snapshot.clone();
    sibling_snapshot
        .attach
        .remove(PipelineAttachFlags::HAS_COLOR_OUTPUT);
    sibling_snapshot.extra = ExtraColorAttachments::NONE;
    let sibling = PipelineRecipe::from_snapshot(
        recipe.vs,
        recipe.ps,
        &sibling_snapshot,
        recipe.vertex_attrs(),
    );

    let mut bytes = Vec::new();
    write_header(&mut bytes);
    let recipes = [recipe, sibling];
    write_bundle(&mut bytes, &sample_entries(), &recipes);
    let records = read_records(&bytes);
    assert!(records.pipelines == recipes);
    assert!(!records.needs_compaction);
}

#[test]
fn duplicate_recipes_share_shader_records_after_compaction() {
    let dir = scratch_dir("recipe-dedup");
    let path = dir.join("mtld3d_shaders.bin");
    let entries = sample_entries();
    let writer = CacheWriter::open(&path).expect("open writer");
    for entry in &entries {
        writer.append_shader(entry).expect("append shader");
        writer
            .append_shader(entry)
            .expect("append duplicate shader");
    }
    let first = sample_recipe();
    let mut second = sample_recipe();
    second
        .snapshot
        .rs
        .flags
        .insert(PipelineRsFlags::BLEND_ENABLE);
    for recipe in [&first, &second, &first] {
        writer.append_pipeline(recipe).expect("append recipe");
    }
    compact(&path).expect("compact recipes");
    let records = read_records(&std::fs::read(&path).expect("read compacted recipes"));
    assert_eq!(records.shaders, entries);
    assert!(records.pipelines == [first, second]);
    assert!(!records.needs_compaction);
    std::fs::remove_dir_all(&dir).expect("remove scratch dir");
}

#[test]
fn malformed_pipeline_payloads_are_rejected_with_valid_checksums() {
    let mut valid = Vec::new();
    sample_recipe().encode(&mut valid);
    for length in 0..valid.len() {
        assert!(PipelineRecipe::decode(&valid[..length]).is_none());
    }
    for (offset, value) in [(0, CachedKind::FfPs as u8), (2, 0x80), (30, 0), (31, 0x80)] {
        let mut invalid = valid.clone();
        invalid[offset] = value;
        let frame = zstd::encode_all(invalid.as_slice(), ZSTD_APPEND_LEVEL)
            .expect("compress invalid recipe");
        let mut bytes = write_file(&[sample_entries()], false);
        push_chunk(
            &mut bytes,
            RECORD_KIND_PIPELINE,
            sample_recipe().disk_key(),
            &frame,
        );
        let records = read_records(&bytes);
        assert!(records.pipelines.is_empty());
        assert!(records.needs_compaction);
    }
}

#[test]
fn mixed_bundle_plus_singles_round_trip() {
    let bundle_entries = sample_entries();
    let later_appends = vec![CacheEntry {
        source: None,
        emitter_version: SHADER_EMITTER_VERSION,
        kind: CachedKind::Sm3Ps,
        key: 0xAAAA_BBBB_CCCC_DDDD,
        msl: "fragment float4 ps_later() { return float4(0,1,0,1); }".into(),
    }];
    let mut buf = Vec::new();
    write_header(&mut buf);
    write_bundle(&mut buf, &bundle_entries, &[]);
    for e in &later_appends {
        write_record(&mut buf, e);
    }
    let (read, needs_compaction) = read_shaders(&buf);
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
    let (read, needs_compaction) = read_shaders(&buf);
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
            source: None,
            emitter_version: SHADER_EMITTER_VERSION,
            kind: CachedKind::Sm2Vs,
            key: 0xAABB,
            msl: "ok before".into(),
        },
    );
    let after_first = buf.len();
    write_record(
        &mut buf,
        &CacheEntry {
            source: None,
            emitter_version: SHADER_EMITTER_VERSION,
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
            source: None,
            emitter_version: SHADER_EMITTER_VERSION,
            kind: CachedKind::Sm3Ps,
            key: 0xEEFF,
            msl: "ok after — forfeit on corruption-stop".into(),
        },
    );
    let (read, needs_compaction) = read_shaders(&buf);
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
            source: None,
            emitter_version: SHADER_EMITTER_VERSION,
            kind: CachedKind::Sm2Vs,
            key: 0x1111,
            msl: "good".into(),
        },
    );
    let bad_start = buf.len();
    write_record(
        &mut buf,
        &CacheEntry {
            source: None,
            emitter_version: SHADER_EMITTER_VERSION,
            kind: CachedKind::Sm2Ps,
            key: 0x2222,
            msl: "frame body will be scrambled".into(),
        },
    );
    // Scramble a byte inside the second chunk's compressed frame
    // (past the 24-byte chunk header).
    buf[bad_start + CHUNK_HEADER_LEN + 2] ^= 0xFF;
    let (read, _) = read_shaders(&buf);
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
            source: None,
            emitter_version: SHADER_EMITTER_VERSION,
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
            source: None,
            emitter_version: SHADER_EMITTER_VERSION,
            kind: CachedKind::Sm3Ps,
            key: 0x4321,
            msl: "after weird".into(),
        },
    );
    let (read, needs_compaction) = read_shaders(&buf);
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].key, 0x4321);
    assert!(needs_compaction);
}

#[test]
fn duplicate_keys_flag_compaction() {
    let dup = CacheEntry {
        source: None,
        emitter_version: SHADER_EMITTER_VERSION,
        kind: CachedKind::Sm3Vs,
        key: 0x5555,
        msl: "first copy".into(),
    };
    let mut buf = Vec::new();
    write_header(&mut buf);
    write_bundle(&mut buf, &[dup.clone(), dup], &[]);
    let (read, needs_compaction) = read_shaders(&buf);
    assert_eq!(read.len(), 1);
    assert!(needs_compaction);
}

#[test]
fn empty_file_with_just_header_is_compacted() {
    let mut buf = Vec::new();
    write_header(&mut buf);
    let (read, needs_compaction) = read_shaders(&buf);
    assert!(read.is_empty());
    assert!(needs_compaction);
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
fn read_header_returns_both_versions() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&SHADER_CACHE_MAGIC);
    buf.extend_from_slice(&99u32.to_le_bytes());
    buf.extend_from_slice(&100u32.to_le_bytes());
    assert_eq!(
        read_header(&buf),
        Ok(CacheHeader {
            format_version: 99,
            shader_schema_version: 100,
        })
    );
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

#[test]
fn stray_file_header_mid_file_is_skipped() {
    let entries = sample_entries();
    let mut buf = Vec::new();
    write_header(&mut buf);
    write_record(&mut buf, &entries[0]);
    // A second creation's header, landing where a chunk header belongs.
    write_header(&mut buf);
    write_record(&mut buf, &entries[1]);
    let (read, needs_compaction) = read_shaders(&buf);
    assert_eq!(read, vec![entries[0].clone(), entries[1].clone()]);
    // The stray header is a reason to rewrite the file dense.
    assert!(needs_compaction);
}

#[test]
fn open_for_append_writes_one_header_across_writers() {
    let dir = scratch_dir("append");
    let path = dir.join("mtld3d_shaders.bin");
    let entries = sample_entries();
    for entry in &entries {
        append_one(&path, entry);
    }
    let bytes = std::fs::read(&path).expect("read cache file");
    assert_eq!(magic_count(&bytes), 1);
    assert_eq!(read_header(&bytes), Ok(CacheHeader::CURRENT));
    let (read, _) = read_shaders(&bytes);
    assert_eq!(read, entries);
    std::fs::remove_dir_all(&dir).expect("remove scratch dir");
}

#[test]
fn concurrent_open_for_append_writes_one_header() {
    const WRITERS: usize = 8;

    let dir = scratch_dir("race");
    let path = dir.join("mtld3d_shaders.bin");
    // Every writer reaches the cold cache in the same instant, which is what a
    // set of devices whose first miss-compiles coincide does.
    let start = std::sync::Barrier::new(WRITERS);
    std::thread::scope(|scope| {
        for key in 0..WRITERS {
            let path = path.as_path();
            let start = &start;
            scope.spawn(move || {
                let key = u64::try_from(key).expect("writer index fits u64");
                start.wait();
                append_one(
                    path,
                    &CacheEntry {
                        source: None,
                        emitter_version: SHADER_EMITTER_VERSION,
                        kind: CachedKind::Sm2Ps,
                        key,
                        msl: format!("fragment float4 ps{key}() {{ return float4({key}); }}"),
                    },
                );
            });
        }
    });
    let bytes = std::fs::read(&path).expect("read cache file");
    assert_eq!(magic_count(&bytes), 1);
    assert_eq!(read_header(&bytes), Ok(CacheHeader::CURRENT));
    let (read, _) = read_shaders(&bytes);
    assert_eq!(read.len(), WRITERS);
    std::fs::remove_dir_all(&dir).expect("remove scratch dir");
}

#[test]
fn compact_renames_one_dense_file_into_place() {
    let dir = scratch_dir("bundle");
    let path = dir.join("mtld3d_shaders.bin");
    let entries = sample_entries();
    for entry in &entries {
        append_one(&path, entry);
    }
    let (_, len) = compact(&path)
        .expect("compact cache")
        .expect("cache needed compaction");
    let bytes = std::fs::read(&path).expect("read cache file");
    assert_eq!(bytes.len(), len);
    assert_eq!(magic_count(&bytes), 1);
    let (read, needs_compaction) = read_shaders(&bytes);
    assert_eq!(read, entries);
    assert!(!needs_compaction);
    // The temporary is renamed rather than left beside the cache file.
    let leftovers = std::fs::read_dir(&dir).expect("list scratch dir").count();
    assert_eq!(leftovers, 2);
    std::fs::remove_dir_all(&dir).expect("remove scratch dir");
}

#[test]
fn append_and_compaction_race_does_not_lose_records() {
    const ROUNDS: u64 = 16;

    let dir = scratch_dir("compact-race");
    let path = dir.join("mtld3d_shaders.bin");
    append_one(&path, &sample_entries()[0]);
    for round in 0..ROUNDS {
        let start = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let cache_path = path.as_path();
            let barrier = &start;
            scope.spawn(move || {
                barrier.wait();
                compact(cache_path).expect("compact raced cache");
            });
            scope.spawn(move || {
                barrier.wait();
                append_one(
                    cache_path,
                    &CacheEntry {
                        source: None,
                        emitter_version: SHADER_EMITTER_VERSION,
                        kind: CachedKind::Sm2Ps,
                        key: round + 1,
                        msl: format!("fragment float4 ps{round}() {{ return 1; }}"),
                    },
                );
            });
        });
    }
    let bytes = std::fs::read(&path).expect("read raced cache");
    let records = read_records(&bytes);
    assert_eq!(
        records.shaders.len(),
        usize::try_from(ROUNDS).expect("round count fits usize") + 1
    );
    std::fs::remove_dir_all(&dir).expect("remove scratch dir");
}

#[test]
fn concurrent_loaders_and_writers_observe_valid_records() {
    const WRITERS: u64 = 2;
    const ROUNDS: u64 = 32;

    let dir = scratch_dir("load-write-race");
    let path = dir.join("mtld3d_shaders.bin");
    append_one(&path, &sample_entries()[0]);
    let start = std::sync::Barrier::new(usize::try_from(WRITERS).expect("writer count fits") + 1);
    std::thread::scope(|scope| {
        for writer in 0..WRITERS {
            let cache_path = path.as_path();
            let barrier = &start;
            scope.spawn(move || {
                barrier.wait();
                for round in 0..ROUNDS {
                    append_one(
                        cache_path,
                        &CacheEntry {
                            source: None,
                            emitter_version: SHADER_EMITTER_VERSION,
                            kind: CachedKind::Sm3Ps,
                            key: 0x10_0000 + writer * ROUNDS + round,
                            msl: format!("fragment float4 ps{writer}_{round}() {{ return 1; }}"),
                        },
                    );
                }
            });
        }
        let cache_path = path.as_path();
        let barrier = &start;
        scope.spawn(move || {
            barrier.wait();
            for _ in 0..ROUNDS {
                let CacheLoad::Current(records) = load(cache_path).expect("load raced cache")
                else {
                    panic!("raced cache stopped being current");
                };
                assert!(!records.shaders.is_empty());
            }
        });
    });
    let CacheLoad::Current(records) = load(&path).expect("load completed cache") else {
        panic!("completed cache stopped being current");
    };
    assert_eq!(
        records.shaders.len(),
        usize::try_from(WRITERS * ROUNDS + 1).expect("record count fits")
    );
    std::fs::remove_dir_all(&dir).expect("remove scratch dir");
}

#[test]
fn load_invalidates_stale_versions_under_lock() {
    let dir = scratch_dir("stale");
    let path = dir.join("mtld3d_shaders.bin");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&SHADER_CACHE_MAGIC);
    bytes.extend_from_slice(&(CACHE_FORMAT_VERSION - 1).to_le_bytes());
    bytes.extend_from_slice(&SHADER_CACHE_SCHEMA_VERSION.to_le_bytes());
    std::fs::write(&path, bytes).expect("write stale cache");
    assert!(matches!(
        load(&path).expect("load stale cache"),
        CacheLoad::InvalidatedVersion(_)
    ));
    assert!(!path.exists());
    std::fs::remove_dir_all(&dir).expect("remove scratch dir");
}

fn programmable_entry(kind: CachedKind) -> CacheEntry {
    use crate::dxso::{VariantFlags, VariantKey, VsSamplerKinds, parse};

    let major = match kind {
        CachedKind::Sm1Vs | CachedKind::Sm1Ps => 1,
        CachedKind::Sm2Vs | CachedKind::Sm2Ps => 2,
        CachedKind::Sm3Vs | CachedKind::Sm3Ps => 3,
        CachedKind::FfVs | CachedKind::FfPs => panic!("programmable fixture only"),
    };
    let header = if kind.is_vertex() {
        0xFFFE_0000
    } else {
        0xFFFF_0000
    };
    let minor = u32::from(major == 1);
    let program = parse(&[header | (major << 8) | minor, 0x0000_FFFF]).expect("parse DXSO");
    let source = if kind.is_vertex() {
        ShaderSource::vertex(
            &program,
            0xA55A,
            3,
            VsSamplerKinds {
                volume_mask: 1,
                cube_mask: 2,
            },
        )
    } else {
        ShaderSource::pixel(
            &program,
            VariantKey {
                alpha_func: 5,
                fog_mode: 4,
                fog_table_mode: 3,
                depth_sampler_mask: 0x1234,
                depth_fetch_mask: 0x0034,
                volume_sampler_mask: 0x4000,
                cube_sampler_mask: 0x8000,
                tt_projected_mask: 0x85,
                color_out_mask: 0x0F,
                sample_mask: 0x5A,
                flags: VariantFlags::all(),
            },
        )
    };
    let key = source.disk_key();
    let msl = source.emit(&kind.entry_name(key)).expect("emit fixture");
    CacheEntry::new(kind, key, msl, Some(source))
}

#[test]
fn emitter_change_retains_and_rebuilds_every_programmable_stage_and_model() {
    for kind in [
        CachedKind::Sm1Vs,
        CachedKind::Sm1Ps,
        CachedKind::Sm2Vs,
        CachedKind::Sm2Ps,
        CachedKind::Sm3Vs,
        CachedKind::Sm3Ps,
    ] {
        let expected = programmable_entry(kind);
        let mut stale = expected.clone();
        stale.emitter_version ^= 1;
        stale.msl = "obsolete MSL must never compile".into();
        for bundle in [false, true] {
            let bytes = write_file(&[vec![stale.clone()]], bundle);
            let mut records = read_records(&bytes);
            assert_eq!(records.shaders.len(), 1);
            let entry = &mut records.shaders[0];
            assert_eq!(entry.source, expected.source, "all emission inputs survive");
            assert!(entry.refresh_msl().expect("rebuild stale MSL"));
            assert_eq!(*entry, expected, "same specialization, name, and key");
            assert!(!entry.refresh_msl().expect("current MSL reuses its source"));
        }
    }
}

#[test]
fn emitter_change_discards_ff_and_only_its_dependent_pipelines() {
    let vs = programmable_entry(CachedKind::Sm3Vs);
    let ps = programmable_entry(CachedKind::Sm3Ps);
    let mut programmable = sample_recipe();
    programmable.vs = ShaderRecordRef::new(vs.kind, vs.key);
    programmable.ps = ShaderRecordRef::new(ps.kind, ps.key);
    let mut mixed = sample_recipe();
    mixed.vs = programmable.vs;
    mixed.ps = ShaderRecordRef::new(CachedKind::FfPs, 42);
    let mut entries = vec![
        vs,
        ps,
        CacheEntry::new(CachedKind::FfPs, 42, "old FF".into(), None),
    ];
    for entry in &mut entries {
        entry.emitter_version ^= 1;
    }
    let mut bytes = Vec::new();
    write_header(&mut bytes);
    write_bundle(&mut bytes, &entries, &[programmable, mixed]);
    let records = read_records(&bytes);
    assert_eq!(records.shaders.len(), 2);
    assert_eq!(records.pipelines.len(), 1);
    assert!(records.pipelines[0].ps.kind.is_programmable());
    assert!(records.needs_compaction);
}

#[test]
fn refreshed_msl_wins_in_either_order_and_compaction_preserves_concurrent_appends() {
    let dir = scratch_dir("emitter-refresh");
    let path = dir.join("mtld3d_shaders.bin");
    let expected = programmable_entry(CachedKind::Sm3Vs);
    let mut stale = expected.clone();
    stale.emitter_version ^= 1;
    stale.msl = "stale".into();
    for current_first in [false, true] {
        let entries = if current_first {
            vec![expected.clone(), stale.clone()]
        } else {
            vec![stale.clone(), expected.clone()]
        };
        let bytes = write_file(&[entries], false);
        assert_eq!(
            read_records(&bytes).shaders.as_slice(),
            std::slice::from_ref(&expected)
        );
    }
    append_one(&path, &stale);
    let CacheLoad::Current(mut loaded) = load(&path).expect("load old emitter") else {
        panic!("emitter changes do not invalidate the file");
    };
    assert!(loaded.shaders[0].refresh_msl().expect("refresh"));
    let other = programmable_entry(CachedKind::Sm2Ps);
    append_one(&path, &other);
    append_one(&path, &loaded.shaders[0]);
    // An older build may append after the refreshed record, too.
    append_one(&path, &stale);
    compact(&path)
        .expect("compact merged file")
        .expect("rewrite needed");
    let CacheLoad::Current(loaded) = load(&path).expect("reload") else {
        panic!("current file");
    };
    assert_eq!(loaded.shaders, [expected, other]);
    assert!(!loaded.needs_compaction);
    std::fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn malformed_source_and_wrong_identity_are_rejected() {
    let entry = programmable_entry(CachedKind::Sm3Vs);
    let mut body = Vec::new();
    entry.encode(&mut body);
    assert!(CacheEntry::decode(entry.kind, entry.key ^ 1, &body).is_none());
    assert!(CacheEntry::decode(CachedKind::Sm3Ps, entry.key, &body).is_none());
    assert!(CacheEntry::decode(CachedKind::FfVs, entry.key, &body).is_none());
    // Header + presence + VS specialization + DXSO count cannot be torn.
    for length in 0..26 {
        assert!(CacheEntry::decode(entry.kind, entry.key, &body[..length]).is_none());
    }
    // A forged token count cannot allocate past the encoded body.
    body[14..18].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(CacheEntry::decode(entry.kind, entry.key, &body).is_none());
}

#[test]
fn regeneration_failure_preserves_dxso_for_retry() {
    let mut entry = programmable_entry(CachedKind::Sm3Vs);
    // A framed record can contain DXSO a newer parser refuses.
    let tokens = [0xFFFE_0300, 0x0000_DEAD, 0x0000_FFFF];
    let mut body = Vec::new();
    entry.emitter_version ^= 1;
    entry.encode(&mut body);
    body[14..18].copy_from_slice(&3u32.to_le_bytes());
    body.splice(22..22, 0x0000_DEADu32.to_le_bytes());
    let key = vs_source_disk_key_programmable(
        crate::ids::ProgramId::from_tokens(&tokens),
        0xA55A,
        3,
        crate::dxso::VsSamplerKinds {
            volume_mask: 1,
            cube_mask: 2,
        },
    );
    let entry = CacheEntry::decode(entry.kind, key, &body).expect("framed DXSO source");
    let before = entry.clone();
    let bytes = write_file(&[vec![entry]], true);
    let mut loaded = read_records(&bytes);
    assert!(loaded.shaders[0].refresh_msl().is_err());
    assert_eq!(loaded.shaders[0], before);
}
