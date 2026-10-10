//! Unit tests for the rename pool behind the encoder's sampled-texture renames.
//!
//! The propositions pinned here: a version is handed back only after its
//! submission has settled, spares are taken newest first and only by their
//! own texture, the parked bytes follow every park, take, trim and drain,
//! the cap and the age bound give storage back, and a version whose texture
//! is gone or changed shape is destroyed rather than parked. Then the write
//! history: which textures keep one, how writes stamp the tile grid, and
//! the copy plan a reused version gets from it, down to the cases that must
//! fall back to whole-level copies.

use super::*;

/// A creation descriptor of the given 2D shape.
fn desc(width: u32, height: u32, levels: u32, pixel_format: PixelFormat) -> TextureCreateDesc {
    TextureCreateDesc {
        tex_id: 1,
        width,
        height,
        depth: 1,
        levels,
        pixel_format,
        storage_mode: StorageMode::Private,
        flags: TextureCreateFlags::empty(),
        swizzle_r: Swizzle::Red,
        swizzle_g: Swizzle::Green,
        swizzle_b: Swizzle::Blue,
        swizzle_a: Swizzle::Alpha,
        usage_flags: TextureUsage::empty(),
    }
}

/// The 512x512 A4R4G4B4 sprite scratch: 512 KiB per version.
fn sprite() -> VersionShape {
    VersionShape::of(&desc(512, 512, 1, PixelFormat::Abgr4Unorm))
}

const SPRITE_BYTES: u64 = 512 * 512 * 2;

fn id(raw: u64) -> TextureId {
    TextureId::from_raw(raw)
}

/// Pop and park everything `settled` covers, returning what came back to destroy.
fn reclaim(versions: &mut TextureVersions<&'static str>, settled: u64) -> Vec<&'static str> {
    let mut destroyed = Vec::new();
    while let Some(entry) = versions.pop_retired(settled) {
        destroyed.extend(versions.park(entry, 0));
    }
    destroyed
}

#[test]
fn shape_bytes_count_every_level_and_face_in_whole_blocks() {
    assert_eq!(sprite().bytes(), SPRITE_BYTES);
    // 8x8 BGRA8 with a full chain: 256 + 64 + 16 + 4 bytes.
    let chain = VersionShape::of(&desc(8, 8, 4, PixelFormat::Bgra8Unorm));
    assert_eq!(chain.bytes(), 340);
    // 100x60 DXT1: levels of 100x60, 50x30, 25x15, 12x7, 6x3, 3x1 and 1x1
    // are 25x15, 13x8, 7x4, 3x2, 2x1, 1x1 and 1x1 blocks of 8 bytes; a level
    // narrower than a block still costs one.
    let dxt = VersionShape::of(&desc(100, 60, 7, PixelFormat::Bc1Rgba));
    let blocks = 25 * 15 + 13 * 8 + 7 * 4 + 3 * 2 + 2 + 1 + 1;
    assert_eq!(dxt.bytes(), blocks * 8);
    let mut cube = desc(16, 16, 1, PixelFormat::Bgra8Unorm);
    cube.flags = TextureCreateFlags::TYPE_CUBE;
    assert_eq!(VersionShape::of(&cube).bytes(), 6 * 16 * 16 * 4);
    let mut volume = desc(8, 8, 2, PixelFormat::R8Unorm);
    volume.depth = 4;
    volume.flags = TextureCreateFlags::TYPE_3D;
    assert_eq!(VersionShape::of(&volume).bytes(), 8 * 8 * 4 + 4 * 4 * 2);
}

#[test]
fn a_first_rename_finds_no_spare() {
    let mut versions = TextureVersions::<&str>::new();
    let (spare, discarded) = versions.take_spare(id(1), sprite(), false, 0);
    assert!(spare.is_none());
    assert!(discarded.is_empty());
}

#[test]
fn a_version_is_not_reclaimed_before_its_submission_settles() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite(), false, 0));
    versions.retire(id(1), "v1", 5);
    assert!(versions.pop_retired(4).is_none());
    let entry = versions.pop_retired(5).expect("seq 5 settled");
    assert_eq!(*entry.views(), "v1");
    assert!(versions.park(entry, 0).is_none());
    assert_eq!(versions.parked_bytes(), SPRITE_BYTES);
}

#[test]
fn the_retiring_queue_stops_at_the_first_unsettled_submission() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite(), false, 0));
    versions.retire(id(1), "v1", 5);
    versions.retire(id(1), "v2", 6);
    versions.retire(id(1), "v3", 7);
    assert!(reclaim(&mut versions, 6).is_empty());
    assert_eq!(versions.parked_bytes(), 2 * SPRITE_BYTES);
    // "v3" is still in flight: the next rename gets "v2", then "v1".
    let (first, _) = versions.take_spare(id(1), sprite(), false, 0);
    let (second, _) = versions.take_spare(id(1), sprite(), false, 0);
    let (third, _) = versions.take_spare(id(1), sprite(), false, 0);
    assert_eq!(first.map(Spare::into_views), Some("v2"));
    assert_eq!(second.map(Spare::into_views), Some("v1"));
    assert!(third.is_none());
    assert_eq!(versions.parked_bytes(), 0);
}

#[test]
fn spares_serve_only_their_own_texture() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite(), false, 0));
    drop(versions.take_spare(id(2), sprite(), false, 0));
    versions.retire(id(1), "a1", 3);
    assert!(reclaim(&mut versions, 3).is_empty());
    let (other, _) = versions.take_spare(id(2), sprite(), false, 0);
    assert!(other.is_none());
    let (own, _) = versions.take_spare(id(1), sprite(), false, 0);
    assert_eq!(own.map(Spare::into_views), Some("a1"));
}

#[test]
fn a_version_of_a_released_texture_is_destroyed_when_it_settles() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite(), false, 0));
    versions.retire(id(1), "parked", 2);
    assert!(reclaim(&mut versions, 2).is_empty());
    versions.retire(id(1), "in flight", 3);
    // Releasing the texture hands back its spares at once and leaves the
    // in-flight version for the drain.
    assert_eq!(versions.forget_texture(id(1)), vec!["parked"]);
    assert_eq!(versions.parked_bytes(), 0);
    assert_eq!(reclaim(&mut versions, 3), vec!["in flight"]);
    assert_eq!(versions.parked_bytes(), 0);
}

#[test]
fn a_texture_back_under_its_id_never_parks_a_version_of_its_former_life() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite(), false, 0));
    versions.retire(id(1), "before", 2);
    // Evicted while that version is in flight, then renamed again under the
    // same id before the drain pops it.
    assert!(versions.forget_texture(id(1)).is_empty());
    let (spare, _) = versions.take_spare(id(1), sprite(), false, 0);
    assert!(spare.is_none());
    versions.retire(id(1), "after", 2);
    assert_eq!(reclaim(&mut versions, 2), vec!["before"]);
    assert_eq!(versions.parked_bytes(), SPRITE_BYTES);
    let (spare, _) = versions.take_spare(id(1), sprite(), false, 0);
    assert_eq!(spare.map(Spare::into_views), Some("after"));
}

#[test]
fn a_shape_change_discards_the_spares_of_the_old_shape() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite(), false, 0));
    versions.retire(id(1), "old parked", 2);
    assert!(reclaim(&mut versions, 2).is_empty());
    versions.retire(id(1), "old retiring", 3);
    let bigger = VersionShape::of(&desc(1024, 512, 1, PixelFormat::Abgr4Unorm));
    let (spare, discarded) = versions.take_spare(id(1), bigger, false, 0);
    assert!(spare.is_none());
    assert_eq!(discarded, vec!["old parked"]);
    assert_eq!(versions.parked_bytes(), 0);
    // A version retired under the old shape cannot serve the new one.
    assert_eq!(reclaim(&mut versions, 3), vec!["old retiring"]);
    versions.retire(id(1), "new", 4);
    assert!(reclaim(&mut versions, 4).is_empty());
    assert_eq!(versions.parked_bytes(), 2 * SPRITE_BYTES);
}

#[test]
fn the_cap_destroys_what_would_take_the_pool_past_it() {
    // 4096x4096 BGRA8 is 64 MiB a version, so four fill the 256 MiB cap.
    let big = || VersionShape::of(&desc(4096, 4096, 1, PixelFormat::Bgra8Unorm));
    assert_eq!(big().bytes() * 4, VERSION_POOL_CAP_BYTES);
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), big(), false, 0));
    for (seq, label) in (1..).zip(["a", "b", "c", "d", "e"]) {
        versions.retire(id(1), label, seq);
    }
    assert_eq!(reclaim(&mut versions, 5), vec!["e"]);
    assert_eq!(versions.parked_bytes(), VERSION_POOL_CAP_BYTES);
    // Taking one makes room for the next.
    let (spare, _) = versions.take_spare(id(1), big(), false, 0);
    assert_eq!(spare.map(Spare::into_views), Some("d"));
    versions.retire(id(1), "f", 6);
    assert!(reclaim(&mut versions, 6).is_empty());
    assert_eq!(versions.parked_bytes(), VERSION_POOL_CAP_BYTES);
}

#[test]
fn spares_unused_past_the_age_bound_are_trimmed_oldest_first() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite(), false, 0));
    versions.retire(id(1), "old", 10);
    versions.retire(id(1), "new", 20);
    assert!(reclaim(&mut versions, 20).is_empty());
    assert!(versions.trim(10 + SPARE_MAX_AGE_SUBMITS).is_empty());
    assert_eq!(versions.trim(11 + SPARE_MAX_AGE_SUBMITS), vec!["old"]);
    assert_eq!(versions.parked_bytes(), SPRITE_BYTES);
    assert_eq!(versions.trim(21 + SPARE_MAX_AGE_SUBMITS), vec!["new"]);
    assert_eq!(versions.parked_bytes(), 0);
}

#[test]
fn a_drain_takes_every_spare_and_retiring_version() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite(), false, 0));
    drop(versions.take_spare(id(2), sprite(), false, 0));
    versions.retire(id(1), "a", 1);
    versions.retire(id(2), "b", 1);
    assert!(reclaim(&mut versions, 1).is_empty());
    versions.retire(id(1), "c", 2);
    let mut drained = versions.drain_versions();
    drained.sort_unstable();
    assert_eq!(drained, vec!["a", "b", "c"]);
    assert_eq!(versions.parked_bytes(), 0);
    assert!(versions.pop_retired(u64::MAX).is_none());
    // The textures stay tracked: a later version parks again.
    versions.retire(id(1), "d", 3);
    assert!(reclaim(&mut versions, 3).is_empty());
    assert_eq!(versions.parked_bytes(), SPRITE_BYTES);
}

// ── Write history and the copy plan ──

/// Texture 1 of `shape`, tracked with a write history from its first rename.
fn tracked(shape: fn() -> VersionShape) -> TextureVersions<&'static str> {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), shape(), true, 0));
    versions
}

/// Retire the live version at `seq`, settle it under `failed` and take it back as a spare.
///
/// Returns the write count the spare holds, `None` for unknown contents.
fn cycle(
    versions: &mut TextureVersions<&'static str>,
    shape: fn() -> VersionShape,
    seq: u64,
    failed: u64,
) -> Option<u64> {
    versions.retire(id(1), "v", seq);
    let entry = versions.pop_retired(seq).expect("settled");
    assert!(versions.park(entry, failed).is_none());
    let (spare, _) = versions.take_spare(id(1), shape(), true, 0);
    spare.expect("the spare just parked").into_parts().1
}

/// An upload footprint of `rect` into level 0 of slice 0 that does not cover it.
fn partial() -> UploadFootprint {
    UploadFootprint::new(0, 0, false)
}

/// The plan as `(slice, level, region)` triples, `None` for a whole level.
fn plan_of(
    versions: &TextureVersions<&'static str>,
    shape: fn() -> VersionShape,
    held: Option<u64>,
    upload: &UploadFootprint,
) -> Vec<(u32, u32, Option<[u32; 4]>)> {
    versions
        .preserve_plan(id(1), &shape(), held, upload)
        .iter()
        .map(|copy| {
            let region = copy.region().map(|r| [r.x(), r.y(), r.width(), r.height()]);
            (copy.slice(), copy.level(), region)
        })
        .collect()
}

fn cube_chain() -> VersionShape {
    let mut d = desc(128, 128, 2, PixelFormat::Bgra8Unorm);
    d.flags = TextureCreateFlags::TYPE_CUBE;
    VersionShape::of(&d)
}

fn dxt_odd() -> VersionShape {
    VersionShape::of(&desc(100, 60, 1, PixelFormat::Bc1Rgba))
}

#[test]
fn only_textures_whose_writers_are_all_uploads_keep_a_history() {
    let plain = desc(64, 64, 1, PixelFormat::Bgra8Unorm);
    assert!(records_every_write(&plain, TextureUsage::empty()));
    let mut cube = desc(64, 64, 1, PixelFormat::Bgra8Unorm);
    cube.flags = TextureCreateFlags::TYPE_CUBE;
    assert!(records_every_write(&cube, TextureUsage::empty()));
    // The encoder adds the render-target bit for an upload pass; only the
    // application's own usage decides.
    let mut widened = desc(64, 64, 1, PixelFormat::Bgra8Unorm);
    widened.usage_flags = TextureUsage::RENDER_TARGET;
    assert!(records_every_write(&widened, TextureUsage::empty()));
    assert!(!records_every_write(&plain, TextureUsage::RENDER_TARGET));
    assert!(!records_every_write(&plain, TextureUsage::DEPTH_STENCIL));
    let mut volume = desc(64, 64, 1, PixelFormat::Bgra8Unorm);
    volume.depth = 4;
    volume.flags = TextureCreateFlags::TYPE_3D;
    assert!(!records_every_write(&volume, TextureUsage::empty()));
    let depth = desc(64, 64, 1, PixelFormat::Depth32Float);
    assert!(!records_every_write(&depth, TextureUsage::empty()));
}

#[test]
fn a_spare_receives_the_box_of_the_writes_after_it_retired() {
    let mut versions = tracked(sprite);
    // Sprites at (0,0) before the version retires are in it already.
    versions.note_write(id(1), 0, 0, &TexelRect::new(0, 0, 129, 112));
    let held = cycle(&mut versions, sprite, 1, 0);
    assert_eq!(held, Some(1));
    // Two later sprites: their union, rounded out to 16-texel tiles.
    versions.note_write(id(1), 0, 0, &TexelRect::new(0, 0, 10, 20));
    versions.note_write(id(1), 0, 0, &TexelRect::new(0, 0, 45, 36));
    let upload = partial();
    assert_eq!(
        plan_of(&versions, sprite, held, &upload),
        vec![(0, 0, Some([0, 0, 48, 48]))]
    );
    // A spare that held the first of them gets the same box: the second
    // write covers the first's tiles.
    assert_eq!(
        plan_of(&versions, sprite, Some(2), &upload),
        vec![(0, 0, Some([0, 0, 48, 48]))]
    );
    assert!(plan_of(&versions, sprite, Some(3), &upload).is_empty());
}

#[test]
fn scattered_writes_give_their_bounding_box() {
    let mut versions = tracked(sprite);
    versions.note_write(id(1), 0, 0, &TexelRect::new(0, 0, 16, 16));
    versions.note_write(id(1), 0, 0, &TexelRect::new(480, 480, 32, 32));
    assert_eq!(
        plan_of(&versions, sprite, Some(0), &partial()),
        vec![(0, 0, None)],
        "opposite corners span the level, which is copied whole",
    );
    assert_eq!(
        plan_of(&versions, sprite, Some(1), &partial()),
        vec![(0, 0, Some([480, 480, 32, 32]))],
        "the older corner is in the spare already",
    );
}

#[test]
fn nothing_written_since_means_no_copy() {
    let mut versions = tracked(sprite);
    versions.note_write(id(1), 0, 0, &TexelRect::new(0, 0, 64, 64));
    assert!(plan_of(&versions, sprite, Some(1), &partial()).is_empty());
}

#[test]
fn a_box_the_upload_rewrites_is_still_copied() {
    // The upload can decline after the rename, so the new version has to
    // be whole without it: only a full-cover upload skips its level.
    let mut versions = tracked(sprite);
    versions.note_write(id(1), 0, 0, &TexelRect::new(0, 0, 20, 20));
    assert_eq!(
        plan_of(&versions, sprite, Some(0), &partial()),
        vec![(0, 0, Some([0, 0, 32, 32]))],
    );
}

#[test]
fn an_untracked_texture_gets_every_subresource_whole() {
    let versions = TextureVersions::<&str>::new();
    assert_eq!(
        plan_of(
            &versions,
            cube_chain,
            Some(3),
            &UploadFootprint::new(0, 0, true)
        ),
        (0..6)
            .flat_map(|slice| (0..2).map(move |level| (slice, level, None)))
            .filter(|&(slice, level, _)| (slice, level) != (0, 0))
            .collect::<Vec<_>>(),
    );
}

#[test]
fn a_full_cover_upload_skips_its_level_even_without_history() {
    let mut versions = TextureVersions::<&str>::new();
    drop(versions.take_spare(id(1), sprite(), false, 0));
    let full = UploadFootprint::new(0, 0, true);
    assert!(plan_of(&versions, sprite, None, &full).is_empty());
    assert!(plan_of(&versions, sprite, Some(5), &full).is_empty());
    assert_eq!(
        plan_of(&versions, sprite, Some(5), &partial()),
        vec![(0, 0, None)],
        "no history: a known spare is still copied whole",
    );
}

#[test]
fn unknown_or_fresh_contents_copy_every_other_subresource_whole() {
    let mut versions = tracked(cube_chain);
    versions.note_write(id(1), 3, 1, &TexelRect::new(0, 0, 8, 8));
    let upload = UploadFootprint::new(3, 1, true);
    let plan = plan_of(&versions, cube_chain, None, &upload);
    assert_eq!(plan.len(), 11, "12 subresources, the rewritten one skipped");
    assert!(plan.iter().all(|&(_, _, region)| region.is_none()));
    assert!(!plan.contains(&(3, 1, None)));
}

#[test]
fn slices_and_levels_keep_their_own_stamps() {
    let mut versions = tracked(cube_chain);
    versions.note_write(id(1), 3, 1, &TexelRect::new(40, 8, 8, 8));
    versions.note_write(id(1), 0, 0, &TexelRect::new(100, 100, 4, 4));
    let upload = UploadFootprint::new(5, 0, false);
    assert_eq!(
        plan_of(&versions, cube_chain, Some(0), &upload),
        vec![
            (0, 0, Some([96, 96, 16, 16])),
            (3, 1, Some([32, 0, 16, 16]))
        ],
    );
    assert_eq!(
        plan_of(&versions, cube_chain, Some(1), &upload),
        vec![(0, 0, Some([96, 96, 16, 16]))]
    );
}

#[test]
fn mip_regeneration_stamps_every_level_but_the_first_on_every_slice() {
    let mut versions = tracked(cube_chain);
    versions.note_write(id(1), 2, 0, &TexelRect::new(0, 0, 16, 16));
    versions.note_mip_regeneration(id(1));
    let upload = UploadFootprint::new(2, 0, false);
    let plan = plan_of(&versions, cube_chain, Some(1), &upload);
    assert_eq!(
        plan,
        (0..6).map(|slice| (slice, 1, None)).collect::<Vec<_>>()
    );
}

#[test]
fn an_off_grid_compressed_upload_writes_the_whole_level() {
    let dxt1 = crate::format::map_d3d_format(mtld3d_types::D3DFMT_DXT1).expect("DXT1 maps");
    let block = Some(&dxt1);
    let level = (100, 60);
    // On the grid, and short of a block only at the level's edge.
    assert_eq!(
        upload_written_rect(TexelRect::new(64, 32, 36, 28), level, block),
        TexelRect::new(64, 32, 36, 28)
    );
    for off_grid in [
        TexelRect::new(2, 0, 8, 8),
        TexelRect::new(0, 4, 6, 8),
        TexelRect::new(0, 0, 8, 10),
    ] {
        assert_eq!(
            upload_written_rect(off_grid, level, block),
            TexelRect::level(100, 60)
        );
    }
    assert_eq!(
        upload_written_rect(TexelRect::new(3, 5, 7, 9), level, None),
        TexelRect::new(3, 5, 7, 9)
    );
}

#[test]
fn an_odd_compressed_level_clamps_its_last_tiles_to_the_edge() {
    let mut versions = tracked(dxt_odd);
    // 100x60 DXT1: 16-texel tiles, 7 columns and 4 rows, the last ones
    // clamped at the level's edge, which a block copy accepts.
    versions.note_write(id(1), 0, 0, &TexelRect::new(96, 56, 4, 4));
    versions.note_write(id(1), 0, 0, &TexelRect::new(64, 32, 4, 4));
    assert_eq!(
        plan_of(&versions, dxt_odd, Some(0), &partial()),
        vec![(0, 0, Some([64, 32, 36, 28]))]
    );
}

#[test]
fn a_tile_is_a_whole_number_of_blocks_and_a_grid_has_at_most_32_a_side() {
    assert_eq!(tile_edge(512, 1), 16);
    assert_eq!(tile_edge(4096, 1), 128);
    assert_eq!(tile_edge(16384, 4), 512);
    assert_eq!(tile_edge(3, 1), 16);
    assert_eq!(tile_edge(1000, 4), 32);
    assert_eq!(tile_edge(1100, 4), 36);
    for (extent, block) in [(1, 1), (100, 4), (1100, 4), (4097, 1), (16384, 4)] {
        let tile = tile_edge(extent, block);
        assert!(tile.is_multiple_of(block));
        assert!(tile >= MIN_TILE_TEXELS);
        assert!(extent.div_ceil(tile) <= MAX_TILES_PER_AXIS);
    }
}

#[test]
fn a_version_live_through_an_aborted_submission_parks_with_unknown_contents() {
    let mut versions = tracked(sprite);
    versions.note_write(id(1), 0, 0, &TexelRect::new(0, 0, 8, 8));
    // The first version was live when tracking started, so any failure at
    // all leaves it unknown.
    assert_eq!(cycle(&mut versions, sprite, 4, 2), None);
    // The version taken at seq 4 is live from 4 on: a failure at 3 is
    // before its time, one at 5 is not.
    assert_eq!(cycle(&mut versions, sprite, 6, 3), Some(1));
    assert_eq!(cycle(&mut versions, sprite, 8, 6), None);
    assert_eq!(cycle(&mut versions, sprite, 9, 0), Some(1));
}

#[test]
fn writes_to_untracked_textures_and_past_the_level_are_ignored() {
    let mut versions = tracked(sprite);
    versions.note_write(id(7), 0, 0, &TexelRect::new(0, 0, 8, 8));
    versions.note_write(id(1), 0, 0, &TexelRect::new(600, 600, 8, 8));
    assert!(!versions.records_writes(id(7)));
    assert!(versions.records_writes(id(1)));
    assert!(plan_of(&versions, sprite, Some(0), &partial()).is_empty());
}

#[test]
fn a_write_to_a_subresource_the_texture_lacks_drops_its_history() {
    for (slice, level) in [(1, 0), (0, 1)] {
        let mut versions = tracked(sprite);
        versions.note_write(id(1), slice, level, &TexelRect::new(0, 0, 8, 8));
        assert!(!versions.records_writes(id(1)));
        // A known spare is copied whole from here on.
        assert_eq!(
            plan_of(&versions, sprite, Some(0), &partial()),
            vec![(0, 0, None)]
        );
    }
}

// ── Bounding the tracked set ──

#[test]
fn trim_visits_only_textures_holding_spares() {
    let mut versions = TextureVersions::new();
    for raw in 1..=3 {
        drop(versions.take_spare(id(raw), sprite(), false, 1));
        versions.retire(id(raw), "v", 1);
    }
    assert!(reclaim(&mut versions, 1).is_empty());
    assert_eq!(versions.holding_spares(), 3);
    // Texture 2 takes its only spare back, so it holds none.
    let (spare, _) = versions.take_spare(id(2), sprite(), false, 2);
    assert!(spare.is_some());
    assert_eq!(versions.holding_spares(), 2);
    // Trimming the other two empties them as well.
    assert_eq!(versions.trim(2 + SPARE_MAX_AGE_SUBMITS).len(), 2);
    assert_eq!(versions.holding_spares(), 0);
    assert_eq!(versions.parked_bytes(), 0);
    // Forgetting and draining leave nothing behind either.
    versions.retire(id(2), "w", 3);
    assert!(reclaim(&mut versions, 3).is_empty());
    assert_eq!(versions.holding_spares(), 1);
    assert_eq!(versions.forget_texture(id(2)), vec!["w"]);
    assert_eq!(versions.holding_spares(), 0);
}

#[test]
fn a_texture_with_no_spares_and_no_rename_for_the_age_bound_stops_being_tracked() {
    let mut versions = TextureVersions::<&str>::new();
    drop(versions.take_spare(id(1), sprite(), true, 10));
    drop(versions.take_spare(id(2), sprite(), true, 10));
    // Texture 2 is renamed again later, so it stays.
    drop(versions.take_spare(id(2), sprite(), true, 50));
    assert!(versions.trim(10 + SPARE_MAX_AGE_SUBMITS).is_empty());
    assert_eq!(versions.tracked(), 2);
    assert!(versions.trim(11 + SPARE_MAX_AGE_SUBMITS).is_empty());
    assert_eq!(versions.tracked(), 1);
    assert!(!versions.records_writes(id(1)));
    assert!(versions.records_writes(id(2)));
    assert!(versions.trim(51 + SPARE_MAX_AGE_SUBMITS).is_empty());
    assert_eq!(versions.tracked(), 0);
}

#[test]
fn a_texture_is_dropped_once_its_last_spare_is_trimmed() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite(), true, 5));
    versions.retire(id(1), "spare", 5);
    assert!(reclaim(&mut versions, 5).is_empty());
    // The spare keeps it while it is young; one trim takes both.
    assert!(versions.trim(5 + SPARE_MAX_AGE_SUBMITS).is_empty());
    assert_eq!(versions.tracked(), 1);
    assert_eq!(versions.trim(6 + SPARE_MAX_AGE_SUBMITS), vec!["spare"]);
    assert_eq!(versions.tracked(), 0);
    // A version still retiring when the state goes is destroyed, not parked.
    drop(versions.take_spare(id(1), sprite(), true, 200));
    versions.retire(id(1), "late", 200);
    assert!(versions.trim(321).is_empty());
    assert_eq!(versions.tracked(), 0);
    assert_eq!(reclaim(&mut versions, 200), vec!["late"]);
}

#[test]
fn a_state_back_under_its_id_keeps_its_own_age() {
    let mut versions = TextureVersions::<&str>::new();
    drop(versions.take_spare(id(1), sprite(), true, 10));
    assert!(versions.forget_texture(id(1)).is_empty());
    drop(versions.take_spare(id(1), sprite(), true, 100));
    // The first state's entry comes due first and must not drop the second.
    assert!(versions.trim(11 + SPARE_MAX_AGE_SUBMITS).is_empty());
    assert_eq!(versions.tracked(), 1);
    assert!(versions.trim(101 + SPARE_MAX_AGE_SUBMITS).is_empty());
    assert_eq!(versions.tracked(), 0);
}

#[test]
fn a_texture_renamed_early_is_dropped_on_time_behind_a_younger_one() {
    let mut versions = TextureVersions::<&str>::new();
    drop(versions.take_spare(id(1), sprite(), true, 0));
    drop(versions.take_spare(id(1), sprite(), true, 10));
    drop(versions.take_spare(id(2), sprite(), true, 50));
    // Texture 1's entry comes due at 121 and goes back under its rename at
    // 10, which is due before texture 2's entry at 50.
    assert!(versions.trim(121).is_empty());
    assert_eq!(versions.tracked(), 2);
    assert!(versions.trim(10 + SPARE_MAX_AGE_SUBMITS + 1).is_empty());
    assert!(!versions.records_writes(id(1)));
    assert!(versions.records_writes(id(2)));
}
