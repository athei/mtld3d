//! Unit tests for the rename pool behind the encoder's sampled-texture renames.
//!
//! The propositions pinned here: a version is handed back only after its
//! submission has settled, spares are taken newest first and only by their
//! own texture, the parked bytes follow every park, take, trim and drain,
//! the cap and the age bound give storage back, and a version whose texture
//! is gone or changed shape is destroyed rather than parked.

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
        destroyed.extend(versions.park(entry));
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
    let (spare, discarded) = versions.take_spare(id(1), sprite());
    assert!(spare.is_none());
    assert!(discarded.is_empty());
}

#[test]
fn a_version_is_not_reclaimed_before_its_submission_settles() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite()));
    versions.retire(id(1), "v1", 5);
    assert!(versions.pop_retired(4).is_none());
    let entry = versions.pop_retired(5).expect("seq 5 settled");
    assert_eq!(*entry.views(), "v1");
    assert!(versions.park(entry).is_none());
    assert_eq!(versions.parked_bytes(), SPRITE_BYTES);
}

#[test]
fn the_retiring_queue_stops_at_the_first_unsettled_submission() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite()));
    versions.retire(id(1), "v1", 5);
    versions.retire(id(1), "v2", 6);
    versions.retire(id(1), "v3", 7);
    assert!(reclaim(&mut versions, 6).is_empty());
    assert_eq!(versions.parked_bytes(), 2 * SPRITE_BYTES);
    // "v3" is still in flight: the next rename gets "v2", then "v1".
    let (first, _) = versions.take_spare(id(1), sprite());
    let (second, _) = versions.take_spare(id(1), sprite());
    let (third, _) = versions.take_spare(id(1), sprite());
    assert_eq!(first.map(Spare::into_views), Some("v2"));
    assert_eq!(second.map(Spare::into_views), Some("v1"));
    assert!(third.is_none());
    assert_eq!(versions.parked_bytes(), 0);
}

#[test]
fn spares_serve_only_their_own_texture() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite()));
    drop(versions.take_spare(id(2), sprite()));
    versions.retire(id(1), "a1", 3);
    assert!(reclaim(&mut versions, 3).is_empty());
    let (other, _) = versions.take_spare(id(2), sprite());
    assert!(other.is_none());
    let (own, _) = versions.take_spare(id(1), sprite());
    assert_eq!(own.map(Spare::into_views), Some("a1"));
}

#[test]
fn a_version_of_a_released_texture_is_destroyed_when_it_settles() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite()));
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
    drop(versions.take_spare(id(1), sprite()));
    versions.retire(id(1), "before", 2);
    // Evicted while that version is in flight, then renamed again under the
    // same id before the drain pops it.
    assert!(versions.forget_texture(id(1)).is_empty());
    let (spare, _) = versions.take_spare(id(1), sprite());
    assert!(spare.is_none());
    versions.retire(id(1), "after", 2);
    assert_eq!(reclaim(&mut versions, 2), vec!["before"]);
    assert_eq!(versions.parked_bytes(), SPRITE_BYTES);
    let (spare, _) = versions.take_spare(id(1), sprite());
    assert_eq!(spare.map(Spare::into_views), Some("after"));
}

#[test]
fn a_shape_change_discards_the_spares_of_the_old_shape() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite()));
    versions.retire(id(1), "old parked", 2);
    assert!(reclaim(&mut versions, 2).is_empty());
    versions.retire(id(1), "old retiring", 3);
    let bigger = VersionShape::of(&desc(1024, 512, 1, PixelFormat::Abgr4Unorm));
    let (spare, discarded) = versions.take_spare(id(1), bigger);
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
    drop(versions.take_spare(id(1), big()));
    for (seq, label) in (1..).zip(["a", "b", "c", "d", "e"]) {
        versions.retire(id(1), label, seq);
    }
    assert_eq!(reclaim(&mut versions, 5), vec!["e"]);
    assert_eq!(versions.parked_bytes(), VERSION_POOL_CAP_BYTES);
    // Taking one makes room for the next.
    let (spare, _) = versions.take_spare(id(1), big());
    assert_eq!(spare.map(Spare::into_views), Some("d"));
    versions.retire(id(1), "f", 6);
    assert!(reclaim(&mut versions, 6).is_empty());
    assert_eq!(versions.parked_bytes(), VERSION_POOL_CAP_BYTES);
}

#[test]
fn spares_unused_past_the_age_bound_are_trimmed_oldest_first() {
    let mut versions = TextureVersions::new();
    drop(versions.take_spare(id(1), sprite()));
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
    drop(versions.take_spare(id(1), sprite()));
    drop(versions.take_spare(id(2), sprite()));
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
