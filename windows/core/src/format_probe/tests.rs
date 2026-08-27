//! Unit tests for the `CheckDeviceFormat` dedup keys.
//!
//! Pins the property the log lines depend on: two queries that differ in any
//! one field get two keys. The named cases are the query pairs a shift-and-or
//! packing of the four fields aliases, where the second query's line is
//! swallowed and the format family it asks about reads as rejected.

use mtld3d_types::{
    D3DFMT_A8R8G8B8, D3DFMT_INTZ, D3DFMT_X8R8G8B8, D3DRTYPE_CUBETEXTURE, D3DRTYPE_TEXTURE,
    D3DUSAGE_QUERY_FILTER, D3DUSAGE_RENDERTARGET,
};
use rustc_hash::FxHashSet;

use super::*;

#[test]
fn query_filter_is_distinct_from_no_usage() {
    let filter = FormatProbeKey::from_query(
        D3DFMT_X8R8G8B8,
        D3DUSAGE_QUERY_FILTER,
        D3DRTYPE_TEXTURE,
        D3DFMT_A8R8G8B8,
    );
    let plain = FormatProbeKey::from_query(D3DFMT_X8R8G8B8, 0, D3DRTYPE_TEXTURE, D3DFMT_A8R8G8B8);
    assert_ne!(filter.raw(), plain.raw());
}

#[test]
fn query_filter_is_distinct_from_render_target() {
    let filter = FormatProbeKey::from_query(
        D3DFMT_X8R8G8B8,
        D3DUSAGE_QUERY_FILTER | D3DUSAGE_RENDERTARGET,
        D3DRTYPE_TEXTURE,
        D3DFMT_A8R8G8B8,
    );
    let plain = FormatProbeKey::from_query(
        D3DFMT_X8R8G8B8,
        D3DUSAGE_RENDERTARGET,
        D3DRTYPE_TEXTURE,
        D3DFMT_A8R8G8B8,
    );
    assert_ne!(filter.raw(), plain.raw());
}

#[test]
fn same_query_gives_same_key() {
    let first = FormatProbeKey::from_query(
        D3DFMT_X8R8G8B8,
        D3DUSAGE_QUERY_FILTER,
        D3DRTYPE_TEXTURE,
        D3DFMT_A8R8G8B8,
    );
    let second = FormatProbeKey::from_query(
        D3DFMT_X8R8G8B8,
        D3DUSAGE_QUERY_FILTER,
        D3DRTYPE_TEXTURE,
        D3DFMT_A8R8G8B8,
    );
    assert_eq!(first.raw(), second.raw());
    assert_eq!(
        FormatProbeKey::from_probe(D3DUSAGE_QUERY_FILTER, D3DRTYPE_TEXTURE, D3DFMT_A8R8G8B8).raw(),
        FormatProbeKey::from_probe(D3DUSAGE_QUERY_FILTER, D3DRTYPE_TEXTURE, D3DFMT_A8R8G8B8).raw()
    );
}

#[test]
fn every_field_moves_the_key() {
    let base = FormatProbeKey::from_query(
        D3DFMT_X8R8G8B8,
        D3DUSAGE_QUERY_FILTER,
        D3DRTYPE_TEXTURE,
        D3DFMT_A8R8G8B8,
    )
    .raw();
    let others = [
        FormatProbeKey::from_query(
            D3DFMT_A8R8G8B8,
            D3DUSAGE_QUERY_FILTER,
            D3DRTYPE_TEXTURE,
            D3DFMT_A8R8G8B8,
        ),
        FormatProbeKey::from_query(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_RENDERTARGET,
            D3DRTYPE_TEXTURE,
            D3DFMT_A8R8G8B8,
        ),
        FormatProbeKey::from_query(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_QUERY_FILTER,
            D3DRTYPE_CUBETEXTURE,
            D3DFMT_A8R8G8B8,
        ),
        FormatProbeKey::from_query(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_QUERY_FILTER,
            D3DRTYPE_TEXTURE,
            D3DFMT_X8R8G8B8,
        ),
    ];
    for other in others {
        assert_ne!(base, other.raw());
    }
}

#[test]
fn the_whole_query_space_stays_distinct() {
    // Every combination a title can ask about, across the display formats we
    // accept as an adapter format, the usage bits that reach this call, both
    // resource types with distinct answers, and a fourcc alongside the plain
    // formats. Fourccs are what make a bit-packing impossible: they occupy the
    // full 32 bits on their own.
    let adapter_formats = [D3DFMT_X8R8G8B8, D3DFMT_A8R8G8B8];
    let usages = [
        0,
        D3DUSAGE_RENDERTARGET,
        D3DUSAGE_QUERY_FILTER,
        D3DUSAGE_QUERY_FILTER | D3DUSAGE_RENDERTARGET,
    ];
    let rtypes = [D3DRTYPE_TEXTURE, D3DRTYPE_CUBETEXTURE];
    let check_formats = [D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8, D3DFMT_INTZ];
    let mut seen = FxHashSet::default();
    let mut count = 0_usize;
    for adapter_format in adapter_formats {
        for usage in usages {
            for rtype in rtypes {
                for check_format in check_formats {
                    count += 1;
                    let key =
                        FormatProbeKey::from_query(adapter_format, usage, rtype, check_format);
                    assert!(seen.insert(key.raw()), "collision at usage {usage:#x}");
                }
            }
        }
    }
    assert_eq!(count, seen.len());
    assert_eq!(count, 48);
}
