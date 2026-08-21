//! Unit tests for the `D3DSTATEBLOCKTYPE` state-block filter.
//!
//! `StateBlockType` decides which states a filtered block captures and writes
//! back, so the membership predicates are pinned by exact set sizes (45 vertex
//! and 60 pixel render states, plus sampler and texture-stage counts) and by
//! spot checks on states that belong to both sets or to only one. A predicate
//! that drifts shows up here, not as an `Apply` clobbering the other pipeline.

use super::{
    D3DSBT_ALL, D3DSBT_PIXELSTATE, D3DSBT_VERTEXSTATE, RENDER_STATE_COUNT, SAMPLER_STATE_COUNT,
    StateBlockType,
};
use crate::TEXTURE_STAGE_STATE_COUNT;

fn bound(n: usize) -> u32 {
    u32::try_from(n).expect("state count fits u32")
}
fn count_render(ty: StateBlockType) -> usize {
    (0..bound(RENDER_STATE_COUNT))
        .filter(|&i| ty.includes_render_state(i))
        .count()
}
fn count_sampler(ty: StateBlockType) -> usize {
    (0..bound(SAMPLER_STATE_COUNT))
        .filter(|&i| ty.includes_sampler_state(i))
        .count()
}
fn count_tss(ty: StateBlockType) -> usize {
    (0..bound(TEXTURE_STAGE_STATE_COUNT))
        .filter(|&i| ty.includes_tss(i))
        .count()
}

#[test]
fn from_d3dsbt_maps_known_types() {
    assert_eq!(
        StateBlockType::from_d3dsbt(D3DSBT_ALL),
        Some(StateBlockType::All)
    );
    assert_eq!(
        StateBlockType::from_d3dsbt(D3DSBT_VERTEXSTATE),
        Some(StateBlockType::Vertex)
    );
    assert_eq!(
        StateBlockType::from_d3dsbt(D3DSBT_PIXELSTATE),
        Some(StateBlockType::Pixel)
    );
    assert_eq!(StateBlockType::from_d3dsbt(0), None);
    assert_eq!(StateBlockType::from_d3dsbt(99), None);
}

/// Membership counts must match the D3D9 state-block classification.
///
/// `D3DSBT_VERTEXSTATE` / `D3DSBT_PIXELSTATE`: 45 vertex render states and
/// 60 pixel render states.
#[test]
fn classification_set_sizes_match_d3d9() {
    assert_eq!(
        count_render(StateBlockType::Vertex),
        45,
        "vertex render states"
    );
    assert_eq!(
        count_render(StateBlockType::Pixel),
        60,
        "pixel render states"
    );
    assert_eq!(
        count_sampler(StateBlockType::Vertex),
        1,
        "vertex sampler states"
    );
    assert_eq!(
        count_sampler(StateBlockType::Pixel),
        12,
        "pixel sampler states"
    );
    assert_eq!(
        count_tss(StateBlockType::Vertex),
        2,
        "vertex texture-stage states"
    );
    assert_eq!(
        count_tss(StateBlockType::Pixel),
        17,
        "pixel texture-stage states"
    );
}

#[test]
fn all_includes_everything() {
    for i in 0..bound(RENDER_STATE_COUNT) {
        assert!(StateBlockType::All.includes_render_state(i));
    }
    for i in 0..bound(SAMPLER_STATE_COUNT) {
        assert!(StateBlockType::All.includes_sampler_state(i));
    }
    for i in 0..bound(TEXTURE_STAGE_STATE_COUNT) {
        assert!(StateBlockType::All.includes_tss(i));
    }
}

#[test]
fn shademode_and_fog_scalars_are_in_both_render_sets() {
    for rs in [
        super::D3DRS_SHADEMODE,
        super::D3DRS_FOGSTART,
        super::D3DRS_FOGEND,
        super::D3DRS_FOGDENSITY,
    ] {
        assert!(StateBlockType::Vertex.includes_render_state(rs));
        assert!(StateBlockType::Pixel.includes_render_state(rs));
    }
    // A pixel-only / vertex-only spot check.
    assert!(StateBlockType::Pixel.includes_render_state(super::D3DRS_ALPHABLENDENABLE));
    assert!(!StateBlockType::Vertex.includes_render_state(super::D3DRS_ALPHABLENDENABLE));
    assert!(StateBlockType::Vertex.includes_render_state(super::D3DRS_LIGHTING));
    assert!(!StateBlockType::Pixel.includes_render_state(super::D3DRS_LIGHTING));
    // All 16 texture-wrap states are pixel state.
    for rs in super::D3DRS_WRAP0..=super::D3DRS_WRAP7 {
        assert!(StateBlockType::Pixel.includes_render_state(rs));
    }
    for rs in super::D3DRS_WRAP8..=super::D3DRS_WRAP15 {
        assert!(StateBlockType::Pixel.includes_render_state(rs));
    }
}
