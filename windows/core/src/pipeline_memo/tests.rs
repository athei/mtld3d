use mtld3d_shared::{
    MetalHandle,
    mtl::{PixelFormat, VertexStepFunction},
};
use mtld3d_types::MAX_STREAMS;

use super::*;
use crate::pipeline_state::{
    ExtraColorAttachments, PipelineAttachFlags, PipelineRsBits, PipelineRsFlags, StreamLayout,
};

/// A distinct snapshot per `id`: its shader functions, declaration and blend factor differ.
fn snapshot(id: u64) -> PipelineSnapshot {
    let mut layouts = [StreamLayout::UNUSED; MAX_STREAMS as usize];
    layouts[0] = StreamLayout {
        stride: 32,
        step: VertexStepFunction::PerVertex,
        step_rate: 1,
    };
    PipelineSnapshot {
        // SAFETY: tests; opaque values never dereferenced.
        vs_fn: unsafe { MetalHandle::new(0x1000 + id) },
        // SAFETY: tests; opaque values never dereferenced.
        ps_fn: unsafe { MetalHandle::new(0x2000 + id) },
        vdecl_hash: 0x3000,
        stream_layouts: layouts,
        color_format: PixelFormat::Bgra8Unorm,
        attach: PipelineAttachFlags::HAS_DEPTH | PipelineAttachFlags::HAS_COLOR_OUTPUT,
        rs: PipelineRsBits {
            flags: PipelineRsFlags::BLEND_ENABLE,
            src_blend: 5,
            dst_blend: 6,
            color_write_mask: 0xF,
            color_write_mask_ext: [0xF; 3],
            ..PipelineRsBits::default()
        },
        extra: ExtraColorAttachments::NONE,
        ps_color_out_mask: 1,
        sample_count: 1,
    }
}

const fn handle(id: u64) -> u64 {
    0x9000 + id
}

#[test]
fn an_empty_memo_answers_nothing() {
    let mut memo = PipelineMemo::default();
    assert!(memo.is_empty());
    assert_eq!(memo.lookup(&snapshot(0)), None);
}

#[test]
fn every_recorded_snapshot_hits_while_the_memo_holds_them_all() {
    let mut memo = PipelineMemo::default();
    let count = u64::try_from(PIPELINE_MEMO_ENTRIES).unwrap();
    for id in 0..count {
        assert_eq!(memo.lookup(&snapshot(id)), None, "{id} misses first");
        memo.record(&snapshot(id), handle(id));
    }
    assert_eq!(memo.len(), PIPELINE_MEMO_ENTRIES);
    // In order, in reverse and repeated: each answers with its own pipeline.
    for id in (0..count).chain((0..count).rev()).chain([3, 3, 7, 3]) {
        assert_eq!(memo.lookup(&snapshot(id)), Some(handle(id)), "{id} hits");
    }
    assert_eq!(
        memo.lookup(&snapshot(count)),
        None,
        "an unrecorded snapshot misses"
    );
}

#[test]
fn a_snapshot_differing_in_any_field_but_the_tag_ones_is_a_miss() {
    let mut memo = PipelineMemo::default();
    memo.record(&snapshot(1), handle(1));
    // Same shaders and declaration, so the same tag: stale blend factors, a
    // stride and a target format each name another pipeline.
    let mut factors = snapshot(1);
    factors.rs.dst_blend = 2;
    let mut stride = snapshot(1);
    stride.stream_layouts[0].stride = 24;
    let mut format = snapshot(1);
    format.color_format = PixelFormat::Rgba8Unorm;
    memo.record(&snapshot(2), handle(2));
    for other in [&factors, &stride, &format] {
        assert_eq!(memo.lookup(other), None);
    }
    memo.record(&factors, handle(3));
    assert_eq!(memo.lookup(&snapshot(1)), Some(handle(1)));
    assert_eq!(memo.lookup(&factors), Some(handle(3)));
    assert_eq!(memo.lookup(&stride), None);
}

#[test]
fn the_least_recently_used_entry_is_replaced() {
    let mut memo = PipelineMemo::default();
    let count = u64::try_from(PIPELINE_MEMO_ENTRIES).unwrap();
    for id in 0..count {
        memo.record(&snapshot(id), handle(id));
    }
    // Use every entry but 5, so 5 is the least recently used.
    for id in (0..count).filter(|&id| id != 5) {
        assert_eq!(memo.lookup(&snapshot(id)), Some(handle(id)));
    }
    memo.record(&snapshot(100), handle(100));
    assert_eq!(memo.len(), PIPELINE_MEMO_ENTRIES);
    assert_eq!(
        memo.lookup(&snapshot(5)),
        None,
        "the least recent entry went"
    );
    assert_eq!(memo.lookup(&snapshot(100)), Some(handle(100)));
    for id in (0..count).filter(|&id| id != 5) {
        assert_eq!(memo.lookup(&snapshot(id)), Some(handle(id)), "{id} stays");
    }
    // The next insert replaces the entry least recently used now: 100, looked up
    // before the loop above touched every other one.
    memo.record(&snapshot(101), handle(101));
    assert_eq!(memo.lookup(&snapshot(100)), None);
}

#[test]
fn a_working_set_larger_than_the_memo_still_hits_its_recent_part() {
    let mut memo = PipelineMemo::default();
    let count = u64::try_from(PIPELINE_MEMO_ENTRIES).unwrap();
    let mut hits = 0;
    for round in 0..3 {
        for id in 0..count + 4 {
            // Each snapshot twice in a row, as runs of draws are.
            for _ in 0..2 {
                match memo.lookup(&snapshot(id)) {
                    Some(found) => {
                        assert_eq!(found, handle(id));
                        hits += 1;
                    }
                    None => memo.record(&snapshot(id), handle(id)),
                }
            }
        }
        assert!(
            memo.len() <= PIPELINE_MEMO_ENTRIES,
            "round {round} stays bounded"
        );
    }
    // The repeat of each run always hits; a cyclic set larger than the memo misses otherwise.
    assert_eq!(hits, 3 * (count + 4));
}

#[test]
fn a_draw_and_its_colourless_variant_keep_their_own_entries() {
    // A pass that drops render target 0 resolves the colourless snapshot of
    // the same draw through the memo too; the queued no-colour sibling of a
    // colour pipeline is never recorded. Alternating them evicts neither.
    let mut memo = PipelineMemo::default();
    let colour = snapshot(1);
    let mut colourless = snapshot(1);
    colourless.remove_color_output();
    assert!(memo.lookup(&colour).is_none());
    memo.record(&colour, handle(1));
    assert!(memo.lookup(&colourless).is_none());
    memo.record(&colourless, handle(2));
    for _ in 0..4 {
        assert_eq!(memo.lookup(&colour), Some(handle(1)));
        assert_eq!(memo.lookup(&colourless), Some(handle(2)));
    }
    assert_eq!(memo.len(), 2);
}

#[test]
fn a_pending_or_failed_snapshot_is_never_answered() {
    // The encoder records only built pipelines, so a snapshot whose build is
    // pending or failed is looked up every time and never becomes an answer.
    let mut memo = PipelineMemo::default();
    let pending = snapshot(7);
    for _ in 0..3 {
        assert_eq!(memo.lookup(&pending), None);
    }
    memo.record(&snapshot(8), handle(8));
    assert_eq!(memo.lookup(&pending), None);
    // Once it builds and is recorded, it answers.
    memo.record(&pending, handle(7));
    assert_eq!(memo.lookup(&pending), Some(handle(7)));
}
