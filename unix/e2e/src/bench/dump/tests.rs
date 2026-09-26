//! Unit tests for the frame-dump reader and the benchmark calibration table.

use super::*;

const SHADOW: &str = "texture TextureId(1) Bgra8Unorm 512x512 slice=0 level=0";
const BACKBUFFER: &str = "backbuffer 800x600";

/// A `draw` event as the dump prints it.
fn draw(seq: u32, rt: &str, ds: &str, vs: &str, ps: &str, tex: &str) -> String {
    format!(
        "draw {seq}: rt={rt} ds={ds}/0x1 vs={vs} ps={ps} z=[1,1,4] blend=[0,5,6,1 sep=0 2,1,1] \
         cull=2 cw=[0xf,0xf,0xf,0xf] alpha=[0,8,0] stencil=[0,8,0x0,0xffffffff,0xffffffff \
         1,1,1 two=0 ccw=8,1,1,1] bias=[0x0,0x0] vp=0,0+512x512 scissor=0 tex=[{tex}]"
    )
}

/// `events` as dump lines of the layer's log, each written `copies` times.
fn log(events: &[String], copies: usize) -> String {
    let mut out = String::from("[2026-09-25T06:03:11Z INFO  mtld3d::d3d9] d3d9.dll v0.11.0\n");
    for (index, event) in events.iter().enumerate() {
        for _ in 0..copies {
            let _ = writeln!(
                out,
                "[2026-09-25T06:03:12Z INFO  mtld3d::d3d9] [dump] {event}"
            );
        }
        if index == 3 {
            let _ = writeln!(out, "[2026-09-25T06:03:12Z WARN  mtld3d::perf] unrelated");
        }
    }
    out
}

/// Two dumped frames and the start of a third; the second is the one read.
fn frames() -> Vec<String> {
    let tex = "s0=TextureId(9)/0x15/64x64";
    vec![
        "frame start (1 of 3)".to_owned(),
        draw(
            0,
            SHADOW,
            "TextureId(2) level=0",
            "ProgramId(5)",
            "ProgramId(6)",
            "",
        ),
        "frame end: 1 draws, command buffer mtld3d-frame-0x1".to_owned(),
        "frame start (2 of 3)".to_owned(),
        "SetRenderTarget(0, TextureId(1)/0x15 l0)".to_owned(),
        "clear flags=0x3 color=0xffffffff z=1 stencil=0 rects=0 rt=texture TextureId(1) \
         Bgra8Unorm 512x512 slice=0 level=0 ds=TextureId(2) level=0"
            .to_owned(),
        draw(
            0,
            SHADOW,
            "TextureId(2) level=0",
            "ProgramId(5)",
            "ProgramId(6)",
            tex,
        ),
        draw(
            1,
            SHADOW,
            "TextureId(2) level=0",
            "ProgramId(5)",
            "ProgramId(6)",
            tex,
        ),
        // A target set and set back before the next draw opens no pass.
        "SetRenderTarget(0, standalone 0x16 800x600)".to_owned(),
        "SetRenderTarget(0, TextureId(1)/0x15 l0)".to_owned(),
        draw(
            2,
            SHADOW,
            "TextureId(2) level=0",
            "ProgramId(5)",
            "ProgramId(6)",
            tex,
        ),
        draw(
            3,
            BACKBUFFER,
            "default",
            "ff",
            "ff",
            &format!("{tex} s1={tex}"),
        ),
        "draw 3 psc: c66=[0.0, 0.0, 0.0, 0.0] c72=[0.0, 0.0, 0.0, 0.0]".to_owned(),
        "StretchRect(src=standalone 0x16 800x600, dst=TextureId(3)/0x15 l0, filter=0)".to_owned(),
        "StretchRect: 1:1 blit queued".to_owned(),
        draw(4, BACKBUFFER, "default", "ff", "ProgramId(7)", ""),
        "frame end: 5 draws, command buffer mtld3d-frame-0x2".to_owned(),
        "frame start (3 of 3)".to_owned(),
        draw(
            0,
            SHADOW,
            "TextureId(2) level=0",
            "ProgramId(5)",
            "ProgramId(6)",
            "",
        ),
    ]
}

fn pass(target: &str, depth: &str, size: Option<Size>, counts: [u32; 4]) -> GamePass {
    let [draws, vertex_ff, pixel_ff, textures] = counts;
    GamePass {
        target: target.to_owned(),
        depth: depth.to_owned(),
        size,
        draws,
        ff_vs: vertex_ff,
        ff_ps: pixel_ff,
        textures,
    }
}

const fn size(width: u32, height: u32) -> Size {
    Size { width, height }
}

fn expected_passes() -> Vec<GamePass> {
    vec![
        pass(
            SHADOW,
            "TextureId(2) level=0",
            Some(size(512, 512)),
            [3, 0, 0, 3],
        ),
        pass(BACKBUFFER, "default", Some(size(800, 600)), [1, 1, 1, 2]),
        pass(BACKBUFFER, "default", Some(size(800, 600)), [1, 1, 0, 0]),
    ]
}

#[test]
fn the_last_complete_frame_splits_where_the_targets_change_or_a_copy_ends_the_pass() {
    let frame = parse_game_log(&log(&frames(), 1)).unwrap();
    assert_eq!(frame.frames, 2);
    assert_eq!(frame.repeats, 0);
    assert_eq!(frame.backbuffer, size(800, 600));
    assert_eq!(frame.passes, expected_passes());
}

#[test]
fn a_log_that_repeats_every_line_is_read_once() {
    let frame = parse_game_log(&log(&frames(), 2)).unwrap();
    assert_eq!(frame.passes, expected_passes());
    assert_eq!(frame.repeats, frames().len());
}

#[test]
fn a_frame_whose_draw_lines_disagree_with_its_end_line_is_an_error() {
    let mut events = frames();
    events.retain(|event| !event.starts_with("draw 4:"));
    let reason = parse_game_log(&log(&events, 1)).unwrap_err();
    assert!(
        reason.contains("frame 2 ends with 5 draws, but 4 draw lines were read"),
        "{reason}"
    );
    let reason = parse_game_log("[2026-09-25T06:03:12Z INFO  mtld3d::d3d9] nothing\n").unwrap_err();
    assert!(reason.contains("no complete [dump] frame"), "{reason}");
}

fn shape_lines(lines: &[&str]) -> Vec<String> {
    lines.iter().map(|line| (*line).to_owned()).collect()
}

/// The benchmark lines that declare the dumped frame, the offscreen pass at the same ratio.
fn matching_bench() -> Vec<String> {
    shape_lines(&[
        "pass 0 1024x1024 draws=3 ff_vs=0 ff_ps=0 tex_per_draw=1.00",
        "pass 1 1600x1200 draws=1 ff_vs=1 ff_ps=1 tex_per_draw=2.00",
        "pass 2 1600x1200 draws=1 ff_vs=1 ff_ps=0 tex_per_draw=0.00",
    ])
}

#[test]
fn shape_lines_parse_and_the_back_buffer_is_the_last_pass_or_its_meta() {
    let bench = parse_bench(&matching_bench(), None).unwrap();
    assert_eq!(bench.passes.len(), 3);
    assert_eq!(bench.backbuffer, size(1600, 1200));
    assert_eq!(bench.backbuffer_from, "the last pass");
    assert_eq!(bench.passes[0].size, size(1024, 1024));
    assert_eq!(bench.passes[1].ff_vs, 1);
    assert_eq!(bench.passes[1].tex_per_draw.to_bits(), 2.0_f64.to_bits());

    let bench = parse_bench(&matching_bench(), Some("1280x720")).unwrap();
    assert_eq!(bench.backbuffer, size(1280, 720));
    assert_eq!(bench.backbuffer_from, "meta backbuffer");
}

#[test]
fn malformed_shape_lines_are_errors() {
    for (lines, expected) in [
        (
            &["pass 1 8x8 draws=1 ff_vs=0 ff_ps=0 tex_per_draw=0"][..],
            "listed from 0 in order",
        ),
        (
            &["pass 0 8x8 draws=1 ff_vs=0 ff_ps=0"][..],
            "no tex_per_draw=",
        ),
        (
            &["pass 0 8x8 draws=1 ff_vs=0 ff_ps=0 tex_per_draw=0 kind=ui"][..],
            "unknown key \"kind\"",
        ),
        (
            &["pass 0 8x0 draws=1 ff_vs=0 ff_ps=0 tex_per_draw=0"][..],
            "bad size",
        ),
        (
            &["pass 0 8x8 draws=many ff_vs=0 ff_ps=0 tex_per_draw=0"][..],
            "draws is not a count",
        ),
        (&[][..], "no shape line"),
    ] {
        let reason = parse_bench(&shape_lines(lines), None).unwrap_err();
        assert!(reason.contains(expected), "{lines:?}: {reason}");
    }
}

#[test]
fn a_bench_that_matches_the_frame_is_within_tolerance() {
    let game = parse_game_log(&log(&frames(), 1)).unwrap();
    let bench = parse_bench(&matching_bench(), None).unwrap();
    let (text, within) = render(&game, &bench, "game g.log", "bench b");
    assert!(within, "{text}");
    assert!(text.contains("WITHIN TOLERANCE"), "{text}");
    assert!(
        text.contains("0.64x0.85"),
        "the offscreen pass is shown relative to its back buffer: {text}"
    );
}

#[test]
fn every_check_out_of_tolerance_is_flagged() {
    let game = parse_game_log(&log(&frames(), 1)).unwrap();
    let bench = parse_bench(
        &shape_lines(&[
            "pass 0 1024x1024 draws=4 ff_vs=1 ff_ps=0 tex_per_draw=2.50",
            "pass 1 1600x1200 draws=1 ff_vs=1 ff_ps=0 tex_per_draw=2.00",
        ]),
        None,
    )
    .unwrap();
    let (text, within) = render(&game, &bench, "game g.log", "bench b");
    assert!(!within);
    let row = |index: &str| {
        text.lines()
            .find(|line| line.starts_with(index))
            .unwrap_or_default()
            .to_owned()
    };
    assert!(row("0/0 ").ends_with("draws,ff_vs,tex"), "{text}");
    assert!(row("1/1 ").ends_with("ff_ps"), "{text}");
    assert!(row("2/- ").ends_with("pass"), "{text}");
    assert!(
        text.contains("OUT OF TOLERANCE: 3 game passes, 2 bench passes; 3 of 3 rows flagged"),
        "{text}"
    );
}

#[test]
fn a_draw_count_within_ten_percent_passes() {
    let game = pass(SHADOW, "d", Some(size(8, 8)), [100, 0, 0, 100]);
    let near = BenchPass {
        size: size(8, 8),
        draws: 110,
        ff_vs: 0,
        ff_ps: 0,
        tex_per_draw: 1.0,
    };
    assert!(flags(Some(&game), Some(&near)).is_empty());
    let far = BenchPass { draws: 111, ..near };
    assert_eq!(flags(Some(&game), Some(&far)), ["draws"]);
}

#[test]
fn a_pass_one_side_lacks_leaves_one_unpaired_row() {
    let game = parse_game_log(&log(&frames(), 1)).unwrap();
    // The benchmark lacks the game's offscreen pass: the two back-buffer
    // passes still pair with the game's and pass.
    let bench = parse_bench(
        &shape_lines(&[
            "pass 0 1600x1200 draws=1 ff_vs=1 ff_ps=1 tex_per_draw=2.00",
            "pass 1 1600x1200 draws=1 ff_vs=1 ff_ps=0 tex_per_draw=0.00",
        ]),
        None,
    )
    .unwrap();
    let (text, within) = render(&game, &bench, "game g.log", "bench b");
    assert!(!within);
    let flagged: Vec<&str> = text
        .lines()
        .filter(|line| line.ends_with("pass") || line.ends_with("draws"))
        .collect();
    assert_eq!(flagged.len(), 1, "{text}");
    assert!(flagged[0].starts_with("0/- "), "{text}");
    assert!(text.contains("1 of 3 rows flagged"), "{text}");
}

#[test]
fn passes_pair_by_kind_and_nearest_size_in_order() {
    let game = GameFrame {
        frames: 1,
        backbuffer: size(1000, 1000),
        passes: vec![
            pass("texture a", "d", Some(size(500, 500)), [1, 0, 0, 0]),
            pass("texture b", "d", Some(size(250, 250)), [1, 0, 0, 0]),
            pass(BACKBUFFER, "d", Some(size(1000, 1000)), [1, 0, 0, 0]),
        ],
        repeats: 0,
    };
    let bench = parse_bench(
        &shape_lines(&[
            "pass 0 128x128 draws=1 ff_vs=0 ff_ps=0 tex_per_draw=0",
            "pass 1 512x512 draws=1 ff_vs=0 ff_ps=0 tex_per_draw=0",
        ]),
        None,
    )
    .unwrap();
    // The benchmark's back buffer is its last pass, 512x512: its 128x128
    // pass is a quarter of it, nearest the game's quarter-size pass.
    assert_eq!(
        pair_passes(&game, &bench),
        [(Some(0), None), (Some(1), Some(0)), (Some(2), Some(1))]
    );
}
