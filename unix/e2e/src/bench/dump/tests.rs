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
            &format!("{tex} s1=TextureId(9)/0x15/64x64"),
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

/// The state mix of `values` in [`STATE_KEYS`] order.
///
/// That is blend, atest, `zwrite_off`, `cull_none`, cmask0, then the switches
/// vs, ps, tex, blend, atest, cull, then the distinct vs, ps, tex.
const fn mix(values: [u32; 14]) -> StateMix {
    StateMix::from_values(values)
}

/// A state mix of nothing but `vs_n`, `ps_n` and `tex_n`.
const fn distinct_only(vs: u32, ps: u32, tex: u32) -> StateMix {
    mix([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, vs, ps, tex])
}

fn pass(
    target: &str,
    depth: &str,
    size: Option<Size>,
    counts: [u32; 4],
    state: StateMix,
) -> GamePass {
    let [draws, vertex_ff, pixel_ff, textures] = counts;
    GamePass {
        target: target.to_owned(),
        depth: depth.to_owned(),
        size,
        draws,
        ff_vs: vertex_ff,
        ff_ps: pixel_ff,
        textures,
        state,
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
            distinct_only(1, 1, 1),
        ),
        // The same texture on two stages is one distinct texture.
        pass(
            BACKBUFFER,
            "default",
            Some(size(800, 600)),
            [1, 1, 1, 2],
            distinct_only(0, 0, 1),
        ),
        pass(
            BACKBUFFER,
            "default",
            Some(size(800, 600)),
            [1, 1, 0, 0],
            distinct_only(0, 1, 0),
        ),
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
    let game = pass(
        SHADOW,
        "d",
        Some(size(8, 8)),
        [100, 0, 0, 100],
        distinct_only(0, 0, 0),
    );
    let near = BenchPass {
        size: size(8, 8),
        draws: 110,
        ff_vs: 0,
        ff_ps: 0,
        tex_per_draw: 1.0,
        state: None,
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
            pass(
                "texture a",
                "d",
                Some(size(500, 500)),
                [1, 0, 0, 0],
                distinct_only(0, 0, 0),
            ),
            pass(
                "texture b",
                "d",
                Some(size(250, 250)),
                [1, 0, 0, 0],
                distinct_only(0, 0, 0),
            ),
            pass(
                BACKBUFFER,
                "d",
                Some(size(1000, 1000)),
                [1, 0, 0, 0],
                distinct_only(0, 0, 0),
            ),
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

/// A draw line of the WoW-14623 dump: fixed function, blending, alpha-tested, no depth writes.
const WOW_FF_DRAW: &str = "draw 0: rt=texture TextureId(4011) Bgra8Unorm 2624x1696 slice=0 \
     level=0 ds=default/0x3 vs=ff ps=ff z=[1,0,4] blend=[1,5,6,1 sep=0 2,1,1] cull=2 \
     cw=[0xf,0xf,0xf,0xf] alpha=[1,7,1] stencil=[0,8,0x0,0xffffffff,0xffffffff 1,1,1 two=0 \
     ccw=8,1,1,1] bias=[0x80000000,0x0] vp=0,0+2624x1696 scissor=0 \
     tex=[s0=TextureId(3343)/0x33545844/128x128]";

#[test]
fn a_draw_line_carrying_every_field_parses() {
    let draw = Draw::parse(WOW_FF_DRAW).unwrap().unwrap();
    assert_eq!(
        draw.target,
        "texture TextureId(4011) Bgra8Unorm 2624x1696 slice=0 level=0"
    );
    assert_eq!(draw.depth, "default");
    assert_eq!((draw.vs, draw.ps), ("ff", "ff"));
    assert_eq!(draw.z_write, 0);
    assert_eq!(draw.blend, "1,5,6,1 sep=0 2,1,1");
    assert_eq!(draw.blend_enable, 1);
    assert_eq!(draw.cull, 2);
    assert_eq!(draw.color_mask, 0xf);
    assert_eq!(draw.alpha, "1,7,1");
    assert_eq!(draw.alpha_enable, 1);
    assert_eq!(draw.stage0_texture(), Some("TextureId(3343)"));
    assert_eq!(draw.texture_ids().collect::<Vec<_>>(), ["TextureId(3343)"]);

    let line = DrawState {
        vs: "ProgramId(11)",
        ps: "ProgramId(12)",
        cull: D3DCULL_NONE,
        cw0: "0x0",
        alpha: "0,7,224",
        tex: "s1=TextureId(8)/0x15/64x64 s3=TextureId(9)/0x15/64x64D vt0=TextureId(7)/0x15/8x8",
        ..PLAIN
    }
    .line();
    let draw = Draw::parse(&line).unwrap().unwrap();
    assert_eq!((draw.vs, draw.ps), ("ProgramId(11)", "ProgramId(12)"));
    assert_eq!(
        (draw.z_write, draw.blend_enable, draw.alpha_enable),
        (1, 0, 0)
    );
    assert_eq!((draw.cull, draw.color_mask), (D3DCULL_NONE, 0));
    assert_eq!(draw.stage0_texture(), None, "stage 0 is empty");
    assert_eq!(
        draw.texture_ids().collect::<Vec<_>>(),
        ["TextureId(8)", "TextureId(9)", "TextureId(7)"]
    );
}

#[test]
fn a_draw_line_lacking_a_field_is_an_error_and_other_events_are_no_draw() {
    for (key, broken) in [
        (" alpha=[", WOW_FF_DRAW.replace(" alpha=[1,7,1]", "")),
        (" cw=[", WOW_FF_DRAW.replace(" cw=[", " colour=[")),
        (" cull=<n>", WOW_FF_DRAW.replace("cull=2", "cull=back")),
        (" tex=[", WOW_FF_DRAW.replace(" tex=[", " textures=[")),
    ] {
        let reason = Draw::parse(&broken).err().unwrap();
        assert!(reason.contains(&format!("no {key}")), "{reason}");
    }
    let reason = Draw::parse(&WOW_FF_DRAW.replace("z=[1,0,4]", "z=[1]"))
        .err()
        .unwrap();
    assert!(reason.contains(" z=[...] has no number at 1"), "{reason}");
    assert!(
        Draw::parse("draw 3 psc: c66=[0.0, 0.0, 0.0, 0.0]")
            .unwrap()
            .is_none()
    );
    assert!(
        Draw::parse("SetRenderTarget(0, TextureId(1)/0x15 l0)")
            .unwrap()
            .is_none()
    );
}

/// The state fields of a synthetic draw line.
struct DrawState<'a> {
    vs: &'a str,
    ps: &'a str,
    z: &'a str,
    blend: &'a str,
    cull: u32,
    /// Render target 0's colour write mask; the other three are `0xf`.
    cw0: &'a str,
    alpha: &'a str,
    tex: &'a str,
}

/// A draw that blends, tests, culls and writes nothing out of the ordinary.
const PLAIN: DrawState<'static> = DrawState {
    vs: "ff",
    ps: "ff",
    z: "1,1,4",
    blend: "0,5,6,1 sep=0 2,1,1",
    cull: 2,
    cw0: "0xf",
    alpha: "0,7,1",
    tex: "",
};

impl DrawState<'_> {
    /// The draw line of this state; the target and depth are fixed.
    fn line(&self) -> String {
        let Self {
            vs,
            ps,
            z,
            blend,
            cull,
            cw0,
            alpha,
            tex,
        } = self;
        format!(
            "draw 0: rt={SHADOW} ds=default/0x3 vs={vs} ps={ps} z=[{z}] blend=[{blend}] \
             cull={cull} cw=[{cw0},0xf,0xf,0xf] alpha=[{alpha}] stencil=[0,8,0x0,0xffffffff,\
             0xffffffff 1,1,1 two=0 ccw=8,1,1,1] bias=[0x0,0x0] vp=0,0+512x512 scissor=0 \
             tex=[{tex}]"
        )
    }
}

#[test]
fn switches_and_distinct_counts_follow_the_draw_order_of_the_pass() {
    let blend_on = "1,5,6,1 sep=0 2,1,1";
    let lines = [
        DrawState {
            vs: "ProgramId(1)",
            ps: "ProgramId(2)",
            blend: blend_on,
            cull: D3DCULL_NONE,
            alpha: "1,7,1",
            tex: "s0=TextureId(1)/0x15/8x8 s1=TextureId(2)/0x15/8x8",
            ..PLAIN
        },
        // Only the alpha reference and the depth write change.
        DrawState {
            vs: "ProgramId(1)",
            blend: blend_on,
            cull: D3DCULL_NONE,
            alpha: "1,7,224",
            z: "1,0,4",
            tex: "s0=TextureId(1)/0x15/8x8",
            ..PLAIN
        },
        // Fixed function, no texture, no colour written.
        DrawState {
            alpha: "0,7,224",
            cw0: "0x0",
            ..PLAIN
        },
        DrawState {
            vs: "ProgramId(3)",
            ps: "ProgramId(2)",
            alpha: "0,7,224",
            tex: "s0=TextureId(3)/0x15/8x8",
            ..PLAIN
        },
        // A texture on a vertex slot counts as distinct, not as stage 0.
        DrawState {
            vs: "ProgramId(1)",
            ps: "ProgramId(2)",
            blend: "1,5,2,1 sep=0 2,1,1",
            cull: D3DCULL_NONE,
            alpha: "0,7,224",
            tex: "s0=TextureId(1)/0x15/8x8 vt0=TextureId(4)/0x15/8x8",
            ..PLAIN
        },
    ]
    .map(|state| state.line());
    let draws: Vec<Draw<'_>> = lines
        .iter()
        .map(|line| Draw::parse(line).unwrap().unwrap())
        .collect();
    assert_eq!(
        StateMix::of(&draws),
        StateMix {
            blend: 3,
            atest: 2,
            zwrite_off: 1,
            cull_none: 3,
            cmask0: 1,
            vs_sw: 3,
            ps_sw: 2,
            tex_sw: 3,
            blend_sw: 2,
            atest_sw: 2,
            cull_sw: 2,
            vs_n: 2,
            ps_n: 1,
            tex_n: 4,
        }
    );
    assert_eq!(
        StateMix::of(&draws[..1]),
        mix([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2])
    );
}

/// A shape line that declares a state mix, the keys in [`STATE_KEYS`] order.
fn with_state(base: &str, values: [u32; 14]) -> String {
    let mut line = base.to_owned();
    for (key, value) in STATE_KEYS.iter().zip(values) {
        let _ = write!(line, " {key}={value}");
    }
    line
}

#[test]
fn shape_lines_with_and_without_the_state_keys_parse() {
    let values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
    let bench = parse_bench(
        &[
            with_state("pass 0 8x8 draws=20 ff_vs=0 ff_ps=0 tex_per_draw=1", values),
            "pass 1 8x8 draws=1 ff_vs=0 ff_ps=0 tex_per_draw=1".to_owned(),
        ],
        None,
    )
    .unwrap();
    assert_eq!(bench.passes[0].state, Some(mix(values)));
    assert_eq!(bench.passes[1].state, None);

    let partial = "pass 0 8x8 draws=1 ff_vs=0 ff_ps=0 tex_per_draw=1 blend=1 vs_sw=2".to_owned();
    let reason = parse_bench(&[partial], None).unwrap_err();
    assert!(
        reason.contains("carries part of the state mix but not atest, zwrite_off"),
        "{reason}"
    );
    let bad = with_state("pass 0 8x8 draws=1 ff_vs=0 ff_ps=0 tex_per_draw=1", values)
        .replace("tex_n=14", "tex_n=-1");
    let reason = parse_bench(&[bad], None).unwrap_err();
    assert!(reason.contains("tex_n is not a count"), "{reason}");
}

#[test]
fn a_bench_without_the_state_keys_is_not_reported_and_not_judged() {
    let mut game = parse_game_log(&log(&frames(), 1)).unwrap();
    // A state mix no benchmark pass matches: unjudged, it flags nothing.
    game.passes[0].state = mix([3, 3, 3, 3, 3, 90, 90, 90, 90, 90, 90, 90, 90, 90]);
    let bench = parse_bench(&matching_bench(), None).unwrap();
    let (text, within) = render(&game, &bench, "game g.log", "bench b");
    assert!(within, "{text}");
    assert!(text.contains("100 / n/r"), "{text}");
    assert!(text.contains("90 / n/r"), "{text}");
    assert!(text.contains("n/r: not reported;"), "{text}");
    assert!(
        text.contains("state mix not reported for 3 bench passes, not judged"),
        "{text}"
    );
}

#[test]
fn a_bench_that_reports_the_state_mix_is_judged_on_it() {
    let game = parse_game_log(&log(&frames(), 1)).unwrap();
    let base = matching_bench();
    let reported = |values: [[u32; 14]; 3]| {
        base.iter()
            .zip(values)
            .map(|(line, values)| with_state(line, values))
            .collect::<Vec<_>>()
    };
    let same = [
        distinct_only(1, 1, 1).values(),
        distinct_only(0, 0, 1).values(),
        distinct_only(0, 1, 0).values(),
    ];
    let bench = parse_bench(&reported(same), None).unwrap();
    let (text, within) = render(&game, &bench, "game g.log", "bench b");
    assert!(within, "{text}");
    assert!(!text.contains("n/r"), "{text}");
    assert!(text.contains("1 / 1"), "{text}");

    // Every draw of the first pass blends in the bench, none in the game,
    // and the bench switches its vertex shader on six of them.
    let mut off = same;
    off[0][0] = 3;
    off[0][5] = 6;
    let bench = parse_bench(&reported(off), None).unwrap();
    let (text, within) = render(&game, &bench, "game g.log", "bench b");
    assert!(!within, "{text}");
    let row = text
        .lines()
        .find(|line| line.starts_with("0/0 "))
        .unwrap_or_default();
    assert!(row.ends_with("blend,vs_sw"), "{text}");
    assert!(text.contains("0 / 100"), "{text}");
    assert!(text.contains("0 / 6"), "{text}");
}

#[test]
fn state_shares_pass_at_ten_points_and_counts_at_fifteen_percent_or_five() {
    let judged = |game_values: [u32; 14], bench_values: [u32; 14]| {
        let game = pass(
            SHADOW,
            "d",
            Some(size(8, 8)),
            [100, 0, 0, 100],
            mix(game_values),
        );
        let bench = BenchPass {
            size: size(8, 8),
            draws: 100,
            ff_vs: 0,
            ff_ps: 0,
            tex_per_draw: 1.0,
            state: Some(mix(bench_values)),
        };
        flags(Some(&game), Some(&bench))
    };
    let with = |slot: usize, value: u32| {
        let mut values = [0; 14];
        values[slot] = value;
        values
    };
    // Shares, of 100 draws on both sides: 10 points pass, 11 do not.
    assert!(judged(with(0, 20), with(0, 30)).is_empty());
    assert!(judged(with(0, 20), with(0, 10)).is_empty());
    assert_eq!(judged(with(0, 20), with(0, 31)), ["blend"]);
    assert_eq!(judged(with(4, 20), with(4, 9)), ["cmask0"]);
    // Counts: 15 % of the game's.
    assert!(judged(with(5, 100), with(5, 115)).is_empty());
    assert!(judged(with(5, 100), with(5, 85)).is_empty());
    assert_eq!(judged(with(5, 100), with(5, 116)), ["vs_sw"]);
    assert_eq!(judged(with(13, 100), with(13, 84)), ["tex_n"]);
    // Counts: never closer than 5, which outweighs 15 % below 34.
    assert!(judged(with(7, 10), with(7, 15)).is_empty());
    assert!(judged(with(7, 0), with(7, 5)).is_empty());
    assert_eq!(judged(with(7, 10), with(7, 16)), ["tex_sw"]);
    assert_eq!(judged(with(7, 0), with(7, 6)), ["tex_sw"]);
}
