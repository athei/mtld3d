//! Unit tests for the pass trace reader and the leg-to-leg shape comparison.

use std::fmt::Write as _;

use super::*;

/// `lines` as the layer logs them on the pass-trace target.
fn trace(lines: &[&str]) -> String {
    lines.iter().fold(String::new(), |mut out, line| {
        let _ = writeln!(out, "[2026-09-26T00:00:00Z TRACE {TRACE_TARGET}] {line}");
        out
    })
}

/// The steady shape of the pass trace in `log`, read the way a log file is.
fn frame_of(log: &str) -> Result<Frame, String> {
    let mut trace = Trace::default();
    for line in log.lines() {
        trace.line(line);
    }
    trace.finish()
}

const OPEN_SHADOW: &str = "pass-open  idx=0 color=0x1000 srgb=0x0 depth=0x2000 size=2048x2048 \
     color_load=Clear { r: 1065353216, g: 1065353216, b: 1065353216, a: 1065353216 } \
     depth_load=Clear { value: 1065353216 } viewport=0,0+2048x2048 extra=0x0";

/// One submission of the steady frame.
fn submission() -> Vec<&'static str> {
    vec![
        "pass-break trigger=set_color_rt prev=0x3000 new=0x1000 slice=0 level=0 new_size=2048x2048",
        OPEN_SHADOW,
        "pass-close idx=0 caller=flush_pending_clears color=0x1000 depth=0x2000 cmds=1 draws=0",
        "pass-open  idx=1 color=0x3000 srgb=0x0 depth=0x4000 size=1280x720 color_load=DontCare \
         depth_load=Clear { value: 1065353216 } viewport=0,0+1280x720 extra=0x0",
        "pass-close idx=1 caller=stretch_rect color=0x3000 depth=0x4000 cmds=223 draws=58",
        "pass-open  idx=2 color=0x5000 srgb=0x0 depth=0x0 size=640x360 color_load=Load \
         depth_load=DontCare viewport=0,0+640x360 extra=0x0",
        "pass-close idx=2 caller=set_color_rt color=0x5000 depth=0x0 cmds=18 draws=1",
        "pass-open  idx=3 color=0x3000 srgb=0x0 depth=0x4000 size=1280x720 color_load=Load \
         depth_load=Load viewport=0,0+1280x720 extra=0x0",
        "pass-close idx=3 caller=submit color=0x3000 depth=0x4000 cmds=70 draws=17",
        "  pass 0: rt=0x1000 2048x2048 color=load depth=0x2000 depth_load=load cmds=1 draws=0",
        "pass-dead-clear drop idx=0 color=0x1000 depth=0x2000:0 (every cleared target \
         overwritten before a read)",
        "pass-load color=0x5000 DontCare → Load (sampled this frame)",
        "pass-store idx=2 depth=0x4000 → DontCare (last-use)",
        "pass-store idx=0 color=0x3000 → DontCare (next-clear at idx=2)",
        "pass-cull dropped=1 dead clear-only passes",
    ]
}

/// A one-pass submission of another shape, such as a warm-up frame.
fn warm_up() -> Vec<&'static str> {
    vec![
        OPEN_SHADOW,
        "pass-close idx=0 caller=submit color=0x1000 depth=0x2000 cmds=1 draws=0",
        "pass-store idx=0 color=0x1000 → DontCare (last-use)",
    ]
}

/// `repeats` copies of `one`, then the start of a submission that may be cut short.
fn steady_of(one: &[&'static str], repeats: usize) -> Vec<&'static str> {
    let mut lines = one.repeat(repeats);
    lines.push(OPEN_SHADOW);
    lines
}

/// Three steady submissions and the start of a fourth.
fn steady() -> Vec<&'static str> {
    steady_of(&submission(), 3)
}

/// The same run with every handle moved, as a second process would log it.
fn moved(log: &str) -> String {
    log.replace("0x1000", "0xa100")
        .replace("0x2000", "0xa200")
        .replace("0x3000", "0xb300")
        .replace("0x4000", "0xc400")
        .replace("0x5000", "0xd500")
}

#[test]
fn a_log_line_splits_into_its_target_and_message() {
    assert_eq!(
        log_message("[2026-09-25T06:03:12Z INFO  mtld3d::d3d9] [dump] frame start (1 of 3)"),
        Some(("mtld3d::d3d9", "[dump] frame start (1 of 3)"))
    );
    assert_eq!(log_message("wine: continuation line"), None);
}

#[test]
fn the_steady_submission_is_read_into_the_canonical_list() {
    let frame = frame_of(&trace(&steady())).unwrap();
    assert_eq!((frame.submission, frame.submissions), (3, 4));
    assert_eq!((frame.agreeing, frame.considered), (3, 3));
    let listed: Vec<String> = frame
        .passes
        .iter()
        .enumerate()
        .map(|(index, pass)| pass.render(index))
        .collect();
    assert_eq!(
        listed,
        [
            "#0 render 2048x2048 color=C0 srgb=- depth=D0 extra=0x0 load=clear(1,1,1,1)/clear(1) \
             store=store/store/store extra_store=- close=flush_pending_clears cmds=1 draws=0 \
             removed as a dead clear",
            "#1 render 1280x720 color=C1 srgb=- depth=D1 extra=0x0 load=dontcare/clear(1) \
             store=dontcare (next-clear at #3)/store/store extra_store=- close=stretch_rect \
             cmds=223 draws=58 kept",
            "#2 render 640x360 color=C2 srgb=- depth=- extra=0x0 load=load/dontcare \
             store=store/store/store extra_store=- close=set_color_rt cmds=18 draws=1 kept",
            "#3 render 1280x720 color=C1 srgb=- depth=D1 extra=0x0 load=load/load \
             store=store/dontcare (last-use)/store extra_store=- close=submit cmds=70 draws=17 \
             kept",
        ]
    );
    assert_eq!(
        frame.rules,
        [
            "pass-load color=C2 DontCare → Load (sampled this frame)",
            "pass-cull dropped=1 dead clear-only passes",
        ]
    );
}

#[test]
fn warm_up_submissions_outside_the_window_are_left_out() {
    let mut lines = warm_up().repeat(10);
    lines.extend(steady_of(&submission(), WINDOW));
    let frame = frame_of(&trace(&lines)).unwrap();
    assert_eq!((frame.agreeing, frame.considered), (WINDOW, WINDOW));
    assert_eq!(frame.passes.len(), 4);
    assert_eq!(frame.submissions, 10 + WINDOW + 1);
}

#[test]
fn a_shape_needs_eighty_percent_of_the_window() {
    // 24 of 30 is 80 %: enough.
    let mut lines = warm_up().repeat(6);
    lines.extend(steady_of(&submission(), 24));
    let frame = frame_of(&trace(&lines)).unwrap();
    assert_eq!((frame.agreeing, frame.considered), (24, 30));
    assert_eq!(frame.passes.len(), 4);

    // 23 of 30 is not.
    let mut lines = warm_up().repeat(7);
    lines.extend(steady_of(&submission(), 23));
    let reason = frame_of(&trace(&lines)).unwrap_err();
    assert!(
        reason.contains("unstable pass shape")
            && reason.contains("last 30 complete submissions is only 23 of them"),
        "{reason}"
    );

    // Shapes that alternate agree on half the window.
    let mut lines = Vec::new();
    for _ in 0..3 {
        lines.extend(submission());
        lines.extend(warm_up());
    }
    lines.push(OPEN_SHADOW);
    let reason = frame_of(&trace(&lines)).unwrap_err();
    assert!(reason.contains("only 3 of them"), "{reason}");
}

#[test]
fn the_same_work_under_other_handles_is_the_same_shape() {
    let log = trace(&steady());
    let base = frame_of(&log).unwrap();
    let cand = frame_of(&moved(&log)).unwrap();
    assert!(diff(&base, &cand).is_empty(), "{:?}", diff(&base, &cand));
}

#[test]
fn a_store_a_rule_no_longer_drops_is_a_per_pass_difference() {
    let base = frame_of(&trace(&steady())).unwrap();
    let mut lines = steady();
    lines.retain(|line| !line.starts_with("pass-store idx=0 color=0x3000"));
    let cand = frame_of(&trace(&lines)).unwrap();
    assert_eq!(
        diff(&base, &cand),
        ["pass #1: color_store dontcare (next-clear at #3) -> store"]
    );
}

#[test]
fn a_store_on_an_extra_render_target_is_kept_apart_from_render_target_0() {
    let one = vec![
        "pass-open  idx=0 color=0x10 srgb=0x0 depth=0x20 size=64x64 color_load=Load \
         depth_load=Load viewport=0,0+64x64 extra=0x1",
        "pass-close idx=0 caller=set_color_rt color=0x10 depth=0x20 cmds=4 draws=1",
        "pass-open  idx=1 color=0x10 srgb=0x0 depth=0x0 size=64x64 color_load=Load \
         depth_load=DontCare viewport=0,0+64x64 extra=0x0",
        "pass-close idx=1 caller=submit color=0x10 depth=0x0 cmds=4 draws=1",
        "pass-store idx=0 color=0x90 → DontCare (next-clear at idx=1)",
        "pass-store idx=1 color=0x91 → DontCare (last-use)",
        "pass-store idx=0 depth=0x21 → DontCare (last-use)",
    ];
    let frame = frame_of(&trace(&steady_of(&one, 2))).unwrap();
    let pass = &frame.passes[0];
    assert_eq!(pass.color_store, "store", "render target 0 keeps its store");
    assert_eq!(pass.extra_store, "C1=dontcare (next-clear at #1)");
    assert_eq!(pass.depth_store, "store");
    assert_eq!(frame.passes[1].extra_store, "-");
    assert_eq!(
        frame.rules,
        [
            "pass-store #1 color=C2 → DontCare (last-use)",
            "pass-store #0 depth=D1 → DontCare (last-use)",
        ],
        "a store on no attachment of the pass it names stays a rule line"
    );
}

#[test]
fn a_load_revert_that_is_gone_is_a_rule_difference() {
    let base = frame_of(&trace(&steady())).unwrap();
    let mut one = submission();
    one.retain(|line| !line.starts_with("pass-load"));
    one.push("pass-merge idx=2 → idx=1 color=0x3000 depth=0x4000 restored=0");
    let cand = frame_of(&trace(&steady_of(&one, 3))).unwrap();
    assert_eq!(
        diff(&base, &cand),
        [
            "rule - pass-load color=C2 DontCare → Load (sampled this frame)",
            "rule + pass-merge #2 → #1 color=C1 depth=D1 restored=0",
        ]
    );
}

#[test]
fn a_pass_one_leg_lacks_is_listed_whole() {
    let base = frame_of(&trace(&steady())).unwrap();
    let mut one = submission();
    let submit = one
        .iter()
        .position(|line| line.starts_with("pass-close idx=3"))
        .unwrap();
    let added = [
        "pass-open  idx=4 color=0x3000 srgb=0x0 depth=0x0 size=1280x720 color_load=Load \
         depth_load=DontCare viewport=0,0+1280x720 extra=0x0",
        "pass-close idx=4 caller=submit color=0x3000 depth=0x0 cmds=9 draws=3",
    ];
    for (offset, line) in added.into_iter().enumerate() {
        one.insert(submit + 1 + offset, line);
    }
    let cand = frame_of(&trace(&steady_of(&one, 3))).unwrap();
    assert_eq!(
        diff(&base, &cand),
        [
            "passes: base 4, cand 5",
            "only in cand: #4 render 1280x720 color=C1 srgb=- depth=- extra=0x0 load=load/dontcare \
             store=store/store/store extra_store=- close=submit cmds=9 draws=3 kept",
        ]
    );
}

#[test]
fn submissions_without_rule_lines_split_where_the_index_restarts() {
    let one = "pass-open  idx=0 color=0x1 srgb=0x0 depth=0x0 size=8x8 color_load=Load \
               depth_load=DontCare viewport=0,0+8x8 extra=0x0";
    let two = "pass-open  idx=1 color=0x2 srgb=0x0 depth=0x0 size=4x4 color_load=Load \
               depth_load=DontCare viewport=0,0+4x4 extra=0x0";
    let frame = frame_of(&trace(&[one, two, one, two, one, two, one])).unwrap();
    assert_eq!((frame.agreeing, frame.considered), (3, 3));
    assert_eq!(frame.passes.len(), 2);
    assert_eq!(frame.passes[1].size, "4x4");
}

#[test]
fn an_upload_pass_goes_in_front_and_the_close_finds_the_open_pass() {
    let lines = [
        "pass-open  idx=0 color=0x1 srgb=0x0 depth=0x0 size=8x8 color_load=Load \
         depth_load=DontCare viewport=0,0+8x8 extra=0x0",
        "upload-pass idx=0 color=0x9 slice=0 level=0 size=256x256 rect=0,0+256x256 load=DontCare",
        "pass-close idx=1 caller=submit color=0x1 depth=0x0 cmds=4 draws=1",
        "pass-coalesce drop idx=0 (clear-only) → fold into idx=1 color=0x1 depth=0x0",
        "pass-open  idx=0 color=0x1 srgb=0x0 depth=0x0 size=8x8 color_load=Load \
         depth_load=DontCare viewport=0,0+8x8 extra=0x0",
    ];
    let frame = frame_of(&trace(&lines)).unwrap();
    assert_eq!(frame.passes[0].kind, "upload");
    assert_eq!(frame.passes[0].color, "C1");
    assert_eq!(frame.passes[0].color_load, "dontcare");
    assert_eq!(frame.passes[0].fate, "coalesced into #1");
    assert_eq!(frame.passes[1].close, "submit");
    assert_eq!(frame.passes[1].draws, "1");
    assert!(frame.rules.is_empty(), "{:?}", frame.rules);
}

#[test]
fn a_log_without_a_complete_submission_is_an_error() {
    let reason = frame_of("[2026-09-26T00:00:00Z WARN  mtld3d::d3d9] something\n").unwrap_err();
    assert!(reason.contains("no pass trace"), "{reason}");
    let reason = frame_of(&trace(&[OPEN_SHADOW])).unwrap_err();
    assert!(reason.contains("one submission"), "{reason}");
}

/// An A/B directory holding one benchmark's shape run in each leg.
struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("mtld3d-bench-shape-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        Self { root }
    }

    /// Write `leg`'s shape run of test `e2e.b` with `log` as its layer log.
    fn write(&self, leg: &str, log: &str) {
        let dir = self.root.join(leg).join(SHAPE_DIR).join("e2e.b");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("e2e-42.log"), log).unwrap();
        fs::write(dir.join("bench-b.metrics"), "meta b layer v1\n").unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn a_changed_shape_fails_unless_accepted() {
    let fixture = Fixture::new("changed");
    fixture.write("base", &trace(&steady()));
    let mut lines = steady();
    lines.retain(|line| !line.starts_with("pass-store idx=2"));
    fixture.write("cand", &moved(&trace(&lines)));

    let comparison = compare_dir(&fixture.root, &[]).unwrap();
    let [report] = comparison.reports.as_slice() else {
        panic!("one report: {comparison:?}");
    };
    assert_eq!(report.bench, "b");
    assert!(report.fails());
    assert_eq!(
        report.diff,
        ["pass #3: depth_store dontcare (last-use) -> store"]
    );
    assert!(
        report
            .summary
            .contains("shared by 3 of the last 3 complete submissions (latest 3 of 4)"),
        "{}",
        report.summary
    );

    for accept in ["shape", "shape:b", "shape:e2e.b"] {
        let comparison = compare_dir(&fixture.root, &[accept.to_owned()]).unwrap();
        assert!(!comparison.reports[0].fails(), "--accept {accept}");
        assert!(comparison.notes.is_empty(), "{:?}", comparison.notes);
    }
    let comparison = compare_dir(&fixture.root, &["shape:other".to_owned()]).unwrap();
    assert!(comparison.reports[0].fails());
    assert_eq!(
        comparison.notes,
        ["--accept shape:other: no shape of that name changed"]
    );
}

#[test]
fn an_unchanged_shape_passes() {
    let fixture = Fixture::new("same");
    fixture.write("base", &trace(&steady()));
    fixture.write("cand", &moved(&trace(&steady())));
    let comparison = compare_dir(&fixture.root, &[]).unwrap();
    assert!(!comparison.reports[0].changed());
}

#[test]
fn an_unstable_leg_is_an_error_naming_its_log() {
    let fixture = Fixture::new("unstable");
    fixture.write("base", &trace(&steady()));
    let mut lines = warm_up().repeat(2);
    lines.extend(steady_of(&submission(), 2));
    fixture.write("cand", &trace(&lines));
    let reason = compare_dir(&fixture.root, &[]).unwrap_err();
    assert!(
        reason.contains("e2e-42.log: unstable pass shape"),
        "{reason}"
    );
}

#[test]
fn a_directory_without_shape_runs_is_a_note() {
    let fixture = Fixture::new("none");
    let comparison = compare_dir(&fixture.root, &[]).unwrap();
    assert!(comparison.reports.is_empty());
    assert!(!comparison.present);
    assert!(
        comparison.notes[0].contains("no shape runs"),
        "{:?}",
        comparison.notes
    );
}

#[test]
fn shape_runs_in_one_leg_or_without_a_trace_are_errors() {
    let fixture = Fixture::new("one-leg");
    fixture.write("base", &trace(&steady()));
    let reason = compare_dir(&fixture.root, &[]).unwrap_err();
    assert!(reason.contains("only one leg"), "{reason}");

    fixture.write(
        "cand",
        "[2026-09-26T00:00:00Z WARN  mtld3d::d3d9] no trace here\n",
    );
    let reason = compare_dir(&fixture.root, &[]).unwrap_err();
    assert!(
        reason.contains("no layer log with a pass trace"),
        "{reason}"
    );
}

#[test]
fn only_shape_names_are_shape_accepts() {
    assert!(is_accept_name("shape"));
    assert!(is_accept_name("shape:wow_335a"));
    assert!(!is_accept_name("perf.draws_pf"));
    assert!(!is_accept_name("shapes"));
    assert_eq!(
        run_dir_name("bench_frame_shape::wow_335a_busy_frame"),
        "bench_frame_shape.wow_335a_busy_frame"
    );
}

/// The level `filter`, a `RUST_LOG` of `target=level` directives, gives `target`.
///
/// The most specific directive whose module path is `target` or a parent of
/// it wins, as `env_logger` picks it.
fn level_for<'a>(filter: &'a str, target: &str) -> Option<&'a str> {
    filter
        .split(',')
        .filter_map(|directive| directive.split_once('='))
        .filter(|(path, _)| target == *path || target.starts_with(&format!("{path}::")))
        .max_by_key(|(path, _)| path.len())
        .map(|(_, level)| level)
}

#[test]
fn the_shape_run_keeps_the_identity_lines_and_the_pass_trace() {
    // The `d3d9.dll ... loaded at` and `mtld3d.so ... initialized` lines, at info.
    assert_eq!(level_for(SHAPE_RUST_LOG, "mtld3d::d3d9"), Some("info"));
    assert_eq!(level_for(SHAPE_RUST_LOG, "mtld3d::unix"), Some("info"));
    assert_eq!(
        level_for(SHAPE_RUST_LOG, "mtld3d::d3d9::passes"),
        Some("trace")
    );
    assert_eq!(level_for(SHAPE_RUST_LOG, "mtld3d::perf"), Some("warn"));
    assert_eq!(level_for(SHAPE_RUST_LOG, "mtld3d::core"), Some("warn"));
}
