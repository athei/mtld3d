//! Run the benchmarks against the two legs, interleaved, and check every run as it lands.
//!
//! A round of a leg is one process of the candidate's test binary running
//! every selected benchmark it carries, with libtest's `--ignored` on one
//! thread, under the leg's Wine loader and prefix and with `log.dir` pointed
//! at `<out>/<leg>/<round>`, where each benchmark writes its
//! `bench-<name>.metrics` and names itself in it (`meta test`). Round `r`
//! runs the base first when `r` is even and the candidate first when it is
//! odd. The benchmarks run in libtest's order, by test path, the same in
//! both legs and every round, so what one leaves in the process for the
//! next (the process-wide pipeline cache, the page-box pool, the address
//! space its memory rows sample) is the same on both sides of each pair;
//! each still creates its own device, whose first perf window opens with
//! it, and warms up and aligns its span as it does alone. After the last
//! round, each benchmark whose metrics declare `shape` lines
//! runs once more in either leg with the pass trace on, into
//! `<out>/<leg>/shape/<test>`; that run is never timed (the trace costs
//! frame time), and its layer log is the pass shape `compare` diffs between
//! the legs. It is stopped once the log holds enough steady submissions
//! (`shape::Watch`), so it writes no metrics: its build is checked from the
//! log's identity lines instead, against the leg's stamp and the images the
//! leg's timed rounds loaded. A benchmark without `shape` lines gets no
//! shape run, with a note: its frame is not meant to be steady. A
//! benchmark that fails or writes no metrics file, or a process whose files
//! report a layer stamp other than its leg's, ends the whole A/B run at
//! once, naming the benchmark: every number after it would be measured
//! against the wrong build or none.
//!
//! The host emitter benchmark, when the run has one, runs the first
//! benchmark processes, in rounds of its own before the end-to-end
//! benchmarks (only the short Wine process that lists them comes before),
//! so that no benchmark's Wine process is still exiting while it times host
//! code, and without a shape run. It is host code, so each leg runs its own
//! tree's `emit_corpus`, built with that leg's profile, with `--metrics`
//! pointed at the same round directory, and the run is checked the same
//! way.

use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    thread,
    time::{Duration, Instant, SystemTime},
};

use super::{
    Leg, SAME_IMAGE_FILE, SHAPE_DIR, WINE_FILE,
    compare::{self, Options},
    machine,
    metrics::{self, Class, MetricsFile},
    shape::{self, Identity, SHAPE_RUST_LOG},
};
use crate::{
    attribute::{self, BinaryOutcome, Launcher as _, Report, TestResult, Verdict},
    binary::{WineLauncher, binary_name, stderr_tail},
    run::ExitKind,
    select::{selected, test_id},
};

/// The metric a progress line shows, when the benchmark reports it.
const PROGRESS_METRIC: &str = "frame.p50";

/// The name under a run directory where a benchmark finds the staged shader caches.
const CORPUS_LINK: &str = "corpus";

/// How progress lines and errors name the host emitter benchmark.
const HOST_ID: &str = "host::emit_corpus";

/// The file in a round directory that keeps what the host benchmark printed.
const HOST_LOG: &str = "host-emit.log";

/// How long to sleep between two looks at a running host benchmark.
const HOST_POLL: Duration = Duration::from_millis(50);

/// How many of its last lines a failed host benchmark's error quotes.
const HOST_TAIL_LINES: usize = 15;

/// The image value of a binary whose log line names none, as the metrics files write it.
const UNKNOWN_IMAGE: &str = "unknown";

/// The meta key in which a benchmark names the libtest path of the test that wrote the file.
const TEST_META: &str = "test";

/// One leg: the Wine that runs it, its prefix, and the layer stamp its runs must report.
#[derive(Debug)]
pub struct LegSpec {
    /// The Wine loader of the tree the leg's layer is installed into.
    pub wine: PathBuf,
    /// The prefix the leg's processes run in.
    pub prefix: PathBuf,
    /// The `meta layer` value every metrics file of the leg must carry.
    pub stamp: String,
}

/// A parsed `bench-ab` invocation.
#[derive(Debug)]
pub struct AbConfig {
    pub base: LegSpec,
    pub cand: LegSpec,
    /// The candidate's test binaries; their `#[ignore]`d tests are the benchmarks.
    pub exes: Vec<PathBuf>,
    /// Patterns selecting the benchmarks by test id; none selects every one.
    pub benches: Vec<String>,
    /// Rounds per benchmark, each one run of either leg.
    pub runs: u32,
    /// The A/B directory the runs write into.
    pub out: PathBuf,
    /// The `MTLD3D_CONFIG` of every run, before the run's own `log.dir`.
    pub config: String,
    /// How long a run may go without a line before it counts as hung.
    pub timeout: Duration,
    pub options: Options,
    /// Where the report is written besides stdout.
    pub report: Option<PathBuf>,
    /// The host emitter benchmark, when both trees have one.
    pub host: Option<HostBench>,
    /// The staged shader caches, linked into every end-to-end run's directory as `corpus`.
    pub corpus_dir: Option<PathBuf>,
}

/// The host emitter benchmark: each leg's own `emit_corpus`, and the caches both read.
#[derive(Debug)]
pub struct HostBench {
    /// The base tree's `emit_corpus`.
    pub base: PathBuf,
    /// The candidate tree's `emit_corpus`.
    pub cand: PathBuf,
    /// Shader caches both legs time besides the synthetic corpora.
    pub corpora: Vec<PathBuf>,
}

/// The images a run loaded: the `d3d9.dll` and the `mtld3d.so`, as the metrics files name them.
#[derive(Debug, PartialEq, Eq)]
pub struct Images {
    pub layer: String,
    pub unix: String,
}

/// One run of the A/B schedule.
#[derive(Debug, PartialEq, Eq)]
pub enum Step {
    /// Round `round` in `leg` of the benchmarks of test binary `group`, one process for all.
    Round { group: usize, round: u32, leg: Leg },
    /// The untimed run of benchmark `bench` in `leg` under the pass trace.
    Shape { bench: usize, leg: Leg },
    /// Round `round` of the host emitter benchmark in `leg`.
    Host { round: u32, leg: Leg },
}

/// One benchmark to run: the binary that carries it and its libtest path.
#[derive(Debug)]
pub struct Bench {
    pub exe: PathBuf,
    pub name: String,
    /// `<binary>::<name>`, how progress lines name it.
    pub id: String,
}

/// Run the A/B comparison `config` describes, then judge it.
///
/// # Errors
///
/// Returns a message when the output directory already holds a run, no
/// benchmark is selected, or a run fails or cannot be trusted; the caller
/// exits with code 2.
pub fn run(config: &AbConfig) -> Result<ExitCode, String> {
    let out = std::path::absolute(&config.out)
        .map_err(|e| format!("could not resolve {}: {e}", config.out.display()))?;
    for leg in [Leg::Base, Leg::Cand] {
        let dir = out.join(leg.dir());
        if dir.exists() {
            return Err(format!(
                "{} already holds a run; an A/B run starts from a directory of its own",
                dir.display()
            ));
        }
    }
    let wine = check_wine(&config.base, &config.cand)?;
    fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let wine_file = out.join(WINE_FILE);
    fs::write(&wine_file, format!("{wine}\n"))
        .map_err(|e| format!("{}: {e}", wine_file.display()))?;
    println!("bench-ab: both legs run {wine}");
    let benches = select_benches(config)?;
    if config.options.allow_same_image {
        // Recorded in the directory so that a later `bench-compare` of it
        // allows what this run allowed.
        let marker = out.join(SAME_IMAGE_FILE);
        fs::write(
            &marker,
            "both legs build one commit from a clean tree; they may load one d3d9.dll image\n",
        )
        .map_err(|e| format!("{}: {e}", marker.display()))?;
    }
    let groups = group_by_binary(&benches);
    println!(
        "bench-ab: {} benchmarks in {} process{} a leg a round{}, {} rounds, both legs each \
         round, into {}",
        benches.len(),
        groups.len(),
        if groups.len() == 1 { "" } else { "es" },
        if config.host.is_some() {
            ", and the host emitter benchmark"
        } else {
            ""
        },
        config.runs,
        out.display()
    );
    // Whether each benchmark's metrics declare its frame in `shape` lines,
    // which is what earns it a shape run after the rounds, and what its
    // rounds wrote that the shape runs are checked against.
    let mut declares_shape = vec![false; benches.len()];
    let mut timed: Vec<Timed> = benches.iter().map(|_| Timed::default()).collect();
    // The Wine installs the legs boot from, whose processes the machine
    // samples count as the run's own.
    let wines: Vec<PathBuf> = [&config.base.wine, &config.cand.wine]
        .into_iter()
        .filter_map(|wine| Some(wine.parent()?.parent()?.to_path_buf()))
        .collect();
    for step in schedule(
        groups.len(),
        benches.len(),
        config.host.is_some(),
        config.runs,
    ) {
        let leg = match &step {
            Step::Round { leg, .. } | Step::Shape { leg, .. } | Step::Host { leg, .. } => leg,
        };
        let spec = match leg {
            Leg::Base => &config.base,
            Leg::Cand => &config.cand,
        };
        match step {
            Step::Round { group, round, leg } => {
                let members: Vec<&Bench> = groups[group].iter().map(|&at| &benches[at]).collect();
                let dir = out.join(leg.dir()).join(round.to_string());
                machine::keep(
                    &dir,
                    &binary_name(&members[0].exe),
                    &machine::sample(&wines),
                )?;
                for (member, path, file) in run_round(config, spec, &members, &dir)? {
                    let at = groups[group][member];
                    check_stamp(&path, &file, spec)?;
                    declares_shape[at] |= !file.shape.is_empty();
                    timed[at].note(&leg, &path, &file);
                    println!(
                        "bench-ab: {} round {}/{} {}: {}",
                        benches[at].id,
                        round + 1,
                        config.runs,
                        leg.dir(),
                        progress(&path, &file)
                    );
                }
            }
            Step::Shape { bench: at, leg } => {
                let bench = &benches[at];
                if !declares_shape[at] {
                    if leg == Leg::Base {
                        println!(
                            "bench-ab: {}: its metrics declare no shape lines, so it gets no \
                             shape run",
                            bench.id
                        );
                    }
                    continue;
                }
                let dir = out
                    .join(leg.dir())
                    .join(SHAPE_DIR)
                    .join(shape::run_dir_name(&bench.name));
                let (log, kind) = run_shape(config, spec, bench, &dir, &timed[at].benches)?;
                let identity = shape::identity(&log)?;
                check_shape_build(&log, &identity, spec, timed[at].images(&leg))?;
                println!(
                    "bench-ab: {} shape run {}: pass trace in {}{}",
                    bench.id,
                    leg.dir(),
                    dir.display(),
                    if kind == ExitKind::Stopped {
                        ", stopped once it held the steady submissions"
                    } else {
                        ""
                    }
                );
            }
            Step::Host { round, leg } => {
                // The schedule has host steps only when the run has the benchmark.
                let Some(host) = &config.host else {
                    continue;
                };
                let exe = match leg {
                    Leg::Base => &host.base,
                    Leg::Cand => &host.cand,
                };
                let dir = out.join(leg.dir()).join(round.to_string());
                machine::keep(&dir, "host", &machine::sample(&wines))?;
                for (path, file) in &run_host(exe, &host.corpora, &dir, config.timeout)? {
                    check_stamp(path, file, spec)?;
                    println!(
                        "bench-ab: {HOST_ID} round {}/{} {}: {}",
                        round + 1,
                        config.runs,
                        leg.dir(),
                        progress(path, file)
                    );
                }
            }
        }
    }
    compare::judge_dir(&out, &config.options, config.report.as_deref())
}

/// The order of the runs, the leg that goes first alternating from round to round.
///
/// The host emitter benchmark, when there is one (`host`), runs its rounds
/// first, both legs back to back, the base first on even rounds, as the
/// run's first benchmark processes: it times host code, and before any
/// end-to-end benchmark's process has run none is still exiting (its
/// session tearing down, the kill of a stopped shape run) on the cores it
/// measures; only the short `--list` of the benchmarks runs under Wine
/// before it. Then every round runs each
/// of the `groups` test binaries' processes in both legs the same way, and
/// after the last round each of the `benches` gets its shape runs, the
/// base's first.
#[must_use]
pub fn schedule(groups: usize, benches: usize, host: bool, runs: u32) -> Vec<Step> {
    let legs = |round: u32| {
        if round.is_multiple_of(2) {
            [Leg::Base, Leg::Cand]
        } else {
            [Leg::Cand, Leg::Base]
        }
    };
    let mut order = Vec::new();
    if host {
        for round in 0..runs {
            for leg in legs(round) {
                order.push(Step::Host { round, leg });
            }
        }
    }
    for round in 0..runs {
        for group in 0..groups {
            for leg in legs(round) {
                order.push(Step::Round { group, round, leg });
            }
        }
    }
    for bench in 0..benches {
        for leg in [Leg::Base, Leg::Cand] {
            order.push(Step::Shape { bench, leg });
        }
    }
    order
}

/// The benchmarks grouped by the test binary that carries them, in the order they were found.
///
/// Each group is one process a leg a round; its members are indices into `benches`.
#[must_use]
pub fn group_by_binary(benches: &[Bench]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (at, bench) in benches.iter().enumerate() {
        match groups
            .iter_mut()
            .find(|group| benches[group[0]].exe == bench.exe)
        {
            Some(group) => group.push(at),
            None => groups.push(vec![at]),
        }
    }
    groups
}

/// Check that both legs run one Wine, and name it.
///
/// The legs differ only in the layer, or the comparison measures Wine too:
/// the loaders have to report the same `--version` and the wineservers
/// beside them have to be the same file, byte for byte, since two builds of
/// one Wine version can still differ.
///
/// # Errors
///
/// Returns a message when a loader cannot be run or the two differ.
pub fn check_wine(base: &LegSpec, cand: &LegSpec) -> Result<String, String> {
    let base_version = wine_version(&base.wine)?;
    let cand_version = wine_version(&cand.wine)?;
    if base_version != cand_version {
        return Err(format!(
            "the legs run different Wines: base {base_version} ({}), cand {cand_version} ({})",
            base.wine.display(),
            cand.wine.display()
        ));
    }
    let server = |spec: &LegSpec| {
        let path = spec.wine.with_file_name("wineserver");
        fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))
    };
    if server(base)? != server(cand)? {
        return Err(format!(
            "the legs run different wineservers beside {} and {}, though both report \
             {base_version}",
            base.wine.display(),
            cand.wine.display()
        ));
    }
    Ok(format!("{base_version}, one wineserver"))
}

/// What `wine --version` prints, which the loader answers without a prefix or a server.
fn wine_version(wine: &Path) -> Result<String, String> {
    let output = Command::new(wine)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("{} --version: {e}", wine.display()))?;
    let version = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !output.status.success() || version.is_empty() {
        return Err(format!(
            "{} --version ended with {} and printed {version:?}",
            wine.display(),
            output.status
        ));
    }
    Ok(version)
}

/// The benchmarks the patterns select, listed out of each binary under the candidate leg.
///
/// A pattern that selects nothing is noted and skipped, since a benchmark
/// set may name benchmarks a checkout does not carry yet; a test two
/// patterns select runs once.
fn select_benches(config: &AbConfig) -> Result<Vec<Bench>, String> {
    let mut found: Vec<Bench> = Vec::new();
    for exe in &config.exes {
        let binary = binary_name(exe);
        let mut launcher = leg_launcher(&config.cand, exe, None, config.timeout)?;
        for name in launcher.list()? {
            let id = test_id(&binary, &name);
            if selected(&id, &config.benches) {
                found.push(Bench {
                    exe: exe.clone(),
                    name,
                    id,
                });
            }
        }
    }
    let ids: Vec<&str> = found.iter().map(|bench| bench.id.as_str()).collect();
    for note in check_selection(&config.benches, &ids, config.host.is_some())? {
        println!("{note}");
    }
    Ok(found)
}

/// Check a selection: the end-to-end benchmarks `ids` the `patterns` found, with or without `host`.
///
/// Returns the notes to print: one per filter that matches nothing, once
/// however often it repeats, leaving out a filter that names the host
/// emitter benchmark when the run has it. A selection without end-to-end
/// benchmarks runs only when some filter names the host emitter and the
/// run has it: the host rounds alone.
///
/// # Errors
///
/// Returns a message when the selection holds nothing to compare.
pub fn check_selection(
    patterns: &[String],
    ids: &[&str],
    host: bool,
) -> Result<Vec<String>, String> {
    let for_host = |pattern: &str| host && names_host(pattern);
    let mut notes = Vec::new();
    let mut seen = BTreeSet::new();
    for pattern in patterns {
        if !seen.insert(pattern.as_str()) || for_host(pattern) {
            continue;
        }
        if !ids.iter().any(|id| id.contains(pattern.as_str())) {
            notes.push(format!(
                "bench-ab: no benchmark matches {pattern:?}; skipped"
            ));
        }
    }
    if ids.is_empty() && !patterns.iter().any(|pattern| for_host(pattern)) {
        return Err("no benchmark selected: nothing to compare".to_owned());
    }
    Ok(notes)
}

/// Whether the filter `pattern` selects the host emitter benchmark, as filters select test paths.
///
/// A filter selects a benchmark whose id contains it, and the host
/// emitter's id is [`HOST_ID`], so `host`, `emit` and `emit_corpus` select
/// it and `emissive` does not.
#[must_use]
pub fn names_host(pattern: &str) -> bool {
    !pattern.is_empty() && HOST_ID.contains(pattern)
}

/// A launcher for `exe` under `spec`'s Wine and prefix, running the `#[ignore]`d tests.
fn leg_launcher(
    spec: &LegSpec,
    exe: &Path,
    log_dir: Option<&Path>,
    timeout: Duration,
) -> Result<WineLauncher, String> {
    Ok(
        WineLauncher::new(&spec.wine, exe, log_dir, timeout, Box::new(|_| {}))?
            .ignored_only(true)
            .with_env("WINEPREFIX", &spec.prefix.to_string_lossy()),
    )
}

/// The `MTLD3D_CONFIG` of a run writing into `dir`: the base config, then its `log.dir`.
///
/// The layer reads the path on the PE side, where the unix root is drive
/// `Z:`. It comes last so that it wins over any `log.dir` in the base.
#[must_use]
pub fn run_config(base: &str, dir: &Path) -> String {
    let log_dir = format!("log.dir=Z:{}", dir.display());
    if base.is_empty() {
        log_dir
    } else {
        format!("{base};{log_dir}")
    }
}

/// Run one round of `benches`, all carried by one test binary, in one process under `spec`.
///
/// Returns every metrics file the process wrote into `dir`, each with the
/// index in `benches` of the benchmark that names itself in it.
///
/// # Errors
///
/// Returns a message naming the benchmarks that did not pass, or one that
/// passed and wrote no metrics, or a file that names no selected benchmark;
/// and one when the driver reported a GPU hang.
fn run_round(
    config: &AbConfig,
    spec: &LegSpec,
    benches: &[&Bench],
    dir: &Path,
) -> Result<Vec<(usize, PathBuf, MetricsFile)>, String> {
    let Some(first) = benches.first() else {
        return Ok(Vec::new());
    };
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if let Some(corpus) = &config.corpus_dir {
        link_corpus(corpus, dir)?;
    }
    let before = metrics_files(dir)?;
    let mut launcher = leg_launcher(spec, &first.exe, Some(dir), config.timeout)?
        .with_env("MTLD3D_CONFIG", &run_config(&config.config, dir));
    let names: Vec<String> = benches.iter().map(|bench| bench.name.clone()).collect();
    let mut outcome = Outcome::default();
    let run = attribute::run_binary(&mut launcher, Some(names), 1, true, &mut outcome)?;
    if run.outcome == BinaryOutcome::GpuHang {
        return Err(format!(
            "the round under {}: the driver reported a GPU hang; no number after it can be \
             trusted{}",
            dir.display(),
            outcome.notes()
        ));
    }
    check_verdicts(benches, &outcome.results, run.failed, dir)
        .map_err(|reason| format!("{reason}{}", outcome.notes()))?;
    let written = new_files(dir, &before)?
        .into_iter()
        .map(|path| metrics::read(&path).map(|file| (path, file)))
        .collect::<Result<Vec<_>, String>>()?;
    assign(benches, written, dir)
}

/// Check that every one of `benches` passed in a round process that reported `results`.
///
/// # Errors
///
/// Returns a message naming each benchmark that did not pass, with what
/// became of it, when any did not or the process failed.
pub fn check_verdicts(
    benches: &[&Bench],
    results: &[TestResult],
    failed: bool,
    dir: &Path,
) -> Result<(), String> {
    let mut not_passed = Vec::new();
    for bench in benches {
        match results.iter().find(|result| result.name == bench.name) {
            Some(result) if result.verdict == Verdict::Passed => {}
            Some(result) => not_passed.push(format!("{}: {:?}", bench.id, result.verdict)),
            None => not_passed.push(format!("{}: no result", bench.id)),
        }
    }
    if not_passed.is_empty() && !failed {
        return Ok(());
    }
    Err(format!(
        "the round under {} did not pass: {}",
        dir.display(),
        if not_passed.is_empty() {
            "the process failed after every benchmark passed".to_owned()
        } else {
            not_passed.join("; ")
        }
    ))
}

/// The benchmark of `benches` that wrote each file of `written`, by the test its `meta test` names.
///
/// # Errors
///
/// Returns a message for a file that names no test or one not in the
/// round, and for a benchmark that wrote no file.
pub fn assign(
    benches: &[&Bench],
    written: Vec<(PathBuf, MetricsFile)>,
    dir: &Path,
) -> Result<Vec<(usize, PathBuf, MetricsFile)>, String> {
    let mut assigned = Vec::with_capacity(written.len());
    for (path, file) in written {
        let test = file.meta.get(TEST_META).ok_or_else(|| {
            format!(
                "{}: no meta {TEST_META} line names the benchmark that wrote it",
                path.display()
            )
        })?;
        let at = benches
            .iter()
            .position(|bench| bench.name == *test)
            .ok_or_else(|| {
                format!(
                    "{}: written by {test}, which the round under {} did not run",
                    path.display(),
                    dir.display()
                )
            })?;
        assigned.push((at, path, file));
    }
    if let Some(bench) = benches
        .iter()
        .enumerate()
        .find(|(at, _)| !assigned.iter().any(|(owner, _, _)| owner == at))
        .map(|(_, bench)| bench)
    {
        return Err(format!(
            "{} passed but wrote no bench-<name>.metrics into {}",
            bench.id,
            dir.display()
        ));
    }
    Ok(assigned)
}

/// Run `bench`'s shape run under `spec` into `dir`, stopped once its log holds the steady frame.
///
/// The process runs under [`SHAPE_RUST_LOG`] and is ended as soon as
/// [`shape::Watch`] has [`shape::STOP_AFTER`] submissions after the
/// benchmark's measured frames start; one that ends first ends on its own.
/// `benches`, the benchmarks the test's timed rounds wrote, go into
/// [`shape::BENCHES_FILE`] beside the log. Returns the layer log the run
/// wrote and how the process ended.
///
/// # Errors
///
/// Returns a message when the directory cannot be written, the process
/// cannot be run, the driver reported a GPU hang, or the process ended on
/// its own with anything but success.
fn run_shape(
    config: &AbConfig,
    spec: &LegSpec,
    bench: &Bench,
    dir: &Path,
    benches: &BTreeSet<String>,
) -> Result<(PathBuf, ExitKind), String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if let Some(corpus) = &config.corpus_dir {
        link_corpus(corpus, dir)?;
    }
    let names = dir.join(shape::BENCHES_FILE);
    let text = benches.iter().fold(String::new(), |mut text, name| {
        let _ = writeln!(text, "{name}");
        text
    });
    fs::write(&names, text).map_err(|e| format!("{}: {e}", names.display()))?;
    let mut launcher = leg_launcher(spec, &bench.exe, Some(dir), config.timeout)?
        .with_env("MTLD3D_CONFIG", &run_config(&config.config, dir))
        .with_env("RUST_LOG", SHAPE_RUST_LOG);
    let mut watch = shape::Watch::default();
    let end = launcher.run_until(&bench.name, &mut |log, stdout| watch.look(log, stdout))?;
    let what = format!("the shape run of {} under {}", bench.id, dir.display());
    if end.gpu_hang {
        return Err(format!(
            "{what}: the driver reported a GPU hang; no number after it can be trusted"
        ));
    }
    if !matches!(end.kind, ExitKind::Stopped | ExitKind::Code(0)) {
        return Err(format!(
            "{what} ended with {}:\n{}",
            end.kind.describe(),
            stderr_tail(&end.stderr)
        ));
    }
    let stem = bench
        .exe
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let log = dir.join(mtld3d_shared::log_paths::log_file_name(&stem, end.pid));
    Ok((log, end.kind))
}

/// Link the staged caches at `corpus` into the run directory `dir` as `corpus`, once.
///
/// A benchmark reads real caches from `corpus` under its log directory, as
/// `make bench` stages them. Both legs' runs link the one staged copy, which
/// the benchmark only copies from, so no leg sees a cache the other did not.
///
/// # Errors
///
/// Returns a message when the link cannot be made, or `dir` already holds a
/// `corpus` that is not a link to `corpus`.
pub fn link_corpus(corpus: &Path, dir: &Path) -> Result<(), String> {
    let link = dir.join(CORPUS_LINK);
    match fs::read_link(&link) {
        Ok(target) if target == corpus => Ok(()),
        Ok(target) => Err(format!(
            "{} links to {}, not to the staged caches in {}",
            link.display(),
            target.display(),
            corpus.display()
        )),
        Err(_) if link.symlink_metadata().is_ok() => Err(format!(
            "{} exists and is not a link to the staged caches",
            link.display()
        )),
        Err(_) => std::os::unix::fs::symlink(corpus, &link)
            .map_err(|e| format!("{}: {e}", link.display())),
    }
}

/// The arguments of one host benchmark run writing into `dir`.
#[must_use]
pub fn host_args(dir: &Path, corpora: &[PathBuf]) -> Vec<OsString> {
    let mut args = vec![OsString::from("--metrics"), dir.as_os_str().to_owned()];
    args.extend(corpora.iter().map(|corpus| corpus.as_os_str().to_owned()));
    args
}

/// Run the host benchmark `exe` once into `dir`, and read the metrics files it wrote there.
///
/// What it prints goes to [`HOST_LOG`] in `dir`. It fails the run when it
/// exits unsuccessfully, runs longer than `timeout`, or writes no metrics.
///
/// # Errors
///
/// Returns a message naming the executable and quoting the end of its output.
pub fn run_host(
    exe: &Path,
    corpora: &[PathBuf],
    dir: &Path,
    timeout: Duration,
) -> Result<Vec<(PathBuf, MetricsFile)>, String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let before = metrics_files(dir)?;
    let log_path = dir.join(HOST_LOG);
    let log = fs::File::create(&log_path).map_err(|e| format!("{}: {e}", log_path.display()))?;
    let log_err = log
        .try_clone()
        .map_err(|e| format!("{}: {e}", log_path.display()))?;
    let what = format!("{HOST_ID} ({}) into {}", exe.display(), dir.display());
    let mut child = Command::new(exe)
        .args(host_args(dir, corpora))
        .stdin(Stdio::null())
        .stdout(log)
        .stderr(log_err)
        .spawn()
        .map_err(|e| format!("{what}: {e}"))?;
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| format!("{what}: {e}"))? {
            break status;
        }
        if started.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "{what} ran longer than {timeout:?}{}",
                tail(&log_path)
            ));
        }
        thread::sleep(HOST_POLL);
    };
    if !status.success() {
        return Err(format!("{what} ended with {status}{}", tail(&log_path)));
    }
    let written = new_files(dir, &before)?;
    if written.is_empty() {
        return Err(format!(
            "{what} succeeded but wrote no bench-<name>.metrics{}",
            tail(&log_path)
        ));
    }
    written
        .into_iter()
        .map(|path| metrics::read(&path).map(|file| (path, file)))
        .collect()
}

/// The last lines of the file at `path`, each on a line of its own, for an error message.
fn tail(path: &Path) -> String {
    let text = fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(HOST_TAIL_LINES);
    lines[start..].iter().fold(
        format!("; the end of {}:", path.display()),
        |mut out, line| {
            let _ = write!(out, "\n{line}");
            out
        },
    )
}

/// The metrics files in `dir` that are new or changed since `before` was taken.
fn new_files(
    dir: &Path,
    before: &BTreeMap<PathBuf, (SystemTime, u64)>,
) -> Result<Vec<PathBuf>, String> {
    Ok(metrics_files(dir)?
        .into_iter()
        .filter(|(path, stamp)| before.get(path) != Some(stamp))
        .map(|(path, _)| path)
        .collect())
}

/// Every metrics file in `dir`, with its modification time and length.
fn metrics_files(dir: &Path) -> Result<BTreeMap<PathBuf, (SystemTime, u64)>, String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut files = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        if entry
            .file_name()
            .to_str()
            .and_then(metrics::bench_of)
            .is_none()
        {
            continue;
        }
        let meta = entry
            .metadata()
            .map_err(|e| format!("{}: {e}", entry.path().display()))?;
        let modified = meta
            .modified()
            .map_err(|e| format!("{}: {e}", entry.path().display()))?;
        files.insert(entry.path(), (modified, meta.len()));
    }
    Ok(files)
}

/// Check that a run's metrics file names the layer its leg installed.
///
/// # Errors
///
/// Returns a message when `meta layer` is missing or is another stamp.
pub fn check_stamp(path: &Path, file: &MetricsFile, spec: &LegSpec) -> Result<(), String> {
    check_layer(
        path,
        file.meta.get("layer").map(String::as_str),
        spec,
        "no meta layer line",
    )
}

/// Check that a shape run's log names the build its leg installed and its timed rounds loaded.
///
/// The stamp is held to the leg's as [`check_stamp`] holds a metrics
/// file's, and the `d3d9.dll` and `mtld3d.so` images to the ones the leg's
/// timed rounds reported, when it has any: a shape run of another build than
/// the numbers it sits beside would compare that build's passes.
///
/// # Errors
///
/// Returns a message when the log names no build, another stamp, or other
/// images than the timed rounds.
pub fn check_shape_build(
    log: &Path,
    identity: &Identity,
    spec: &LegSpec,
    timed: Option<&Images>,
) -> Result<(), String> {
    check_layer(
        log,
        identity.layer.as_deref(),
        spec,
        "no d3d9.dll load line names the layer's build",
    )?;
    let Some(timed) = timed else {
        return Ok(());
    };
    let ran = Images {
        layer: identity
            .layer_image
            .clone()
            .unwrap_or_else(|| UNKNOWN_IMAGE.to_owned()),
        unix: identity
            .unix_image
            .clone()
            .unwrap_or_else(|| UNKNOWN_IMAGE.to_owned()),
    };
    if ran == *timed {
        return Ok(());
    }
    Err(format!(
        "{}: the shape run loaded d3d9.dll {} and mtld3d.so {}, the leg's timed rounds {} and \
         {}; the shape would be another build's",
        log.display(),
        ran.layer,
        ran.unix,
        timed.layer,
        timed.unix
    ))
}

/// Hold `layer`, the stamp a run names, to its leg's; `missing` says what is absent without one.
fn check_layer(
    path: &Path,
    layer: Option<&str>,
    spec: &LegSpec,
    missing: &str,
) -> Result<(), String> {
    match layer {
        Some(layer) if layer == spec.stamp => Ok(()),
        Some(layer) => Err(format!(
            "{}: the run loaded layer {layer}, the leg installed {}; the prefix or the Wine \
             tree is not the one the leg was built into, or the build is stale",
            path.display(),
            spec.stamp
        )),
        None => Err(format!("{}: {missing}", path.display())),
    }
}

/// The progress text of one metrics file: its benchmark and its median frame time.
///
/// A file without a frame time (the host benchmark's) shows its first
/// time metric instead.
fn progress(path: &Path, file: &MetricsFile) -> String {
    let bench = path
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(metrics::bench_of)
        .unwrap_or_default();
    let shown = file.metrics.get_key_value(PROGRESS_METRIC).or_else(|| {
        file.metrics
            .iter()
            .find(|(_, metric)| metric.class == Class::Time)
    });
    shown.map_or_else(
        || format!("{bench}: no {PROGRESS_METRIC}"),
        |(name, metric)| {
            format!(
                "{bench}: {name} {:.3} {}",
                metric.value,
                metric.unit.as_str()
            )
        },
    )
}

/// What a benchmark's timed rounds reported that its shape runs are checked against.
#[derive(Default)]
struct Timed {
    /// The benchmarks its metrics files are named after.
    benches: BTreeSet<String>,
    /// The images the base leg's rounds loaded.
    base: Option<Images>,
    /// The images the candidate leg's rounds loaded.
    cand: Option<Images>,
}

impl Timed {
    /// Take note of one file a round of `leg` wrote: its benchmark and the images it loaded.
    fn note(&mut self, leg: &Leg, path: &Path, file: &MetricsFile) {
        if let Some(bench) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(metrics::bench_of)
        {
            self.benches.insert(bench.to_owned());
        }
        let image = |key: &str| {
            file.meta
                .get(key)
                .cloned()
                .unwrap_or_else(|| UNKNOWN_IMAGE.to_owned())
        };
        let images = Images {
            layer: image("layer_image"),
            unix: image("layer_unix_image"),
        };
        match leg {
            Leg::Base => self.base = Some(images),
            Leg::Cand => self.cand = Some(images),
        }
    }

    /// The images `leg`'s rounds loaded, `None` before its first round.
    const fn images(&self, leg: &Leg) -> Option<&Images> {
        match leg {
            Leg::Base => self.base.as_ref(),
            Leg::Cand => self.cand.as_ref(),
        }
    }
}

/// What one benchmark process reported.
#[derive(Default)]
struct Outcome {
    results: Vec<TestResult>,
    notes: Vec<String>,
}

impl Outcome {
    /// The runner's notes about the process, one per line, for an error message.
    fn notes(&self) -> String {
        self.notes.iter().fold(String::new(), |mut out, note| {
            let _ = write!(out, "\n{note}");
            out
        })
    }
}

impl Report for Outcome {
    fn result(&mut self, result: TestResult) {
        self.results.push(result);
    }

    fn note(&mut self, note: &str) {
        self.notes.push(note.to_owned());
    }
}

#[cfg(test)]
mod tests;
