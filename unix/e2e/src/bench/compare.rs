//! Judge a finished A/B directory: pair the legs round by round and give every metric a verdict.
//!
//! Round `i` of the base and round `i` of the candidate ran back to back, so
//! each pair saw the same machine state and the comparison is made pair by
//! pair, never between the two legs' pooled numbers. Every rule turns a
//! metric's pairs into a value where more is worse whichever way the metric
//! is better, so one threshold serves both directions:
//!
//! - `time` and `noisy`: the ratio `cand / base` per pair. A regression is a
//!   median ratio above `1 + max(T, 3 sigma)`, sigma being 1.4826 times the
//!   MAD of the finite ratios, with at least 80 % of the pairs worse. `T` is
//!   8 % for a tail percentile (a name with `p99`), which moves more between
//!   runs, and 3 % otherwise. An improvement is the mirror image. A `time`
//!   metric's median difference must also exceed five steps of the
//!   resolution its value is printed with ([`time_step`]), since a value of
//!   a few steps moves by tens of percent when it crosses one. A metric
//!   whose base median is zero has no ratio and is judged by its median
//!   difference against a small absolute floor instead.
//! - `bytes`: the same with `T` at 3 % whatever the name, and the median
//!   difference must also exceed 4 MiB, since a few percent of a small
//!   footprint is allocator noise.
//! - `spikes`: the difference per pair. A regression is a median difference
//!   above `max(2, 3 MAD)`.
//! - `exact`: any pair that differs is a change, worse or better, and a
//!   change fails the comparison unless it is accepted by name, because the
//!   workload fixes these numbers and a change means the layer does
//!   different work.
//! - `info`: reported, never judged.
//!
//! Both legs run the candidate's benchmark binary, so a metric only the base
//! has was made to vanish by the candidate's layer: it fails the comparison
//! like an exact change unless it is accepted by name. One only the
//! candidate has (the `perf.*` metrics against a base without the `perf-kv`
//! line) is listed as added. A metric some rounds of a leg carry and others
//! do not is an error, except the three the `perf-kv` line may leave out of
//! a window ([`OPTIONAL_METRICS`]: the fault counts without a fault sample,
//! the pool's GPU copies without any): with one window a round those come
//! and go, and are reported as incomplete with a note, never judged. When
//! the legs measured perf windows of different lengths (`meta window_s`, a
//! base older than the 2 s interval), the rows whose value grows with the
//! span, tail percentiles, worst values and spike counts, are reported, not
//! judged, with a note.
//!
//! Rounds whose process started on a busy machine (see `machine`) are
//! listed as warnings above the notes; a warning never changes a verdict.
//!
//! Beside the numbers, each scene benchmark's untimed shape runs are
//! compared: the passes of its steady submission and every load/store
//! decision on them must be the same in both legs, and a difference fails
//! the comparison like an exact metric does unless `shape` or
//! `shape:<bench>` is accepted (see `shape`).
//!
//! Before any of that the directory has to be trustworthy: both legs hold
//! the same rounds, every benchmark wrote a file in every round of its leg,
//! the metrics keep their definitions, each leg ran one build of each kind
//! of benchmark binary throughout ([`Kind`]),
//! the two legs ran two different `d3d9.dll` images, and both ran the same
//! profile. Anything else is an error, exit code 2, not a verdict.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use super::{
    Leg, SAME_IMAGE_FILE, SHAPE_DIR, WINE_FILE, machine,
    metrics::{self, Class, Direction, Metric, MetricsFile, Unit},
    shape::{self, ShapeReport},
    stats::{MAD_SIGMA, mad, median},
};

/// The noise floor of a ratio-judged metric, as a fraction.
const RATIO_FLOOR: f64 = 0.03;

/// The noise floor of a ratio-judged tail percentile, which moves more from run to run.
const RATIO_FLOOR_TAIL: f64 = 0.08;

/// How many estimated standard deviations a median has to clear.
const SIGMA_FACTOR: f64 = 3.0;

/// The floor of a spike count's median difference, in events.
const SPIKE_FLOOR: f64 = 2.0;

/// The floor of a zero-base difference in a time: 0.1 ms, in the metric's own unit.
const ZERO_BASE_FLOOR_MS: f64 = 0.1;

/// How many steps of its printed resolution a `time` metric's median difference has to clear.
const RESOLUTION_STEPS: f64 = 5.0;

/// The step of a `perf.*` time in ms: the layer's `perf-kv` line prints three decimals.
const PERF_MS_STEP: f64 = 0.001;

/// The step of a benchmark's own time in ms: `Value::Ms` prints four decimals.
const MS_STEP: f64 = 0.0001;

/// The step of a time in us: the host emitter benchmark prints three decimals.
const US_STEP: f64 = 0.001;

/// The step of a time in ns: `Value::Ns` prints one decimal.
const NS_STEP: f64 = 0.1;

/// The prefix of the metrics a benchmark copies from the layer's `perf-kv` line.
const PERF_PREFIX: &str = "perf.";

/// The meta keys every metrics file has to carry for the sanity checks.
const REQUIRED_META: [&str; 5] = [
    "layer",
    "layer_image",
    "arch",
    "profile",
    "debug_assertions",
];

/// The meta keys naming a loaded image, each with the binary it names.
///
/// `layer_image` is required, `layer_unix_image` optional: a key absent in
/// both legs is no evidence either way.
const IMAGE_META: [(&str, &str); 2] = [
    ("layer_image", "d3d9.dll"),
    ("layer_unix_image", "mtld3d.so"),
];

/// The image value of a binary that carries no image ID.
const UNKNOWN_IMAGE: &str = "unknown";

/// The note of an A/A run whose legs loaded one image.
const SAME_IMAGE_NOTE: &str = "legs loaded identical binaries (A/A)";

/// The meta keys both legs have to agree on: comparing two profiles measures the profiles.
const MATCHING_META: [&str; 3] = ["arch", "profile", "debug_assertions"];

/// The meta key that names the kind of binary a metrics file came from.
///
/// Absent in the end-to-end benchmarks' files, which run the layer under
/// Wine; `host` in the host emitter benchmark's, a native binary of each
/// leg's own tree with an architecture and a profile of its own.
const KIND_META: &str = "kind";

/// The meta keys that may differ between the legs, and between the rounds of one leg.
///
/// `window_s` is the length of the layer's perf window the run measured,
/// which a base older than the 2 s interval sets at 5 s; the comparison
/// reads it apart (see [`SPAN_META`]).
/// They name the build or the run, not the workload: which binaries ran
/// (the release stamp and image IDs, the host emitter's version), the suite
/// configuration (which carries each run's own `log.dir`), and the clock's
/// calibration, which every process measures afresh. Every other key
/// defines what a benchmark did, so it has to be the same in every file of
/// that benchmark in both legs; a key a later benchmark adds is held to
/// that without a change here.
const RUN_META: [&str; 9] = [
    "window_s",
    "layer",
    "layer_image",
    "layer_unix_image",
    "host_image",
    "emitter",
    "config",
    "tsc_hz",
    "tsc_granularity_ns",
];

/// The meta key naming the seconds of the perf window a run measured, `none` without one.
///
/// When the legs measured windows of different lengths, the metrics whose
/// value grows with the span (see [`span_scaled`]) are reported, not judged.
const SPAN_META: &str = "window_s";

/// The metrics a round may lack, from the keys the `perf-kv` line may leave out of a window.
///
/// `docs/ARCHITECTURE.md` names the three keys: the fault counts, absent
/// from a window without a fault sample, and the GPU copies of the vertex
/// and index buffer pool, absent when it has none to count. A benchmark
/// reads one window a round, so a round lacks them now and then. Any other
/// metric a round lacks is an error.
const OPTIONAL_METRICS: [&str; 3] = [
    "perf.faults_minor_pf",
    "perf.faults_major_pf",
    "perf.vbib_gpu_copy_pf",
];

/// The meta key a benchmark of a real shader cache carries, naming the cache.
///
/// A build that cannot read the cache's format writes no file for it, so
/// such a benchmark may be in one leg alone.
const CORPUS_META: &str = "corpus";

/// The meta keys every host benchmark file has to carry: it loads no layer image.
const HOST_REQUIRED_META: [&str; 4] = ["layer", "arch", "profile", "debug_assertions"];

/// The image key of the host benchmark and the binary it names.
const HOST_IMAGE_META: [(&str, &str); 1] = [("host_image", "emit_corpus")];

/// The end-to-end benchmarks, which run the layer under Wine.
const LAYER_KIND: Kind = Kind {
    tag: None,
    label: "layer",
    required: &REQUIRED_META,
    images: &IMAGE_META,
    distinct_images: true,
};

/// The host emitter benchmark, a native binary of each leg's own tree.
const HOST_KIND: Kind = Kind {
    tag: Some("host"),
    label: "host",
    required: &HOST_REQUIRED_META,
    images: &HOST_IMAGE_META,
    distinct_images: false,
};

/// The kinds of benchmark binary, each with the checks its builds get.
const KINDS: [Kind; 2] = [LAYER_KIND, HOST_KIND];

/// A kind of benchmark binary, told apart by the `kind` meta value its files carry.
///
/// Each kind's files are checked among themselves: a leg runs one build of
/// each kind throughout, and the two legs run builds of one profile. The
/// kinds differ in what names a build. The end-to-end files name the images
/// the layer loaded, which must differ between the legs, since each leg is
/// told apart from the other only by the layer it installed. The host files
/// name the benchmark binary itself, which each leg builds out of its own
/// tree and runs by path, and which a change outside the code it links
/// leaves byte for byte the same, so one image in both legs is a note there.
pub struct Kind {
    /// The `kind` meta value, `None` for files that carry none.
    tag: Option<&'static str>,
    /// How the report names the kind.
    label: &'static str,
    /// The meta keys each of its files has to carry.
    required: &'static [&'static str],
    /// The meta keys naming an image, each with the binary it names.
    images: &'static [(&'static str, &'static str)],
    /// Whether one image in both legs outside an A/A run is an error rather than a note.
    distinct_images: bool,
}

impl Kind {
    /// Whether `file` is of this kind.
    fn holds(&self, file: &MetricsFile) -> bool {
        file.meta.get(KIND_META).map(String::as_str) == self.tag
    }

    /// The files of this kind among a leg's rounds.
    fn files<'a>(&self, rounds: &'a [BTreeMap<String, Loaded>]) -> Vec<&'a Loaded> {
        rounds
            .iter()
            .flat_map(BTreeMap::values)
            .filter(|loaded| self.holds(&loaded.file))
            .collect()
    }
}

/// What changes a comparison's verdicts beyond the numbers.
#[derive(Debug, Default)]
pub struct Options {
    /// Exact metrics whose change is expected, by name; `shape` or `shape:<bench>` for shapes.
    pub accept: Vec<String>,
    /// Both legs build one commit from a clean tree, so they may load one image.
    pub allow_same_image: bool,
}

/// What became of one metric.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Within the noise.
    Neutral,
    /// Worse beyond the noise.
    Regression,
    /// Better beyond the noise.
    Improvement,
    /// An exact metric that moved.
    Changed {
        /// Some pair moved the worse way.
        worse: bool,
        /// Its name was given to `--accept`.
        accepted: bool,
    },
    /// An `info` metric, reported only.
    Info,
    /// Only the candidate has it.
    Added,
    /// Only the base has it: the candidate's layer made it vanish, since both legs run one binary.
    Removed {
        /// Its name was given to `--accept`.
        accepted: bool,
    },
    /// Some rounds of a leg have it and others do not, so it has no pairs to judge.
    Incomplete,
}

impl Verdict {
    /// The verdict column of the report.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Neutral => "ok",
            Self::Regression => "REGRESSION",
            Self::Improvement => "improved",
            Self::Changed { accepted: true, .. } => "changed, accepted",
            Self::Changed {
                worse: true,
                accepted: false,
            } => "CHANGED (worse)",
            Self::Changed {
                worse: false,
                accepted: false,
            } => "CHANGED (better)",
            Self::Info => "info",
            Self::Added => "added",
            Self::Removed { accepted: true } => "removed, accepted",
            Self::Removed { accepted: false } => "REMOVED",
            Self::Incomplete => "incomplete, not judged",
        }
    }

    /// Whether this verdict fails the comparison.
    #[must_use]
    pub const fn fails(&self) -> bool {
        matches!(
            self,
            Self::Regression
                | Self::Changed {
                    accepted: false,
                    ..
                }
                | Self::Removed { accepted: false }
        )
    }
}

/// One metric's row in a benchmark's table.
#[derive(Debug)]
pub struct Row {
    pub metric: String,
    pub base: String,
    pub cand: String,
    pub change: String,
    pub noise: String,
    pub verdict: Verdict,
}

/// One benchmark's part of the report.
#[derive(Debug)]
pub struct BenchReport {
    pub bench: String,
    pub rows: Vec<Row>,
}

/// The whole judgement of an A/B directory.
#[derive(Debug)]
pub struct Comparison {
    /// What was compared: the two builds, the profile, the pairs.
    pub header: Vec<String>,
    pub benches: Vec<BenchReport>,
    /// Each benchmark's pass shape in the two legs.
    pub shapes: Vec<ShapeReport>,
    /// Remarks that are no verdict, such as an accepted name nothing matched.
    pub notes: Vec<String>,
    /// Rounds that started on a busy machine, which never change a verdict.
    pub warnings: Vec<String>,
}

impl Comparison {
    /// Whether any verdict fails the comparison, a changed pass shape included.
    #[must_use]
    pub fn failed(&self) -> bool {
        self.rows().any(|row| row.verdict.fails()) || self.shapes.iter().any(ShapeReport::fails)
    }

    fn rows(&self) -> impl Iterator<Item = &Row> {
        self.benches.iter().flat_map(|bench| bench.rows.iter())
    }

    /// The report: the header, a table per benchmark, the notes and the summary line.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        for line in &self.header {
            let _ = writeln!(out, "{line}");
        }
        for bench in &self.benches {
            out.push('\n');
            let _ = writeln!(out, "== {}", bench.bench);
            render_table(&mut out, &bench.rows);
        }
        for shape in &self.shapes {
            out.push('\n');
            let _ = writeln!(out, "== shape {}", shape.bench);
            let _ = writeln!(out, "{}", shape.summary);
            if shape.changed() {
                for line in &shape.diff {
                    let _ = writeln!(out, "  {line}");
                }
                let verdict = if shape.accepted {
                    "shape changed, accepted"
                } else {
                    "SHAPE CHANGE"
                };
                let _ = writeln!(out, "{verdict}");
            } else {
                let _ = writeln!(out, "shape unchanged");
            }
        }
        if !self.warnings.is_empty() {
            out.push('\n');
            for warning in &self.warnings {
                let _ = writeln!(out, "WARNING: {warning}");
            }
        }
        if !self.notes.is_empty() {
            out.push('\n');
            for note in &self.notes {
                let _ = writeln!(out, "note: {note}");
            }
        }
        out.push('\n');
        out.push_str(&self.summary());
        out.push('\n');
        out
    }

    /// The one-line summary that ends the report.
    #[must_use]
    pub fn summary(&self) -> String {
        let count =
            |test: fn(&Verdict) -> bool| self.rows().filter(|row| test(&row.verdict)).count();
        let judged = self.benches.len();
        let regressions = count(|v| *v == Verdict::Regression);
        let improvements = count(|v| *v == Verdict::Improvement);
        let changes = count(|v| matches!(v, Verdict::Changed { .. }));
        let accepted = count(|v| matches!(v, Verdict::Changed { accepted: true, .. }));
        let added = count(|v| *v == Verdict::Added);
        let removed = count(|v| matches!(v, Verdict::Removed { .. }));
        let incomplete = count(|v| *v == Verdict::Incomplete);
        let verdict = if self.failed() { "FAIL" } else { "PASS" };
        let shapes_changed = self.shapes.iter().filter(|s| s.changed()).count();
        let shapes_accepted = self
            .shapes
            .iter()
            .filter(|s| s.changed() && s.accepted)
            .count();
        format!(
            "bench-compare: {verdict}: {judged} benchmarks, {} metrics: {regressions} regressed, \
             {improvements} improved, {changes} exact changed ({accepted} accepted), {added} \
             added, {removed} removed, {incomplete} incomplete; {shapes_changed} of {} shapes \
             changed ({shapes_accepted} \
             accepted)",
            self.rows().count(),
            self.shapes.len()
        )
    }
}

/// Judge the A/B directory `dir`, print the report, and write it to `report` too.
///
/// # Errors
///
/// Returns a message when the directory cannot be read or trusted, or the
/// report cannot be written.
pub fn judge_dir(dir: &Path, options: &Options, report: Option<&Path>) -> Result<ExitCode, String> {
    let comparison = evaluate(dir, options)?;
    let text = comparison.render();
    print!("{text}");
    if let Some(path) = report {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        fs::write(path, &text).map_err(|e| format!("{}: {e}", path.display()))?;
        println!("bench-compare: report written to {}", path.display());
    }
    Ok(if comparison.failed() {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    })
}

/// Judge the A/B directory `dir`.
///
/// `options.allow_same_image` also holds when the directory carries the
/// file `bench-ab` leaves for a run of one clean commit against itself.
///
/// # Errors
///
/// Returns a message when the directory cannot be read, a metrics file is
/// malformed, the rounds do not pair up, or the builds are not the ones an
/// A/B comparison needs.
pub fn evaluate(dir: &Path, options: &Options) -> Result<Comparison, String> {
    let base = load_leg(&dir.join(Leg::Base.dir()))?;
    let cand = load_leg(&dir.join(Leg::Cand.dir()))?;
    if base.len() != cand.len() {
        return Err(format!(
            "mismatched rounds in {}: base has {}, cand has {}",
            dir.display(),
            base.len(),
            cand.len()
        ));
    }
    let allow_same_image = options.allow_same_image || dir.join(SAME_IMAGE_FILE).exists();
    check_kinds(&base, &cand)?;
    let mut kinds = Vec::new();
    for kind in &KINDS {
        if let Some(builds) = check_builds(&base, &cand, allow_same_image, kind)? {
            kinds.push((kind, builds));
        }
    }
    if kinds.is_empty() {
        return Err(format!("{}: neither leg has a metrics file", dir.display()));
    }
    let mut comparison = compare(&base, &cand, options)?;
    let shapes = shape::compare_dir(dir, &options.accept)?;
    if shapes.present {
        check_shape_runs(&base[0], &shapes.reports, &mut comparison.notes)?;
    }
    comparison.shapes = shapes.reports;
    comparison.warnings = machine::warnings(dir, &[Leg::Base.dir(), Leg::Cand.dir()], base.len());
    comparison.notes.extend(shapes.notes);
    let mut header = vec![format!("bench-compare: {}", dir.display())];
    for (kind, builds) in kinds {
        comparison.notes.extend(builds.notes);
        let build = format!(
            "{} profile, debug assertions {}, {}",
            builds.profile, builds.debug_assertions, builds.arch
        );
        if kind.tag.is_none() {
            header.push(format!(
                "layer: base {}   cand {}",
                builds.base_layer, builds.cand_layer
            ));
            header.push(format!(
                "images (base / cand): {}",
                builds.images.join("; ")
            ));
            header.push(format!("{build}; {} round pairs", base.len()));
        } else {
            header.push(format!(
                "{} benchmarks: base {}   cand {}; images (base / cand): {}; {build}",
                kind.label,
                builds.base_layer,
                builds.cand_layer,
                builds.images.join("; ")
            ));
        }
    }
    comparison.header = header;
    comparison.header.extend([
        format!(
            "wine: {}",
            fs::read_to_string(dir.join(WINE_FILE))
                .map_or_else(|_| "not recorded".to_owned(), |wine| wine.trim().to_owned())
        ),
        "change: + is worse whichever way the metric is better; noise: sigma of the pair ratios, \
         the MAD of spike differences, or how many exact pairs differ"
            .to_owned(),
    ]);
    Ok(comparison)
}

/// A metrics file and where it came from.
#[derive(Debug)]
pub struct Loaded {
    pub path: PathBuf,
    pub file: MetricsFile,
}

/// Read one leg's directory: rounds `0..N`, each a directory of `bench-<name>.metrics`.
///
/// What it returns holds, per round, each benchmark's file by benchmark name.
/// The `shape` directory beside the rounds holds the untimed shape runs,
/// whose numbers are no round's.
///
/// # Errors
///
/// Returns a message when the directory is missing or unreadable, a
/// subdirectory is not a round number, the rounds are not `0..N`, or a
/// metrics file cannot be read or parsed.
pub fn load_leg(dir: &Path) -> Result<Vec<BTreeMap<String, Loaded>>, String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut rounds: BTreeMap<usize, PathBuf> = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name();
        if name == SHAPE_DIR {
            continue;
        }
        let round = name
            .to_str()
            .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
            .and_then(|n| n.parse::<usize>().ok())
            .ok_or_else(|| format!("{}: not a round number", path.display()))?;
        rounds.insert(round, path);
    }
    if rounds.is_empty() {
        return Err(format!("{}: no rounds", dir.display()));
    }
    if rounds.keys().copied().ne(0..rounds.len()) {
        let found: Vec<String> = rounds.keys().map(ToString::to_string).collect();
        return Err(format!(
            "{}: the rounds are {}, not 0..{}",
            dir.display(),
            found.join(", "),
            rounds.len()
        ));
    }
    rounds.values().map(|round| load_round(round)).collect()
}

/// Read every `bench-<name>.metrics` in one round's directory.
fn load_round(dir: &Path) -> Result<BTreeMap<String, Loaded>, String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut files = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = entry.file_name();
        let Some(bench) = name.to_str().and_then(metrics::bench_of) else {
            continue;
        };
        let path = entry.path();
        let file = metrics::read(&path)?;
        files.insert(bench.to_owned(), Loaded { path, file });
    }
    Ok(files)
}

/// What the two legs ran, once the checks passed.
#[derive(Debug)]
pub struct Builds {
    pub base_layer: String,
    pub cand_layer: String,
    /// Each image key both legs carry: `<binary> <base image> / <cand image>`.
    pub images: Vec<String>,
    pub profile: String,
    pub debug_assertions: String,
    pub arch: String,
    /// What the checks could not establish, for the report.
    pub notes: Vec<String>,
}

/// Check that every metrics file names a kind this comparison knows.
///
/// # Errors
///
/// Returns a message naming the first file whose `kind` is none of [`KINDS`].
pub fn check_kinds(
    base: &[BTreeMap<String, Loaded>],
    cand: &[BTreeMap<String, Loaded>],
) -> Result<(), String> {
    let unknown = base
        .iter()
        .chain(cand)
        .flat_map(BTreeMap::values)
        .find(|loaded| !KINDS.iter().any(|kind| kind.holds(&loaded.file)));
    if let Some(loaded) = unknown {
        return Err(format!(
            "{}: meta {KIND_META} {:?} is no kind of benchmark this comparison knows",
            loaded.path.display(),
            loaded.file.meta.get(KIND_META).map_or("", String::as_str)
        ));
    }
    Ok(())
}

/// Check that each leg ran one build of `kind` throughout, and the legs two of one profile.
///
/// Within a leg every file of the kind has to name the same layer stamp,
/// image IDs, profile, debug-assertion state and architecture. Across the
/// legs the last three have to match, and for a kind with
/// `distinct_images` the image IDs have to differ: the two legs are
/// separate builds, so one image in both means one binary was loaded twice.
/// That holds for every image key both legs carry, while a leg carrying
/// one alone is an error. The release stamps may be equal, since a
/// candidate with uncommitted changes carries the stamp of the commit it
/// sits on. The one exception is a true A/A run, one commit against itself
/// from a clean tree, where a deterministic build gives both legs the same
/// image: `allow_same_image` says the refs are that, and equal stamps
/// confirm it. `None` when neither leg has a file of the kind.
///
/// # Errors
///
/// Returns a message naming the files or values that disagree, or the leg
/// that has files of the kind when the other has none.
pub fn check_builds(
    base: &[BTreeMap<String, Loaded>],
    cand: &[BTreeMap<String, Loaded>],
    allow_same_image: bool,
    kind: &Kind,
) -> Result<Option<Builds>, String> {
    let (base_files, cand_files) = (kind.files(base), kind.files(cand));
    match (base_files.is_empty(), cand_files.is_empty()) {
        (true, true) => return Ok(None),
        (false, false) => {}
        (base_empty, _) => {
            let (has, lacks) = if base_empty {
                (Leg::Cand, Leg::Base)
            } else {
                (Leg::Base, Leg::Cand)
            };
            return Err(format!(
                "the {} leg has {} benchmark files and the {} leg none: the legs did not run \
                 the same benchmarks",
                has.dir(),
                kind.label,
                lacks.dir()
            ));
        }
    }
    let base_meta = leg_meta(&Leg::Base, &base_files, kind.required)?;
    let cand_meta = leg_meta(&Leg::Cand, &cand_files, kind.required)?;
    for key in MATCHING_META {
        if base_meta[key] != cand_meta[key] {
            return Err(format!(
                "the legs ran different builds: meta {key} is {:?} in base, {:?} in cand; \
                 comparing two {key}s measures the {key}s, not the change",
                base_meta[key], cand_meta[key]
            ));
        }
    }
    let a_a = allow_same_image && base_meta["layer"] == cand_meta["layer"];
    let mut notes = Vec::new();
    let mut images = Vec::new();
    for &(key, binary) in kind.images {
        let pair = (
            leg_optional(&Leg::Base, &base_files, key)?,
            leg_optional(&Leg::Cand, &cand_files, key)?,
        );
        let same = if a_a {
            SameImage::AA
        } else if kind.distinct_images {
            SameImage::Error
        } else {
            SameImage::Unchanged
        };
        check_image(key, binary, &pair, &same, &mut notes)?;
        if let (Some(base_image), Some(cand_image)) = pair {
            images.push(format!("{binary} {base_image} / {cand_image}"));
        }
    }
    Ok(Some(Builds {
        base_layer: base_meta["layer"].clone(),
        cand_layer: cand_meta["layer"].clone(),
        images,
        profile: base_meta["profile"].clone(),
        debug_assertions: base_meta["debug_assertions"].clone(),
        arch: base_meta["arch"].clone(),
        notes,
    }))
}

/// What one image in both legs means.
enum SameImage {
    /// A true A/A run, where a deterministic build gives both legs one image: a note.
    AA,
    /// A kind whose binary a change may leave as it was: a note.
    Unchanged,
    /// A kind whose legs must run two binaries: an error.
    Error,
}

/// Check one image key across the legs, see [`check_builds`].
fn check_image(
    key: &str,
    binary: &str,
    pair: &(Option<String>, Option<String>),
    same: &SameImage,
    notes: &mut Vec<String>,
) -> Result<(), String> {
    let (base_image, cand_image) = match pair {
        (None, None) => return Ok(()),
        (Some(base_image), Some(cand_image)) => (base_image, cand_image),
        (Some(_), None) | (None, Some(_)) => {
            return Err(format!(
                "meta {key} is in one leg's metrics only: the legs did not run the same \
                 benchmark binary"
            ));
        }
    };
    if base_image == UNKNOWN_IMAGE || cand_image == UNKNOWN_IMAGE {
        notes.push(format!(
            "a leg's {binary} carries no image ID, so nothing shows the legs ran two builds of it"
        ));
    } else if base_image == cand_image {
        let note = match same {
            SameImage::AA => SAME_IMAGE_NOTE.to_owned(),
            SameImage::Unchanged => format!(
                "both legs ran {binary} image {base_image}: the change leaves the code it links \
                 unchanged"
            ),
            SameImage::Error => {
                return Err(format!(
                    "both legs loaded {binary} image {base_image}: one binary ran twice, so one \
                     leg did not run the build it was meant to"
                ));
            }
        };
        if !notes.contains(&note) {
            notes.push(note);
        }
    }
    Ok(())
}

/// An optional meta value of one leg, checked to be the same, or absent, in every file.
fn leg_optional(leg: &Leg, files: &[&Loaded], key: &str) -> Result<Option<String>, String> {
    let mut first: Option<(Option<&String>, &Path)> = None;
    for loaded in files {
        let value = loaded.file.meta.get(key);
        match first {
            None => first = Some((value, &loaded.path)),
            Some((seen, seen_path)) if seen != value => {
                let shown = |v: Option<&String>| {
                    v.map_or_else(|| "absent".to_owned(), |v| format!("{v:?}"))
                };
                return Err(format!(
                    "the {} leg did not run one build: meta {key} is {} in {} and {} in {}",
                    leg.dir(),
                    shown(seen),
                    seen_path.display(),
                    shown(value),
                    loaded.path.display()
                ));
            }
            Some(_) => {}
        }
    }
    Ok(first.and_then(|(value, _)| value.cloned()))
}

/// The required meta values of one leg, checked to be the same in every file.
fn leg_meta(
    leg: &Leg,
    files: &[&Loaded],
    required: &[&'static str],
) -> Result<BTreeMap<&'static str, String>, String> {
    let mut seen: BTreeMap<&'static str, (String, &Path)> = BTreeMap::new();
    for loaded in files {
        for &key in required {
            let value = loaded
                .file
                .meta
                .get(key)
                .ok_or_else(|| format!("{}: no meta {key} line", loaded.path.display()))?;
            match seen.get(key) {
                None => {
                    seen.insert(key, (value.clone(), &loaded.path));
                }
                Some((first, first_path)) if first != value => {
                    return Err(format!(
                        "the {} leg did not run one build: meta {key} is {first:?} in {} and \
                         {value:?} in {}",
                        leg.dir(),
                        first_path.display(),
                        loaded.path.display()
                    ));
                }
                Some(_) => {}
            }
        }
    }
    Ok(seen
        .into_iter()
        .map(|(key, (value, _))| (key, value))
        .collect())
}

/// Check that every benchmark whose metrics declare `shape` lines has a shape run.
///
/// `round` is one round of the base leg. A benchmark without `shape` lines
/// gets no shape run by design, which the report notes.
///
/// # Errors
///
/// Returns a message naming a benchmark that declares its frame and has no
/// shape run: the run did not finish.
fn check_shape_runs(
    round: &BTreeMap<String, Loaded>,
    reports: &[ShapeReport],
    notes: &mut Vec<String>,
) -> Result<(), String> {
    for (bench, loaded) in round {
        let has_run = reports
            .iter()
            .any(|report| report.bench.split('+').any(|name| name == bench));
        match (loaded.file.shape.is_empty(), has_run) {
            (false, false) => {
                return Err(format!(
                    "incomplete run: benchmark {bench} declares shape lines and has no shape run"
                ));
            }
            (true, false) => notes.push(format!(
                "{bench}: no shape run; its metrics declare no shape lines"
            )),
            (_, true) => {}
        }
    }
    Ok(())
}

/// Compare the two legs benchmark by benchmark and metric by metric.
///
/// # Errors
///
/// Returns a message when a benchmark is missing from some rounds of a leg
/// or from the other leg altogether, or a metric changes its definition or
/// comes and goes between rounds.
pub fn compare(
    base: &[BTreeMap<String, Loaded>],
    cand: &[BTreeMap<String, Loaded>],
    options: &Options,
) -> Result<Comparison, String> {
    let base_benches = leg_benches(&Leg::Base, base)?;
    let cand_benches = leg_benches(&Leg::Cand, cand)?;
    let mut benches = Vec::new();
    let mut notes = Vec::new();
    for bench in base_benches.symmetric_difference(&cand_benches) {
        let (ran, rounds) = if base_benches.contains(bench) {
            ("base", base)
        } else {
            ("cand", cand)
        };
        if !cache_corpus(bench, rounds) {
            return Err(format!(
                "incomplete run: benchmark {bench} ran only in the {ran} leg; both legs have to \
                 run every benchmark for the run to be judged"
            ));
        }
        notes.push(format!(
            "{bench} skipped: only the {ran} leg's build could read that shader cache (a cache \
             format one of the builds does not read), so there is nothing to pair"
        ));
    }
    for bench in base_benches.intersection(&cand_benches) {
        check_workload(bench, base, cand)?;
        let rows = bench_rows(bench, base, cand, options, &mut notes)?;
        for row in rows.iter().filter(|row| row.verdict == Verdict::Incomplete) {
            notes.push(format!(
                "{bench}: {} is in some rounds of a leg and not in others ({}), as a key the \
                 perf-kv line may leave out can be; it is not judged",
                row.metric, row.change
            ));
        }
        benches.push(BenchReport {
            bench: bench.clone(),
            rows,
        });
    }
    for name in options
        .accept
        .iter()
        .filter(|name| !shape::is_accept_name(name))
    {
        let matched = benches.iter().flat_map(|b| b.rows.iter()).any(|row| {
            row.metric == *name
                && matches!(
                    row.verdict,
                    Verdict::Changed { accepted: true, .. } | Verdict::Removed { accepted: true }
                )
        });
        if !matched {
            notes.push(format!(
                "--accept {name}: no exact metric of that name changed or was removed"
            ));
        }
    }
    Ok(Comparison {
        header: Vec::new(),
        benches,
        shapes: Vec::new(),
        notes,
        warnings: Vec::new(),
    })
}

/// Whether `bench` measures a real shader cache, going by its files in `rounds`.
///
/// Its files carry [`CORPUS_META`]. The host emitter and the cold-start
/// benchmark skip a cache whose format their build does not read and write
/// no file for it, so a cache that a format change between the two builds
/// makes readable to one leg only leaves its benchmark in that leg alone.
fn cache_corpus(bench: &str, rounds: &[BTreeMap<String, Loaded>]) -> bool {
    let mut files = rounds
        .iter()
        .filter_map(|round| round.get(bench))
        .peekable();
    files.peek().is_some() && files.all(|loaded| loaded.file.meta.contains_key(CORPUS_META))
}

/// Check that `bench` ran one workload in every file of both legs.
///
/// Every meta key outside [`RUN_META`] defines the workload, such as the
/// entries the benchmark's harness added (`config_entries`) or the path it
/// took (`depth_path`), and has the same value, or is absent, in every file
/// of the benchmark.
///
/// # Errors
///
/// Returns a message naming the key and the two values that differ.
pub fn check_workload(
    bench: &str,
    base: &[BTreeMap<String, Loaded>],
    cand: &[BTreeMap<String, Loaded>],
) -> Result<(), String> {
    let files: Vec<(&Leg, &Loaded)> = [(&Leg::Base, base), (&Leg::Cand, cand)]
        .into_iter()
        .flat_map(|(leg, rounds)| {
            rounds
                .iter()
                .filter_map(|round| round.get(bench))
                .map(move |loaded| (leg, loaded))
        })
        .collect();
    let keys: BTreeSet<&String> = files
        .iter()
        .flat_map(|(_, loaded)| loaded.file.meta.keys())
        .filter(|key| !RUN_META.contains(&key.as_str()))
        .collect();
    let shown =
        |value: Option<&String>| value.map_or_else(|| "absent".to_owned(), |v| format!("{v:?}"));
    for key in keys {
        let Some(((first_leg, first), rest)) = files.split_first().map(|(f, r)| (*f, r)) else {
            continue;
        };
        let expected = first.file.meta.get(key);
        if let Some((leg, loaded)) = rest
            .iter()
            .find(|(_, loaded)| loaded.file.meta.get(key) != expected)
        {
            let within = if *leg == first_leg {
                format!("the {} leg changed it between rounds", leg.dir())
            } else {
                "the legs ran different workloads".to_owned()
            };
            return Err(format!(
                "{bench}: meta {key} is {} in {} and {} in {}; {within}, so the numbers \
                 cannot be compared",
                shown(expected),
                first.path.display(),
                shown(loaded.file.meta.get(key)),
                loaded.path.display()
            ));
        }
    }
    Ok(())
}

/// The benchmarks of one leg, each checked to have a file in every round.
fn leg_benches(leg: &Leg, rounds: &[BTreeMap<String, Loaded>]) -> Result<BTreeSet<String>, String> {
    let all: BTreeSet<String> = rounds.iter().flat_map(BTreeMap::keys).cloned().collect();
    for (index, round) in rounds.iter().enumerate() {
        if let Some(missing) = all.iter().find(|bench| !round.contains_key(*bench)) {
            return Err(format!(
                "mismatched rounds: {}/{index} has no bench-{missing}.metrics, which other \
                 rounds of that leg have",
                leg.dir()
            ));
        }
    }
    Ok(all)
}

/// The rows of one benchmark both legs ran.
fn bench_rows(
    bench: &str,
    base: &[BTreeMap<String, Loaded>],
    cand: &[BTreeMap<String, Loaded>],
    options: &Options,
    notes: &mut Vec<String>,
) -> Result<Vec<Row>, String> {
    let base_series = leg_series(bench, base)?;
    let cand_series = leg_series(bench, cand)?;
    let (base_spans, cand_spans) = (spans(bench, base), spans(bench, cand));
    let spans_differ = base_spans != cand_spans;
    if spans_differ {
        notes.push(format!(
            "{bench}: the legs measured perf windows of different lengths (base {}, cand {} \
             s), so its p99, max and spike rows, which grow with the span, are reported and \
             not judged",
            listed(&base_spans),
            listed(&cand_spans)
        ));
    }
    let names: BTreeSet<&String> = base_series
        .keys()
        .chain(cand_series.keys())
        .copied()
        .collect();
    let mut rows = Vec::new();
    for name in names {
        let (in_base, in_cand) = (base_series.get(name), cand_series.get(name));
        let partial = |series: Option<&(&Metric, Vec<f64>)>, rounds: usize| {
            series.is_some_and(|(_, values)| values.len() != rounds)
        };
        let optional = OPTIONAL_METRICS.contains(&name.as_str());
        let partly = partial(in_base, base.len()) || partial(in_cand, cand.len());
        if partly && !optional {
            let leg = if partial(in_base, base.len()) {
                "base"
            } else {
                "cand"
            };
            return Err(format!(
                "{bench}: metric {name} is in some rounds of the {leg} leg and not in others; \
                 the rounds of a leg must carry the same metrics (only {} may come and go)",
                OPTIONAL_METRICS.join(", ")
            ));
        }
        if partly || (optional && (in_base.is_none() || in_cand.is_none())) {
            let counted = |series: Option<&(&Metric, Vec<f64>)>, rounds: usize, leg: &str| {
                format!(
                    "{} of {rounds} {leg} rounds",
                    series.map_or(0, |(_, values)| values.len())
                )
            };
            let shown = |series: Option<&(&Metric, Vec<f64>)>| {
                series.map_or_else(
                    || "-".to_owned(),
                    |(definition, values)| with_unit(median(values), definition),
                )
            };
            rows.push(Row {
                metric: name.clone(),
                base: shown(in_base),
                cand: shown(in_cand),
                change: format!(
                    "in {}, {}",
                    counted(in_base, base.len(), "base"),
                    counted(in_cand, cand.len(), "cand")
                ),
                noise: String::new(),
                verdict: Verdict::Incomplete,
            });
            continue;
        }
        let row = match (in_base, in_cand) {
            (Some((definition, base_values)), Some((cand_definition, cand_values))) => {
                if !definition.same_definition(cand_definition) {
                    return Err(format!(
                        "{bench}: metric {name} is {} in base and {} in cand; the legs cannot \
                         be compared on it",
                        definition.definition(),
                        cand_definition.definition()
                    ));
                }
                let accepted = options.accept.iter().any(|accepted| accepted == name);
                if spans_differ && span_scaled(name, definition) {
                    let mut row = judge(name, definition, base_values, cand_values, accepted);
                    "window lengths differ".clone_into(&mut row.noise);
                    row.verdict = Verdict::Info;
                    row
                } else {
                    judge(name, definition, base_values, cand_values, accepted)
                }
            }
            (Some((definition, values)), None) => {
                let accepted = options.accept.iter().any(|accepted| accepted == name);
                presence_row(name, definition, values, Verdict::Removed { accepted })
            }
            (None, Some((definition, values))) => {
                presence_row(name, definition, values, Verdict::Added)
            }
            (None, None) => continue,
        };
        rows.push(row);
    }
    Ok(rows)
}

/// One benchmark's metrics in one leg: each one's definition and its value in each round with it.
///
/// A metric some rounds lack has fewer values than the leg has rounds;
/// `bench_rows` reports it as [`Verdict::Incomplete`].
fn leg_series<'a>(
    bench: &str,
    rounds: &'a [BTreeMap<String, Loaded>],
) -> Result<BTreeMap<&'a String, (&'a Metric, Vec<f64>)>, String> {
    let mut series: BTreeMap<&String, (&Metric, Vec<f64>)> = BTreeMap::new();
    for round in rounds {
        let loaded = &round[bench];
        for (name, metric) in &loaded.file.metrics {
            let entry = series.entry(name).or_insert_with(|| (metric, Vec::new()));
            if !entry.0.same_definition(metric) {
                return Err(format!(
                    "{}: metric {name} is {} here and {} in an earlier round",
                    loaded.path.display(),
                    metric.definition(),
                    entry.0.definition()
                ));
            }
            entry.1.push(metric.value);
        }
    }
    Ok(series)
}

/// The window lengths the rounds of one leg name for `bench` (`window_s`), empty for files without.
fn spans(bench: &str, rounds: &[BTreeMap<String, Loaded>]) -> BTreeSet<String> {
    rounds
        .iter()
        .filter_map(|round| round.get(bench)?.file.meta.get(SPAN_META).cloned())
        .collect()
}

/// A set of window lengths as a note lists them.
fn listed(spans: &BTreeSet<String>) -> String {
    if spans.is_empty() {
        "unnamed".to_owned()
    } else {
        spans.iter().cloned().collect::<Vec<_>>().join("/")
    }
}

/// Whether a metric's value grows with the span it was measured over.
///
/// A tail percentile, a worst value and a count of spikes all see more of
/// the rare slow frames in a longer span. An exact metric is fixed by the
/// workload whatever the span, and a median, a mean or a per-frame count is
/// not tied to it.
fn span_scaled(name: &str, definition: &Metric) -> bool {
    definition.class != Class::Exact
        && (definition.class == Class::Spikes
            || name.contains("p99")
            || name.rsplit('.').next() == Some("max"))
}

/// The row of a metric only one leg has.
fn presence_row(name: &str, definition: &Metric, values: &[f64], verdict: Verdict) -> Row {
    let shown = with_unit(median(values), definition);
    let (base, cand) = if verdict == Verdict::Added {
        ("-".to_owned(), shown)
    } else {
        (shown, "-".to_owned())
    };
    Row {
        metric: name.to_owned(),
        base,
        cand,
        change: String::new(),
        noise: String::new(),
        verdict,
    }
}

/// Judge one metric over its round pairs.
///
/// `base[i]` and `cand[i]` are round `i` of either leg; `accepted` says the
/// metric's name was given to `--accept`.
#[must_use]
pub fn judge(name: &str, definition: &Metric, base: &[f64], cand: &[f64], accepted: bool) -> Row {
    let mut row = Row {
        metric: name.to_owned(),
        base: with_unit(median(base), definition),
        cand: with_unit(median(cand), definition),
        change: String::new(),
        noise: String::new(),
        verdict: Verdict::Info,
    };
    let lower = definition.direction == Direction::Lower;
    let pairs = base.iter().zip(cand);
    let worse_by: Vec<f64> = pairs
        .clone()
        .map(|(&b, &c)| if lower { c - b } else { b - c })
        .collect();
    match definition.class {
        Class::Info => {}
        Class::Time | Class::Noisy | Class::Bytes => {
            let ratios: Vec<f64> = pairs
                .map(|(&b, &c)| if lower { ratio(c, b) } else { ratio(b, c) })
                .collect();
            if median(base) == 0.0 {
                judge_zero_base(&mut row, definition, &worse_by);
                return row;
            }
            let center = median(&ratios);
            // A pair on a zero base has an infinite ratio: it counts toward
            // the median and the 80 % rule, but a spread has to be finite.
            let finite: Vec<f64> = ratios.iter().copied().filter(|r| r.is_finite()).collect();
            let spread = mad(&finite);
            let sigma = if spread.is_finite() {
                MAD_SIGMA * spread
            } else {
                0.0
            };
            let floor = if name.contains("p99") && definition.class != Class::Bytes {
                RATIO_FLOOR_TAIL
            } else {
                RATIO_FLOOR
            };
            let threshold = floor.max(SIGMA_FACTOR * sigma);
            let worse = ratios.iter().filter(|&&r| r > 1.0).count();
            let better = ratios.iter().filter(|&&r| r < 1.0).count();
            if center.is_nan() {
                row.change.push_str("undefined");
                row.verdict = Verdict::Neutral;
                return row;
            }
            let mut regressed = center > 1.0 + threshold && most(worse, ratios.len());
            let mut improved = center < 1.0 - threshold && most(better, ratios.len());
            row.change = format!("{:+.2}%", (center - 1.0) * 100.0);
            row.noise = format!("sigma {:.2}%", sigma * 100.0);
            if definition.class == Class::Time
                && let Some(step) = time_step(name, &definition.unit)
            {
                let delta = median(&worse_by);
                let past = past_floor(delta, step);
                if (regressed || improved) && !past {
                    let _ = write!(
                        row.change,
                        " ({:+} {}, within five steps of {})",
                        number(delta),
                        definition.unit.as_str(),
                        number(step)
                    );
                }
                regressed &= past && delta > 0.0;
                improved &= past && delta < 0.0;
            }
            if definition.class == Class::Bytes {
                let delta = median(&worse_by);
                let min = definition.unit.four_mib().unwrap_or(0.0);
                regressed &= delta > min;
                improved &= delta < -min;
                let _ = write!(
                    row.change,
                    " ({:+} {})",
                    number(delta),
                    definition.unit.as_str()
                );
            }
            row.verdict = verdict(regressed, improved);
        }
        Class::Spikes => {
            let center = median(&worse_by);
            let spread = mad(&worse_by);
            let threshold = SPIKE_FLOOR.max(SIGMA_FACTOR * spread);
            row.change = format!("{:+}", number(center));
            row.noise = format!("MAD {}", number(spread));
            row.verdict = verdict(center > threshold, center < -threshold);
        }
        Class::Exact => {
            let differ = worse_by.iter().filter(|&&d| d != 0.0).count();
            let worse = worse_by.iter().any(|&d| d > 0.0);
            row.change = format!("{:+}", number(median(&worse_by)));
            row.noise = format!("{differ}/{} pairs differ", worse_by.len());
            row.verdict = if differ == 0 {
                Verdict::Neutral
            } else {
                Verdict::Changed { worse, accepted }
            };
        }
    }
    row
}

/// Judge a ratio-judged metric whose base median is zero, which has no ratio to judge by.
///
/// The median difference decides, against a floor in the metric's unit
/// (see [`zero_base_floor`]), with the same 80 % rule as a ratio: a time
/// the base did not spend at all and the candidate spends a little of is
/// reported, and fails only past the floor.
fn judge_zero_base(row: &mut Row, definition: &Metric, worse_by: &[f64]) {
    let center = median(worse_by);
    let floor = zero_base_floor(&definition.unit);
    let worse = worse_by.iter().filter(|&&d| d > 0.0).count();
    let better = worse_by.iter().filter(|&&d| d < 0.0).count();
    row.change = format!(
        "{:+} {} (zero base)",
        number(center),
        definition.unit.as_str()
    );
    row.noise = format!("floor {}", number(floor));
    row.verdict = verdict(
        center > floor && most(worse, worse_by.len()),
        center < -floor && most(better, worse_by.len()),
    );
}

/// The least difference a zero-base metric has to move by, in its own unit.
///
/// 0.1 ms for a time, two for a count (the spike floor), 0.03 for a ratio
/// (the ratio floor), and 4 MiB for memory.
fn zero_base_floor(unit: &Unit) -> f64 {
    match unit {
        Unit::Ms => ZERO_BASE_FLOOR_MS,
        Unit::Us => ZERO_BASE_FLOOR_MS * 1e3,
        Unit::Ns => ZERO_BASE_FLOOR_MS * 1e6,
        Unit::Count => SPIKE_FLOOR,
        Unit::Ratio => RATIO_FLOOR,
        Unit::Mib | Unit::Bytes => unit.four_mib().unwrap_or_default(),
    }
}

/// The resolution a `time` metric's value is printed with, in its own unit; `None` for no time.
///
/// A `time` metric's median difference has to exceed five of these steps
/// ([`past_floor`]), so that a value only a few steps large cannot pass the
/// ratio rule by crossing one step: a `perf-kv` time of 0.005 ms that reads
/// 0.004 ms in the candidate is 20 % better and one step apart. The
/// `perf.*` times come from that line, three decimals of a millisecond, so
/// their step is 0.001 ms; a benchmark's own times in ms have four decimals
/// (0.0001 ms), the host emitter's in us three (0.001 us) and the times in
/// ns one (0.1 ns).
#[must_use]
pub fn time_step(name: &str, unit: &Unit) -> Option<f64> {
    match unit {
        Unit::Ms if name.starts_with(PERF_PREFIX) => Some(PERF_MS_STEP),
        Unit::Ms => Some(MS_STEP),
        Unit::Us => Some(US_STEP),
        Unit::Ns => Some(NS_STEP),
        Unit::Count | Unit::Mib | Unit::Bytes | Unit::Ratio => None,
    }
}

/// Whether the difference `delta` is more than five steps of `step`, counted in whole steps.
///
/// The difference is rounded to the nearest step before it is compared, so
/// one printed as exactly five steps stays within the floor however the
/// subtraction rounds: 0.011 - 0.006 is 0.005000000000000001.
#[must_use]
pub fn past_floor(delta: f64, step: f64) -> bool {
    (delta.abs() / step).round() > RESOLUTION_STEPS
}

/// `numerator / denominator`, with two zeros equal and a zero denominator infinitely worse.
fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator != 0.0 {
        numerator / denominator
    } else if numerator != 0.0 {
        f64::INFINITY
    } else {
        1.0
    }
}

/// Whether `count` of `pairs` is at least 80 % of them.
const fn most(count: usize, pairs: usize) -> bool {
    count * 5 >= pairs * 4
}

const fn verdict(regressed: bool, improved: bool) -> Verdict {
    if regressed {
        Verdict::Regression
    } else if improved {
        Verdict::Improvement
    } else {
        Verdict::Neutral
    }
}

/// A value and its unit, the way the report shows a median.
fn with_unit(value: f64, definition: &Metric) -> String {
    format!("{} {}", number(value), definition.unit.as_str())
}

/// A value as the report shows it: a whole number plain, anything else to three places.
const fn number(value: f64) -> Number {
    Number(value)
}

/// A value formatted for the report, see [`number`].
struct Number(f64);

impl std::fmt::Display for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.0;
        let precision = if value.fract() == 0.0 { 0 } else { 3 };
        if f.sign_plus() {
            write!(f, "{value:+.precision$}")
        } else {
            write!(f, "{value:.precision$}")
        }
    }
}

/// Append one benchmark's table: a header and a row per metric, columns padded to fit.
fn render_table(out: &mut String, rows: &[Row]) {
    let header = ["metric", "base", "cand", "change", "noise", "verdict"];
    let cells: Vec<[&str; 6]> = rows
        .iter()
        .map(|row| {
            [
                row.metric.as_str(),
                row.base.as_str(),
                row.cand.as_str(),
                row.change.as_str(),
                row.noise.as_str(),
                row.verdict.label(),
            ]
        })
        .collect();
    let mut widths = header.map(str::len);
    for line in &cells {
        for (width, cell) in widths.iter_mut().zip(line) {
            *width = (*width).max(cell.len());
        }
    }
    for line in std::iter::once(&header).chain(&cells) {
        let mut text = String::new();
        for (index, (cell, width)) in line.iter().zip(widths).enumerate() {
            if index + 1 == line.len() {
                text.push_str(cell);
            } else {
                let _ = write!(text, "{cell:<width$}  ");
            }
        }
        let _ = writeln!(out, "{}", text.trim_end());
    }
}

#[cfg(test)]
mod tests;
