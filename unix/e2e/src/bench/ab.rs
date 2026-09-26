//! Run the benchmarks against the two legs, interleaved, and check every run as it lands.
//!
//! Each run is one process of the candidate's test binary running one
//! benchmark with libtest's `--ignored`, under the leg's Wine loader and
//! prefix and with `log.dir` pointed at `<out>/<leg>/<round>`, where the
//! benchmark writes its `bench-<name>.metrics`. For each benchmark, round
//! `r` runs the base first when `r` is even and the candidate first when it
//! is odd. A run that fails, writes no metrics file, or reports a layer
//! stamp other than its leg's ends the whole A/B run at once: every number
//! after it would be measured against the wrong build or none.

use std::{
    collections::BTreeMap,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    time::{Duration, SystemTime},
};

use super::{
    Leg, SAME_IMAGE_FILE, WINE_FILE,
    compare::{self, Options},
    metrics::{self, MetricsFile},
};
use crate::{
    attribute::{self, BinaryOutcome, Launcher as _, Report, TestResult, Verdict},
    binary::{WineLauncher, binary_name},
    select::{selected, test_id},
};

/// The metric a progress line shows, when the benchmark reports it.
const PROGRESS_METRIC: &str = "frame.p50";

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
    println!(
        "bench-ab: {} benchmarks, {} rounds, both legs each round, into {}",
        benches.len(),
        config.runs,
        out.display()
    );
    for (bench, round, leg) in schedule(benches.len(), config.runs) {
        let bench = &benches[bench];
        let spec = match leg {
            Leg::Base => &config.base,
            Leg::Cand => &config.cand,
        };
        let dir = out.join(leg.dir()).join(round.to_string());
        let written = run_one(config, spec, bench, &dir)?;
        for (path, file) in &written {
            check_stamp(path, file, spec)?;
            println!(
                "bench-ab: {} round {}/{} {}: {}",
                bench.id,
                round + 1,
                config.runs,
                leg.dir(),
                progress(path, file)
            );
        }
    }
    compare::judge_dir(&out, &config.options, config.report.as_deref())
}

/// The order of the runs: `(benchmark, round, leg)`, the leg that goes first alternating.
///
/// Every benchmark runs all its rounds before the next starts, and within
/// a round both legs run back to back, the base first on even rounds.
#[must_use]
pub fn schedule(benches: usize, runs: u32) -> Vec<(usize, u32, Leg)> {
    let mut order = Vec::new();
    for bench in 0..benches {
        for round in 0..runs {
            let (first, second) = if round % 2 == 0 {
                (Leg::Base, Leg::Cand)
            } else {
                (Leg::Cand, Leg::Base)
            };
            order.push((bench, round, first));
            order.push((bench, round, second));
        }
    }
    order
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
    for pattern in &config.benches {
        if !found
            .iter()
            .any(|bench| bench.id.contains(pattern.as_str()))
        {
            println!("bench-ab: no benchmark matches {pattern:?}; skipped");
        }
    }
    if found.is_empty() {
        return Err("no benchmark selected: nothing to compare".to_owned());
    }
    Ok(found)
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

/// Run `bench` once under `spec`, and read the metrics files it wrote into `dir`.
fn run_one(
    config: &AbConfig,
    spec: &LegSpec,
    bench: &Bench,
    dir: &Path,
) -> Result<Vec<(PathBuf, MetricsFile)>, String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let before = metrics_files(dir)?;
    let mut launcher = leg_launcher(spec, &bench.exe, Some(dir), config.timeout)?
        .with_env("MTLD3D_CONFIG", &run_config(&config.config, dir));
    let mut outcome = Outcome::default();
    let run = attribute::run_binary(
        &mut launcher,
        Some(vec![bench.name.clone()]),
        1,
        true,
        &mut outcome,
    )?;
    let what = format!("{} under {}", bench.id, dir.display());
    if run.outcome == BinaryOutcome::GpuHang {
        return Err(format!(
            "{what}: the driver reported a GPU hang; no number after it can be trusted{}",
            outcome.notes()
        ));
    }
    let passed = outcome
        .results
        .iter()
        .any(|result| result.name == bench.name && result.verdict == Verdict::Passed);
    if !passed || run.failed {
        let verdicts: Vec<String> = outcome
            .results
            .iter()
            .map(|result| format!("{} {:?}", result.name, result.verdict))
            .collect();
        return Err(format!(
            "{what} did not pass: {}{}",
            if verdicts.is_empty() {
                "no result".to_owned()
            } else {
                verdicts.join("; ")
            },
            outcome.notes()
        ));
    }
    let written: Vec<PathBuf> = metrics_files(dir)?
        .into_iter()
        .filter(|(path, stamp)| before.get(path) != Some(stamp))
        .map(|(path, _)| path)
        .collect();
    if written.is_empty() {
        return Err(format!(
            "{what} passed but wrote no bench-<name>.metrics into {}",
            dir.display()
        ));
    }
    written
        .into_iter()
        .map(|path| metrics::read(&path).map(|file| (path, file)))
        .collect()
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
    match file.meta.get("layer") {
        Some(layer) if *layer == spec.stamp => Ok(()),
        Some(layer) => Err(format!(
            "{}: the run loaded layer {layer}, the leg installed {}; the prefix or the Wine \
             tree is not the one the leg was built into, or the build is stale",
            path.display(),
            spec.stamp
        )),
        None => Err(format!("{}: no meta layer line", path.display())),
    }
}

/// The progress text of one metrics file: its benchmark and its median frame time.
fn progress(path: &Path, file: &MetricsFile) -> String {
    let bench = path
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(metrics::bench_of)
        .unwrap_or_default();
    file.metrics.get(PROGRESS_METRIC).map_or_else(
        || format!("{bench}: no {PROGRESS_METRIC}"),
        |metric| {
            format!(
                "{bench}: {PROGRESS_METRIC} {:.3} {}",
                metric.value,
                metric.unit.as_str()
            )
        },
    )
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
