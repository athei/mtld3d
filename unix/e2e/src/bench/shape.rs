//! The pass shape of a benchmark's steady frame, read from the layer's pass trace, leg to leg.
//!
//! After its timed rounds, `bench-ab` runs every benchmark whose metrics
//! declare `shape` lines (the scene benchmarks) once more in each leg with
//! the pass trace on ([`SHAPE_RUST_LOG`]) and keeps that run's layer log
//! under `<leg>/shape/<test>/`; nothing of that run is timed. A benchmark
//! without `shape` lines, such as one that meets new shaders by design and
//! whose submissions depend on how often it retries, gets no shape run. The trace
//! names every pass the encoder opens and closes and the decisions the
//! load/store rules take when the submission is sent. The timed numbers
//! cannot catch a wrong decision, since a store left out makes a frame
//! faster, not slower, so the two legs' decisions are compared exactly
//! instead: any difference is a shape change, and a shape change fails the
//! comparison unless `--accept` names `shape` or `shape:<bench>`.
//!
//! A submission's passes are opened in index order from 0, and its rule
//! lines follow when it is sent, so a recording line after a rule line, or a
//! `pass-open` whose index is not the next one, starts the next submission.
//! The last submission of a log may be cut short, so only the ones another
//! follows count as complete. Of the last [`WINDOW`] complete submissions the
//! most common canonical shape is compared, and fewer than 80 % of them
//! agreeing is an error rather than a verdict: a frame that changes from
//! frame to frame says nothing about the change under test. The log is read
//! once, line by line, keeping only those canonical shapes.
//!
//! The canonical list names attachments by first appearance within the
//! submission (`C0`, `C1` for colour targets, `D0` for depth), since the
//! handles are addresses that differ from run to run. Each pass carries its
//! kind, size, attachments, load actions as recorded, the store action of
//! each plane as the indexed `pass-store` lines set it (`store` when none
//! does), the stores set on its extra render targets, the call that closed
//! it, its command and draw counts, and the rule that removed it, if one
//! did. Rule lines that name no pass by index (the load reverts, the strips,
//! the cull count, the joins, the stores logged without an index) are kept
//! in log order with their handles replaced, so every decision the trace
//! records is compared.

use std::{
    collections::{BTreeMap, VecDeque},
    fs::{self, File},
    io::{BufRead as _, BufReader},
    path::{Path, PathBuf},
};

use super::{Leg, SHAPE_DIR, metrics};

/// The `RUST_LOG` of a shape run: the pass trace, and warnings from the rest of the layer.
pub const SHAPE_RUST_LOG: &str = "mtld3d=warn,mtld3d::d3d9::passes=trace";

/// How many of a log's last complete submissions the compared shape is chosen from.
pub const WINDOW: usize = 30;

/// The log target of the pass trace.
const TRACE_TARGET: &str = "mtld3d::d3d9::passes";

/// The `--accept` name that accepts every shape change.
const ACCEPT_ALL: &str = "shape";

/// The prefix of an `--accept` name that accepts one benchmark's shape change.
const ACCEPT_PREFIX: &str = "shape:";

/// How the canonical list writes an attachment slot that holds nothing.
const NONE: &str = "-";

/// One pass of the canonical list, every field as the report prints it.
#[derive(Debug, PartialEq, Eq)]
pub struct Pass {
    /// `render`, or `upload` for a texture-upload pass spliced into the front.
    pub kind: String,
    pub size: String,
    pub color: String,
    pub srgb: String,
    pub depth: String,
    /// The mask of render targets 1..3 the pass attaches.
    pub extra: String,
    pub color_load: String,
    pub depth_load: String,
    pub color_store: String,
    pub depth_store: String,
    pub stencil_store: String,
    /// `<label>=<store>` for each extra render target a `pass-store` line names, `-` for none.
    pub extra_store: String,
    /// The call that closed the pass, `-` when no `pass-close` line names it.
    pub close: String,
    pub cmds: String,
    pub draws: String,
    /// `kept`, or the rule that removed the pass.
    pub fate: String,
}

impl Pass {
    /// The pass's fields, named, in the order the report lists them.
    #[must_use]
    pub fn fields(&self) -> [(&'static str, &str); 16] {
        [
            ("kind", &self.kind),
            ("size", &self.size),
            ("color", &self.color),
            ("srgb", &self.srgb),
            ("depth", &self.depth),
            ("extra", &self.extra),
            ("color_load", &self.color_load),
            ("depth_load", &self.depth_load),
            ("color_store", &self.color_store),
            ("depth_store", &self.depth_store),
            ("stencil_store", &self.stencil_store),
            ("extra_store", &self.extra_store),
            ("close", &self.close),
            ("cmds", &self.cmds),
            ("draws", &self.draws),
            ("fate", &self.fate),
        ]
    }

    /// A pass as recording leaves it, before its own line fills in what it names.
    fn recorded() -> Self {
        Self {
            kind: String::new(),
            size: String::new(),
            color: NONE.to_owned(),
            srgb: NONE.to_owned(),
            depth: NONE.to_owned(),
            extra: NONE.to_owned(),
            color_load: NONE.to_owned(),
            depth_load: NONE.to_owned(),
            color_store: "store".to_owned(),
            depth_store: "store".to_owned(),
            stencil_store: "store".to_owned(),
            extra_store: NONE.to_owned(),
            close: NONE.to_owned(),
            cmds: "?".to_owned(),
            draws: "?".to_owned(),
            fate: "kept".to_owned(),
        }
    }

    /// The pass as one line of the canonical list.
    #[must_use]
    pub fn render(&self, index: usize) -> String {
        format!(
            "#{index} {} {} color={} srgb={} depth={} extra={} load={}/{} store={}/{}/{} \
             extra_store={} close={} cmds={} draws={} {}",
            self.kind,
            self.size,
            self.color,
            self.srgb,
            self.depth,
            self.extra,
            self.color_load,
            self.depth_load,
            self.color_store,
            self.depth_store,
            self.stencil_store,
            self.extra_store,
            self.close,
            self.cmds,
            self.draws,
            self.fate
        )
    }
}

/// The canonical shape of one submission, and how representative it is of its log.
#[derive(Debug, Default)]
pub struct Frame {
    /// Every pass the submission recorded, in recorded order.
    pub passes: Vec<Pass>,
    /// The rule lines no pass took, in log order, handles replaced by labels.
    pub rules: Vec<String>,
    /// Which submission of the log this is, from 1: the latest of those with this shape.
    pub submission: usize,
    /// How many submissions the log holds.
    pub submissions: usize,
    /// How many of the complete submissions looked at have this shape.
    pub agreeing: usize,
    /// How many complete submissions were looked at: the last [`WINDOW`], or fewer.
    pub considered: usize,
}

impl Frame {
    /// Whether `other` has the same passes and rule lines, whichever submissions they are.
    fn same_shape(&self, other: &Self) -> bool {
        self.passes == other.passes && self.rules == other.rules
    }
}

/// One benchmark's pass shape in the two legs.
#[derive(Debug)]
pub struct ShapeReport {
    /// The benchmark, as its metrics files name it.
    pub bench: String,
    /// The directory of its shape runs, named after its libtest path.
    pub run: String,
    /// Which submissions were compared and how big they are.
    pub summary: String,
    /// The differences, one per line; empty when the two shapes are the same.
    pub diff: Vec<String>,
    /// `--accept` named `shape` or this benchmark's shape.
    pub accepted: bool,
}

impl ShapeReport {
    /// Whether the two legs' shapes differ.
    #[must_use]
    pub const fn changed(&self) -> bool {
        !self.diff.is_empty()
    }

    /// Whether this shape fails the comparison: it changed and nothing accepted that.
    #[must_use]
    pub const fn fails(&self) -> bool {
        self.changed() && !self.accepted
    }
}

/// The shape reports of an A/B directory and the remarks that are no verdict.
#[derive(Debug, Default)]
pub struct ShapeComparison {
    pub reports: Vec<ShapeReport>,
    pub notes: Vec<String>,
    /// The directory holds shape runs at all.
    pub present: bool,
}

/// The target and message of one line of the layer's log: `[<time> <LEVEL> <target>] <message>`.
#[must_use]
pub fn log_message(line: &str) -> Option<(&str, &str)> {
    let (header, message) = line.strip_prefix('[')?.split_once("] ")?;
    let target = header.split_whitespace().nth(2)?;
    Some((target, message))
}

/// The directory a benchmark's shape run writes into, named after its libtest path.
#[must_use]
pub fn run_dir_name(test: &str) -> String {
    test.replace("::", ".")
}

/// Whether an `--accept` name is about shapes rather than a metric.
#[must_use]
pub fn is_accept_name(name: &str) -> bool {
    name == ACCEPT_ALL || name.starts_with(ACCEPT_PREFIX)
}

/// The differences between two canonical shapes, one line each; empty when they are the same.
///
/// Passes are paired by recorded index and compared field by field; the
/// rule lines no pass took are compared as a list, `-` for a line only the
/// base has and `+` for one only the candidate has.
#[must_use]
pub fn diff(base: &Frame, cand: &Frame) -> Vec<String> {
    let mut out = Vec::new();
    if base.passes.len() != cand.passes.len() {
        out.push(format!(
            "passes: base {}, cand {}",
            base.passes.len(),
            cand.passes.len()
        ));
    }
    for index in 0..base.passes.len().max(cand.passes.len()) {
        match (base.passes.get(index), cand.passes.get(index)) {
            (Some(b), Some(c)) => {
                let (was, is) = (b.fields(), c.fields());
                let changes: Vec<String> = was
                    .iter()
                    .zip(is.iter())
                    .filter(|(was, is)| was.1 != is.1)
                    .map(|(was, is)| format!("{} {} -> {}", was.0, was.1, is.1))
                    .collect();
                if !changes.is_empty() {
                    out.push(format!("pass #{index}: {}", changes.join(", ")));
                }
            }
            (Some(b), None) => out.push(format!("only in base: {}", b.render(index))),
            (None, Some(c)) => out.push(format!("only in cand: {}", c.render(index))),
            (None, None) => {}
        }
    }
    out.extend(list_diff(&base.rules, &cand.rules));
    out
}

/// Compare the shape runs of the A/B directory `dir`, benchmark by benchmark.
///
/// A directory whose legs hold no shape run (one written before shape runs
/// existed) is compared on its numbers alone, with a note saying so.
///
/// # Errors
///
/// Returns a message when only one leg holds shape runs, a benchmark's shape
/// run is in one leg only, a run directory holds no single layer log with a
/// pass trace, or a trace has no complete submission.
pub fn compare_dir(dir: &Path, accept: &[String]) -> Result<ShapeComparison, String> {
    let base = shape_runs(&dir.join(Leg::Base.dir()).join(SHAPE_DIR))?;
    let cand = shape_runs(&dir.join(Leg::Cand.dir()).join(SHAPE_DIR))?;
    let mut comparison = ShapeComparison::default();
    let (base, mut cand) = match (base, cand) {
        (None, None) => {
            comparison
                .notes
                .push("no shape runs in this directory: the pass shapes were not compared".into());
            return Ok(comparison);
        }
        (Some(base), Some(cand)) => {
            comparison.present = true;
            (base, cand)
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(format!(
                "only one leg of {} holds shape runs: both legs run every benchmark's shape run",
                dir.display()
            ));
        }
    };
    if let Some(name) = base.keys().find(|name| !cand.contains_key(*name)) {
        return Err(format!("the shape run of {name} is in the base leg only"));
    }
    if let Some(name) = cand.keys().find(|name| !base.contains_key(*name)) {
        return Err(format!("the shape run of {name} is in the cand leg only"));
    }
    for (name, mut base_run) in base {
        let mut cand_run = cand.remove(&name).unwrap_or_default();
        let base_frame = base_run.frame()?;
        let cand_frame = cand_run.frame()?;
        let bench = if base_run.benches.is_empty() {
            name.clone()
        } else {
            base_run.benches.join("+")
        };
        let accepted = accept.iter().any(|token| {
            token == ACCEPT_ALL
                || token.strip_prefix(ACCEPT_PREFIX).is_some_and(|wanted| {
                    *wanted == name || base_run.benches.iter().any(|b| b == wanted)
                })
        });
        comparison.reports.push(ShapeReport {
            summary: format!(
                "base: {}; cand: {}",
                base_frame.describe(),
                cand_frame.describe()
            ),
            diff: diff(&base_frame, &cand_frame),
            bench,
            run: name,
            accepted,
        });
    }
    for token in accept.iter().filter(|token| is_accept_name(token)) {
        let wanted = token.strip_prefix(ACCEPT_PREFIX);
        let matched = comparison.reports.iter().any(|report| {
            report.changed()
                && wanted.is_none_or(|wanted| {
                    report.run == wanted || report.bench.split('+').any(|b| b == wanted)
                })
        });
        if !matched {
            comparison
                .notes
                .push(format!("--accept {token}: no shape of that name changed"));
        }
    }
    Ok(comparison)
}

/// One benchmark's shape run in one leg: its trace, where it came from, and its benchmarks.
#[derive(Default)]
struct ShapeRun {
    log: PathBuf,
    trace: Trace,
    benches: Vec<String>,
}

impl ShapeRun {
    /// The run's steady canonical shape, an error naming its log when it has none.
    fn frame(&mut self) -> Result<Frame, String> {
        std::mem::take(&mut self.trace)
            .finish()
            .map_err(|reason| format!("{}: {reason}", self.log.display()))
    }
}

/// The shape runs under a leg's `shape` directory by run directory name; `None` when it is absent.
fn shape_runs(dir: &Path) -> Result<Option<BTreeMap<String, ShapeRun>>, String> {
    if !dir.exists() {
        return Ok(None);
    }
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut runs = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        runs.insert(name, shape_run(&path)?);
    }
    Ok(Some(runs))
}

/// The layer log holding the pass trace of one run directory, and the benchmarks it names.
fn shape_run(dir: &Path) -> Result<ShapeRun, String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut logs = Vec::new();
    let mut benches = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        let file_name = entry.file_name().to_string_lossy().into_owned();
        if let Some(bench) = metrics::bench_of(&file_name) {
            benches.push(bench.to_owned());
        } else if Path::new(&file_name)
            .extension()
            .is_some_and(|extension| extension == "log")
        {
            logs.push(entry.path());
        }
    }
    benches.sort();
    logs.sort();
    // The test binary's listing may leave a log of its own; the run's is
    // the one with the trace in it.
    let mut traced = Vec::new();
    for log in logs {
        let trace = read_trace(&log)?;
        if trace.submissions > 0 {
            traced.push((log, trace));
        }
    }
    if traced.len() > 1 {
        return Err(format!(
            "{}: {} layer logs carry a pass trace; a shape run is one process",
            dir.display(),
            traced.len()
        ));
    }
    let (log, trace) = traced.pop().ok_or_else(|| {
        format!(
            "{}: no layer log with a pass trace; the shape run did not have \
             RUST_LOG={SHAPE_RUST_LOG}",
            dir.display()
        )
    })?;
    Ok(ShapeRun {
        log,
        trace,
        benches,
    })
}

/// Read the pass trace of the log at `path` in one pass.
fn read_trace(path: &Path) -> Result<Trace, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut trace = Trace::default();
    let mut bytes = Vec::new();
    loop {
        bytes.clear();
        let read = reader
            .read_until(b'\n', &mut bytes)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if read == 0 {
            return Ok(trace);
        }
        trace.line(String::from_utf8_lossy(&bytes).trim_end_matches(['\n', '\r']));
    }
}

/// One line of the pass trace that the shape reads.
enum Line<'a> {
    /// `pass-open`, with its index.
    Open(usize, &'a str),
    /// `upload-pass`, with its index.
    Upload(usize, &'a str),
    Close(&'a str),
    /// Any other `pass-` line but `pass-break`: a decision of the load/store rules.
    Rule(&'a str),
}

/// What kind of pass-trace line `message` is; `None` for a line the shape does not read.
///
/// `pass-break` lines are left out: the `pass-close` that follows names the
/// same trigger, and a break with no pass open closes nothing. So are the
/// indented lines of the periodic perf summary, which land in whichever
/// submission is being built when its window closes.
fn classify(message: &str) -> Option<Line<'_>> {
    if message.starts_with(char::is_whitespace) {
        return None;
    }
    let index = || word_value(message, "idx").and_then(|idx| idx.parse::<usize>().ok());
    match message.split_whitespace().next()? {
        "pass-open" => Some(Line::Open(index()?, message)),
        "upload-pass" => Some(Line::Upload(index()?, message)),
        "pass-close" => Some(Line::Close(message)),
        "pass-break" => None,
        word if word.starts_with("pass-") => Some(Line::Rule(message)),
        _ => None,
    }
}

/// A pass trace read line by line: the submission being built and the last complete ones.
#[derive(Default)]
struct Trace {
    current: Option<Builder>,
    /// Passes the current submission has recorded, the index its next `pass-open` names.
    recorded: usize,
    /// The current submission has logged a rule line, so its recording is over.
    ruled: bool,
    /// The last [`WINDOW`] complete submissions, oldest first.
    complete: VecDeque<Frame>,
    /// Submissions started, the current one included.
    submissions: usize,
}

impl Trace {
    /// Read one line of the layer's log; lines of other targets are skipped.
    fn line(&mut self, text: &str) {
        let Some((target, message)) = log_message(text) else {
            return;
        };
        if target != TRACE_TARGET {
            return;
        }
        let Some(line) = classify(message) else {
            return;
        };
        // An upload pass is spliced into the front of the current submission
        // at any time, so an `upload-pass` line starts a new submission only
        // after a rule line. When a submission logged no rule line at all and
        // the next one opens with an upload pass, that pass is read into the
        // one before; the next one's `pass-open` then names an index that is
        // not the next, and starts it there. Such a shape differs from its
        // neighbours and loses the vote, so it costs agreement, not a verdict.
        let starts = match line {
            Line::Open(index, _) => self.ruled || index != self.recorded,
            Line::Upload(..) => self.ruled,
            Line::Close(_) | Line::Rule(_) => false,
        };
        if starts || self.current.is_none() {
            if let Some(done) = self.current.take() {
                let mut frame = done.frame;
                frame.submission = self.submissions;
                self.complete.push_back(frame);
                if self.complete.len() > WINDOW {
                    self.complete.pop_front();
                }
            }
            self.current = Some(Builder::default());
            self.submissions += 1;
            self.recorded = 0;
            self.ruled = false;
        }
        match line {
            Line::Open(index, _) => self.recorded = index + 1,
            Line::Upload(..) => self.recorded += 1,
            Line::Rule(_) => self.ruled = true,
            Line::Close(_) => {}
        }
        if let Some(builder) = self.current.as_mut() {
            builder.push(&line);
        }
    }

    /// The most common shape of the complete submissions kept, if 80 % of them share it.
    fn finish(mut self) -> Result<Frame, String> {
        if self.submissions == 0 {
            return Err(format!(
                "no pass trace: no {TRACE_TARGET} line opens a pass; the run needs \
                 RUST_LOG={SHAPE_RUST_LOG}"
            ));
        }
        let considered = self.complete.len();
        let mut best: Option<(usize, usize)> = None;
        for (index, frame) in self.complete.iter().enumerate() {
            let alike = self
                .complete
                .iter()
                .filter(|other| other.same_shape(frame))
                .count();
            // The later of two equally common shapes wins, so the latest
            // submission of the winning shape is the one reported.
            if best.is_none_or(|(_, count)| alike >= count) {
                best = Some((index, alike));
            }
        }
        let Some((index, agreeing)) = best else {
            return Err(
                "the pass trace holds one submission, and only a submission another one \
                 follows is known to be complete"
                    .to_owned(),
            );
        };
        if agreeing * 5 < considered * 4 {
            return Err(format!(
                "unstable pass shape: the most common shape of the last {considered} complete \
                 submissions is only {agreeing} of them, under 80 %; a frame that changes from \
                 frame to frame cannot be compared"
            ));
        }
        let mut frame = self.complete.remove(index).unwrap_or_default();
        frame.submissions = self.submissions;
        frame.agreeing = agreeing;
        frame.considered = considered;
        Ok(frame)
    }
}

/// Handle labels in first-seen order: `C<n>` for colour targets, `D<n>` for depth.
#[derive(Default)]
struct Labels {
    names: BTreeMap<String, String>,
    colors: usize,
    depths: usize,
}

impl Labels {
    /// The label of `handle`, assigned on first sight; a null handle has none.
    fn name(&mut self, handle: &str, depth: bool) -> String {
        if handle.is_empty() || handle == "0x0" {
            return NONE.to_owned();
        }
        if let Some(name) = self.names.get(handle) {
            return name.clone();
        }
        let name = if depth {
            self.depths += 1;
            format!("D{}", self.depths - 1)
        } else {
            self.colors += 1;
            format!("C{}", self.colors - 1)
        };
        self.names.insert(handle.to_owned(), name.clone());
        name
    }
}

/// One submission's canonical shape as its lines arrive.
#[derive(Default)]
struct Builder {
    labels: Labels,
    frame: Frame,
    /// The recorded index of each pass still in the list, once the rules start removing passes.
    ///
    /// The indices the rule lines name are positions in it.
    live: Option<Vec<usize>>,
}

impl Builder {
    /// Add what one line of the submission says.
    fn push(&mut self, line: &Line<'_>) {
        let labels = &mut self.labels;
        let frame = &mut self.frame;
        match *line {
            Line::Open(index, message) => {
                let fields = fields(message);
                let get = |key: &str| field(&fields, key);
                let pass = Pass {
                    kind: "render".to_owned(),
                    size: get("size"),
                    color: labels.name(&get("color"), false),
                    srgb: labels.name(&get("srgb"), false),
                    depth: labels.name(&get("depth"), true),
                    extra: get("extra"),
                    color_load: load(&get("color_load")),
                    depth_load: load(&get("depth_load")),
                    ..Pass::recorded()
                };
                let at = index.min(frame.passes.len());
                frame.passes.insert(at, pass);
            }
            Line::Upload(index, message) => {
                let fields = fields(message);
                let get = |key: &str| field(&fields, key);
                let pass = Pass {
                    kind: "upload".to_owned(),
                    size: get("size"),
                    color: labels.name(&get("color"), false),
                    color_load: load(&get("load")),
                    ..Pass::recorded()
                };
                let at = index.min(frame.passes.len());
                frame.passes.insert(at, pass);
            }
            Line::Close(message) => {
                let fields = fields(message);
                let get = |key: &str| field(&fields, key);
                let index = get("idx").parse::<usize>().ok();
                if let Some(pass) = index.and_then(|index| frame.passes.get_mut(index)) {
                    pass.close = get("caller");
                    pass.cmds = get("cmds");
                    pass.draws = get("draws");
                } else {
                    frame.rules.push(canonical_rule(message, labels));
                }
            }
            Line::Rule(message) => {
                let text = canonical_rule(message, labels);
                let positions = self
                    .live
                    .get_or_insert_with(|| (0..frame.passes.len()).collect());
                if !apply_rule(message, positions, &mut frame.passes, labels) {
                    frame.rules.push(text);
                }
            }
        }
    }
}

impl Frame {
    /// One leg's part of a report's summary line.
    fn describe(&self) -> String {
        format!(
            "{} passes and {} rule lines, shared by {} of the last {} complete submissions \
             (latest {} of {})",
            self.passes.len(),
            self.rules.len(),
            self.agreeing,
            self.considered,
            self.submission,
            self.submissions
        )
    }
}

/// Apply a rule line that names a pass by index to that pass; `false` when it names none.
///
/// The rules that remove a pass (Rule I's dead clears, Rule E's coalesced
/// clears) run before the store rules and log the index each removal had at
/// the time, so following the removals in `live` maps every index they and
/// the store lines name back to a recorded pass. The rules after them name
/// either no pass or positions after a cull that logs only a count, so their
/// lines stay in the rule list. A store goes to the attachment whose label
/// the line names: render target 0, the depth plane, or one of the extra
/// render targets of a pass that attaches any; a store naming none of those
/// stays in the rule list too.
fn apply_rule(
    message: &str,
    live: &mut Vec<usize>,
    passes: &mut [Pass],
    labels: &mut Labels,
) -> bool {
    let indices: Vec<usize> = message
        .split_whitespace()
        .filter_map(|word| word.strip_prefix("idx="))
        .filter_map(|idx| idx.trim_end_matches(')').parse::<usize>().ok())
        .collect();
    let recorded = |index: usize| live.get(index).copied();
    match message.split_whitespace().next() {
        Some("pass-dead-clear") => {
            let Some(pass) = indices.first().and_then(|&i| recorded(i)) else {
                return false;
            };
            "removed as a dead clear".clone_into(&mut passes[pass].fate);
            live.retain(|&p| p != pass);
            true
        }
        Some("pass-coalesce") => {
            let (Some(pass), Some(target)) = (
                indices.first().and_then(|&i| recorded(i)),
                indices.get(1).and_then(|&i| recorded(i)),
            ) else {
                return false;
            };
            passes[pass].fate = format!("coalesced into #{target}");
            live.retain(|&p| p != pass);
            true
        }
        Some("pass-store") => {
            let Some(pass) = indices.first().and_then(|&i| recorded(i)) else {
                return false;
            };
            let Some((_, decision)) = message.split_once("→ ") else {
                return false;
            };
            let (action, reason) = decision.split_once(' ').unwrap_or((decision, ""));
            let reason: Vec<String> = reason
                .split_whitespace()
                .map(|word| {
                    word.strip_prefix("idx=")
                        .and_then(|rest| {
                            let digits = rest.trim_end_matches(')');
                            let target = recorded(digits.parse::<usize>().ok()?)?;
                            Some(format!("#{target}{}", &rest[digits.len()..]))
                        })
                        .unwrap_or_else(|| word.to_owned())
                })
                .collect();
            let action = action.to_ascii_lowercase();
            let value = if reason.is_empty() {
                action
            } else {
                format!("{action} {}", reason.join(" "))
            };
            let Some((key, handle)) = message.split_whitespace().find_map(|word| {
                ["color=", "depth=", "stencil="]
                    .into_iter()
                    .find_map(|key| Some((key, word.strip_prefix(key)?)))
            }) else {
                return false;
            };
            let target = &mut passes[pass];
            let label = labels.name(handle, key != "color=");
            match key {
                "color=" if label == target.color => target.color_store = value,
                "color=" if target.extra != NONE && target.extra != "0x0" => {
                    let entry = format!("{label}={value}");
                    if target.extra_store == NONE {
                        target.extra_store = entry;
                    } else {
                        target.extra_store = format!("{},{entry}", target.extra_store);
                    }
                }
                "depth=" if label == target.depth => target.depth_store = value,
                "stencil=" if label == target.depth => target.stencil_store = value,
                _ => return false,
            }
            true
        }
        _ => false,
    }
}

/// A rule line with its handles replaced by labels and its indices written `#<n>`.
fn canonical_rule(message: &str, labels: &mut Labels) -> String {
    let words: Vec<String> = message
        .split_whitespace()
        .map(|word| {
            for (key, depth) in [
                ("color=", false),
                ("srgb=", false),
                ("depth=", true),
                ("stencil=", true),
            ] {
                if let Some(value) = word.strip_prefix(key)
                    && value.starts_with("0x")
                {
                    let end = value.find(':').unwrap_or(value.len());
                    let name = labels.name(&value[..end], depth);
                    return format!("{key}{name}{}", &value[end..]);
                }
            }
            word.strip_prefix("idx=")
                .map_or_else(|| word.to_owned(), |index| format!("#{index}"))
        })
        .collect();
    words.join(" ")
}

/// The `key=value` fields of a trace line, a value in braces taken whole.
///
/// The load actions print as `Clear { r: 1065353216, ... }`, so a word that
/// opens a brace, and every word after it until the brace closes, belongs
/// to the value before it.
fn fields(message: &str) -> Vec<(&str, String)> {
    let mut out: Vec<(&str, String)> = Vec::new();
    let mut open = 0usize;
    for word in message.split_whitespace() {
        let opens = word.matches('{').count();
        let closes = word.matches('}').count();
        if open > 0 || word.starts_with('{') {
            if let Some((_, value)) = out.last_mut() {
                value.push(' ');
                value.push_str(word);
            }
            open = (open + opens).saturating_sub(closes);
            continue;
        }
        let Some((key, value)) = word.split_once('=') else {
            continue;
        };
        out.push((key, value.to_owned()));
        open = opens.saturating_sub(closes);
    }
    out
}

/// The value of `key` among `fields`, `?` when the line has none.
fn field(fields: &[(&str, String)], key: &str) -> String {
    fields
        .iter()
        .find(|(name, _)| *name == key)
        .map_or_else(|| "?".to_owned(), |(_, value)| value.clone())
}

/// The value of the first `key=value` word of `message`.
fn word_value<'a>(message: &'a str, key: &str) -> Option<&'a str> {
    message
        .split_whitespace()
        .find_map(|word| word.strip_prefix(key)?.strip_prefix('='))
}

/// A load action as the canonical list writes it: `load`, `dontcare` or `clear(<values>)`.
///
/// The trace prints a clear's values as the bits of each float.
fn load(value: &str) -> String {
    match value {
        "Load" => "load".to_owned(),
        "DontCare" => "dontcare".to_owned(),
        _ if value.starts_with("Clear") => {
            let values: Vec<String> = value
                .split(|c: char| !c.is_ascii_digit())
                .filter(|digits| !digits.is_empty())
                .map(|digits| {
                    digits.parse::<u32>().map_or_else(
                        |_| digits.to_owned(),
                        |bits| f32::from_bits(bits).to_string(),
                    )
                })
                .collect();
            format!("clear({})", values.join(","))
        }
        _ => value.to_owned(),
    }
}

/// The lines only one of two lists has, `rule - <line>` for the base and `rule + <line>` for cand.
///
/// A longest-common-subsequence walk, so a line inserted into the middle of
/// the list shows as one line rather than as every line after it.
fn list_diff(base: &[String], cand: &[String]) -> Vec<String> {
    let (n, m) = (base.len(), cand.len());
    let mut common = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            common[i][j] = if base[i] == cand[j] {
                common[i + 1][j + 1] + 1
            } else {
                common[i + 1][j].max(common[i][j + 1])
            };
        }
    }
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n || j < m {
        if base.get(i).is_some_and(|line| cand.get(j) == Some(line)) {
            i += 1;
            j += 1;
        } else if j == m || (i < n && common[i + 1][j] >= common[i][j + 1]) {
            out.push(format!("rule - {}", base[i]));
            i += 1;
        } else {
            out.push(format!("rule + {}", cand[j]));
            j += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests;
