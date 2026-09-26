//! Support shared by the synthetic benchmarks, which `make bench` runs and `make test` skips.
//!
//! No test of its own. The benchmarks, one or more in each `bench_*.rs`
//! beside this file, are `#[ignore]`d: they take seconds each and
//! their numbers depend on the machine and on whatever else it is running,
//! so they measure and report instead of asserting, and the ordinary suite
//! lists them as ignored. `make bench` runs them alone, one at a time in one
//! process, through the runner's `--ignored`.
//!
//! Each benchmark writes a plain-text report, `bench-<name>.txt`, into the
//! directory the layer writes its log to (`log.dir`, which `make bench`
//! points at its output directory), and beside it `bench-<name>.metrics`,
//! the same numbers one record per line for a program to compare (see
//! [`Metrics`]). On a `PERF=1` build the report carries the rows of the
//! layer's five-second `mtld3d::perf` summary that cover the measured frames,
//! copied out of that log, and the metrics file the counters of the `perf-kv`
//! line the layer logs after each of them. Frame times are taken on the API
//! thread from one `Present` return to the next with [`TscClock`], the
//! layer's own `rdtsc` primitive at its calibrated rate (under Wine
//! `QueryPerformanceCounter` ticks at 100 ns and costs a call into `ntdll`),
//! and the device presents with `D3DPRESENT_INTERVAL_IMMEDIATE` so the
//! display does not pace it. Beside that the report gives the time from a
//! `Present` return to the next `Present` call, the API thread's own work on
//! the frame, which tells a frame bound by the API thread from one bound
//! behind `Present`. Each benchmark also samples the process's address space
//! after its warm-up and at the end of its measured frames, and its peak
//! working set.

use core::fmt::Write as _;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

use mtld3d_shared::tsc::{rdtsc, tsc_hz, u64_to_f64_exact};
use mtld3d_tests::{Harness, MemorySample, Texture, TexturedVertex, config_value, config_var};
use mtld3d_types::{
    D3D_OK, D3DDECL_END, D3DDECLMETHOD_DEFAULT, D3DDECLTYPE_D3DCOLOR, D3DDECLTYPE_FLOAT2,
    D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_COLOR, D3DDECLUSAGE_POSITION, D3DDECLUSAGE_TEXCOORD,
    D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, D3DVERTEXELEMENT9,
};

/// The `meta` keys every metrics file carries, which [`Metrics::meta`] may not repeat.
const COMMON_META: [&str; 10] = [
    "layer",
    "layer_image",
    "layer_unix_image",
    "arch",
    "profile",
    "debug_assertions",
    "config",
    "config_entries",
    "tsc_hz",
    "tsc_granularity_ns",
];

/// Back-to-back counter reads whose smallest nonzero step [`TscClock`] reports as its granularity.
const GRANULARITY_READS: u32 = 100_000;

/// The marker that opens one window of the `mtld3d::perf` summary in the layer log.
const PERF_HEADER: &str = "── perf  window=";

/// What precedes the pairs of the machine-read `perf-kv` line the layer logs after each window.
///
/// The line is `perf-kv v1 window_s=<x> frames=<n> key=value ...` after the
/// logger's prefix; `docs/ARCHITECTURE.md` lists the keys and what each
/// suffix means.
const PERF_KV: &str = "] perf-kv v1 ";

/// The blocks of a perf window a report copies, by the words their first row starts with.
///
/// Each block runs to the next blank row, and only a row at [`BLOCK_INDENT`]
/// opens one. The rest of the grid (resource renames, keys gating,
/// allocator footprint) is left in the log. A block a build's grid does not
/// have is skipped.
const PERF_BLOCKS: [&str; 10] = [
    "buckets:",
    "API thread",
    "Encoder thread",
    "Submit thread",
    "Present thread",
    "GPU",
    "Frame total",
    "Caches",
    "Commands / passes",
    "Compilation",
];

/// Bytes of one [`TexturedVertex`], the stride of every stream the benchmarks bind.
pub const STRIDE: u32 = 24;

/// The declaration of the [`TexturedVertex`] layout: position, colour, one texture coordinate.
pub const TEXTURED_DECL: [D3DVERTEXELEMENT9; 4] = [
    element(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
    element(12, D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_COLOR),
    element(16, D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_TEXCOORD),
    D3DDECL_END,
];

/// The indent of the first row of a block in the perf grid.
///
/// Deeper rows that happen to start with a block's words (`GPU copy` under
/// a resource row) belong to the block they sit in and open nothing.
const BLOCK_INDENT: usize = 4;

/// What precedes the build stamp on the line the layer's `d3d9.dll` logs when it loads.
///
/// The line is `d3d9.dll <build> <image id> loaded at <base>`, after the
/// logger's `[<time> <level> <target>] ` prefix; `<build>` is the release
/// identity the build stamped in from `git describe`.
const LAYER_STAMP: &str = "] d3d9.dll ";

/// What follows the build stamp and the image ID on the layer's load line.
const LAYER_LOADED: &str = " loaded at ";

/// What precedes the build stamp on the line the layer's unix library logs when it starts.
///
/// The line is `mtld3d.so <build> <image id> initialized`, the unix side's
/// counterpart of the `d3d9.dll` load line, and most of the layer's code is
/// in that library.
const UNIX_STAMP: &str = "] mtld3d.so ";

/// What follows the build stamp and the image ID on the unix library's line.
const UNIX_INITIALIZED: &str = " initialized";

/// The block of a perf window that [`LayerLog::compilation_rows`] copies.
const COMPILATION_BLOCK: [&str; 1] = ["Compilation"];

/// The least time the calls of one [`CallTimes`] sample add up to.
///
/// At least a hundred times the clock's step, so one step is under 1 % of a
/// sample. Under Rosetta `rdtsc` steps by 41 ns (a run's
/// `tsc_granularity_ns`), which puts the floor at 4.1 us; 5 us keeps a
/// margin over it. A clock whose metrics report a coarser step needs a
/// higher floor.
pub const SAMPLE_FLOOR: Duration = Duration::from_micros(5);

/// The least time a single call takes to count as a spike, whatever the median.
///
/// The `LockRect` stalls a game shows take milliseconds; this keeps a
/// median of a few ticks from turning every third-tick call into one.
pub const SPIKE_FLOOR: Duration = Duration::from_micros(50);

/// The fewest samples whose p99 a comparison reads as a time rather than as context.
pub const MIN_P99_SAMPLES: usize = 50;

/// Edge of every pattern texture, in texels.
const TEXTURE_EDGE: u32 = 64;

/// The shader model of a programmable material.
pub enum Model {
    /// `vs_2_0` with `ps_2_0`.
    Sm2,
    /// `vs_3_0` with `ps_3_0`.
    Sm3,
}

/// The benchmarks' clock: the layer's `rdtsc` primitive, its calibrated rate, its finest step.
///
/// `mtld3d_shared::tsc` is what the layer's perf counters read, so a
/// benchmark's times and the layer's own divide by one calibration. The rate
/// is latched once per process by its first reader: a benchmark calls
/// [`Self::calibrated`] before it times anything, so no conversion made
/// while it measures waits for the calibration, and the conversions read
/// the latched rate from anywhere. A count is comparable only within one
/// process.
pub struct TscClock {
    hz: u64,
    granularity_ns: f64,
}

impl TscClock {
    /// Latch the calibrated rate and measure the smallest step between back-to-back reads.
    ///
    /// # Panics
    /// Panics if the rate is zero.
    pub fn calibrated() -> Self {
        let hz = tsc_hz();
        assert!(hz > 0, "the rdtsc rate is not zero");
        let mut smallest = u64::MAX;
        let mut last = rdtsc();
        for _ in 0..GRANULARITY_READS {
            let now = rdtsc();
            let step = now.wrapping_sub(last);
            if step > 0 {
                smallest = smallest.min(step);
            }
            last = now;
        }
        let granularity_ns = if smallest == u64::MAX {
            0.0
        } else {
            Self::ticks_ns(smallest)
        };
        Self { hz, granularity_ns }
    }

    /// The counter now.
    #[inline]
    pub fn now() -> u64 {
        rdtsc()
    }

    /// The time from `start`, a [`Self::now`] reading, to now.
    #[inline]
    pub fn since(start: u64) -> Duration {
        Self::duration(Self::now().saturating_sub(start))
    }

    /// `ticks` of the counter as a duration, to the nanosecond.
    pub fn duration(ticks: u64) -> Duration {
        let nanos = u128::from(ticks) * 1_000_000_000 / u128::from(tsc_hz());
        Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
    }

    /// `ticks` of the counter in nanoseconds, with the fraction.
    pub fn ticks_ns(ticks: u64) -> f64 {
        u64_to_f64_exact(ticks) * 1e9 / u64_to_f64_exact(tsc_hz())
    }
}

/// Present-to-Present frame times, and the API thread's work before each `Present`.
///
/// Read with [`TscClock`], which the benchmark calibrated before it started this.
pub struct FrameClock {
    last: u64,
    times: Vec<Duration>,
    work: Vec<Duration>,
}

impl FrameClock {
    /// A clock whose first frame ends at the next [`Self::present`].
    pub fn start(capacity: usize) -> Self {
        Self {
            last: TscClock::now(),
            times: Vec::with_capacity(capacity),
            work: Vec::with_capacity(capacity),
        }
    }

    /// `Present` the frame the API thread has just finished, and time it.
    ///
    /// # Panics
    /// Panics if `Present` fails.
    pub fn present(&mut self, h: &Harness) {
        let called = TscClock::now();
        ok(h.present(), "Present");
        let now = TscClock::now();
        self.work
            .push(TscClock::duration(called.saturating_sub(self.last)));
        self.times
            .push(TscClock::duration(now.saturating_sub(self.last)));
        self.last = now;
    }

    /// Frames timed so far.
    pub const fn frames(&self) -> usize {
        self.times.len()
    }

    /// Wall time the timed frames add up to.
    pub fn elapsed(&self) -> Duration {
        self.times.iter().sum()
    }

    /// Mean, median, 99th percentile and worst frame of the timed frames.
    ///
    /// # Panics
    /// Panics if no frame was timed.
    pub fn stats(&self) -> FrameStats {
        FrameStats::of(&self.times)
    }

    /// The same summary of the API thread's work from each frame's start to its `Present` call.
    ///
    /// # Panics
    /// Panics if no frame was timed.
    pub fn work_stats(&self) -> FrameStats {
        FrameStats::of(&self.work)
    }

    /// How many timed frames took longer than `limit`.
    pub fn over(&self, limit: Duration) -> usize {
        self.times.iter().filter(|&&time| time > limit).count()
    }
}

/// The summary of a run of frame times.
pub struct FrameStats {
    pub frames: usize,
    pub mean: Duration,
    pub p50: Duration,
    pub p99: Duration,
    pub max: Duration,
}

impl FrameStats {
    /// The summary of `times`, which need not be sorted.
    ///
    /// # Panics
    /// Panics if `times` is empty.
    pub fn of(times: &[Duration]) -> Self {
        let mut sorted = times.to_vec();
        sorted.sort_unstable();
        let frames = sorted.len();
        assert!(frames > 0, "a benchmark times at least one frame");
        let count = u32::try_from(frames).expect("frame count fits u32");
        Self {
            frames,
            mean: sorted.iter().sum::<Duration>() / count,
            p50: sorted[nearest_rank(frames, 50)],
            p99: sorted[nearest_rank(frames, 99)],
            max: sorted[frames - 1],
        }
    }

    /// One report row: mean, nearest-rank p50 and p99, and max, in milliseconds.
    pub fn row(&self) -> String {
        format!(
            "mean {:.3} ms  p50 {:.3} ms  p99 {:.3} ms  max {:.3} ms (nearest-rank)",
            ms(self.mean),
            ms(self.p50),
            ms(self.p99),
            ms(self.max)
        )
    }
}

/// The times of one kind of call a benchmark makes itself, summarised well above the clock's tick.
///
/// A call far shorter than a frame is timed within a few steps of the
/// clock, and its single time moves with the machine more than with the
/// layer. A sample here is the mean time per call over whole frames
/// instead: [`Self::add`] sums a frame's calls, and
/// [`Self::end_frame`] closes a sample once the frames it spans add up to
/// at least [`SAMPLE_FLOOR`], so one clock step is under 1 % of it. Frequent
/// calls give one sample a frame, rare ones one per several frames. The
/// slowest single call is kept, and so is every single call of at least
/// [`SPIKE_FLOOR`], the calls [`Self::spikes`] counts.
#[derive(Default)]
pub struct CallTimes {
    open: Duration,
    open_calls: u32,
    samples: Vec<Duration>,
    calls: u64,
    max: Duration,
    slow: Vec<Duration>,
}

impl CallTimes {
    /// Count one call that took `took`.
    pub fn add(&mut self, took: Duration) {
        self.open += took;
        self.open_calls += 1;
        self.calls += 1;
        self.max = self.max.max(took);
        if took >= SPIKE_FLOOR {
            self.slow.push(took);
        }
    }

    /// End a frame: close the open sample if its calls add up to [`SAMPLE_FLOOR`].
    pub fn end_frame(&mut self) {
        if self.open >= SAMPLE_FLOOR {
            self.samples.push(self.open / self.open_calls);
            self.open = Duration::ZERO;
            self.open_calls = 0;
        }
    }

    /// Calls counted.
    pub const fn calls(&self) -> u64 {
        self.calls
    }

    /// The summary of the per-call samples, or `None` when no sample was closed.
    pub fn stats(&self) -> Option<FrameStats> {
        (!self.samples.is_empty()).then(|| FrameStats::of(&self.samples))
    }

    /// Single calls slower than twice the median sample and than [`SPIKE_FLOOR`], and that limit.
    ///
    /// With no sample closed the limit is [`SPIKE_FLOOR`] alone.
    pub fn spikes(&self) -> (usize, Duration) {
        let limit = self
            .stats()
            .map_or(SPIKE_FLOOR, |stats| (stats.p50 * 2).max(SPIKE_FLOOR));
        let count = self.slow.iter().filter(|&&took| took > limit).count();
        (count, limit)
    }

    /// One report row: the samples' p50 and p99 per call, and the slowest single call, in ns.
    pub fn row(&self) -> String {
        let summary = self.stats().map_or_else(
            || "no sample closed".to_owned(),
            |stats| {
                format!(
                    "per call p50 {} ns  p99 {} ns (nearest-rank over {} samples of whole frames)",
                    stats.p50.as_nanos(),
                    stats.p99.as_nanos(),
                    stats.frames
                )
            },
        );
        format!(
            "{summary}, {} calls, slowest single call {} ns",
            self.calls,
            self.max.as_nanos()
        )
    }
}

/// The layer's log of this process, read for its perf summary.
///
/// The layer names the file `<exe stem>-<host pid>.log`, and the host pid is
/// not visible from inside Wine, so the file is the newest log of this
/// executable in the log directory, and it must have been written since the
/// benchmark started: `make bench` runs one process at a time into a
/// directory of its own, and this process is writing its log. The layer
/// creates the file on its log thread's first write, so it is looked for
/// after the warm-up, by which time the device has logged.
pub struct LayerLog {
    path: Option<PathBuf>,
}

impl LayerLog {
    /// Find this process's log, the newest of this executable's written at or after `since`.
    ///
    /// `since` is taken before the benchmark creates its device: device
    /// creation always logs, so the file's modification time passes it
    /// however quiet the warm-up is. A mark taken after the device can
    /// precede the last line the layer's log thread writes, and a warm-up
    /// that logs nothing then leaves no log written since it.
    ///
    /// # Panics
    /// Panics when no such log exists: every number the report would copy
    /// from the log would then be another run's, or missing without saying so.
    pub fn find(since: SystemTime) -> Self {
        let stem = std::env::current_exe()
            .ok()
            .and_then(|exe| {
                exe.file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
            })
            .unwrap_or_default();
        let prefix = format!("{stem}-");
        let path = fs::read_dir(log_dir())
            .into_iter()
            .flatten()
            .flatten()
            .filter(|entry| {
                let path = entry.path();
                let log = path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("log"));
                log && entry.file_name().to_string_lossy().starts_with(&prefix)
            })
            .filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
            .filter(|(modified, _)| *modified >= since)
            .max_by_key(|(modified, _)| *modified)
            .map(|(_, path)| path);
        assert!(
            path.is_some(),
            "no layer log of {stem} written since the benchmark started in {}",
            log_dir().display()
        );
        Self { path }
    }

    /// The log at `path`: another process's, whose file the benchmark that ran it knows.
    pub const fn at(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }

    /// The log's current length, the position a measured phase starts or ends at.
    pub fn mark(&self) -> u64 {
        self.path
            .as_deref()
            .and_then(|path| fs::metadata(path).ok())
            .map_or(0, |meta| meta.len())
    }

    /// The perf window that covers only frames between `from` and `to`, or the best there is.
    ///
    /// A window is emitted when it ends, so the first one written after
    /// `from` began before it and is [`PerfRows::Partial`]; any later one
    /// lies wholly inside and is [`PerfRows::Full`]. The last such window is
    /// the one returned.
    pub fn perf_rows(&self, from: u64, to: u64) -> PerfRows {
        let Some(bytes) = self.path.as_deref().and_then(|path| fs::read(path).ok()) else {
            return PerfRows::Absent;
        };
        let start = usize::try_from(from).map_or(bytes.len(), |at| at.min(bytes.len()));
        let end = usize::try_from(to).map_or(bytes.len(), |at| at.min(bytes.len()));
        let text = String::from_utf8_lossy(&bytes[start..end.max(start)]);
        let lines: Vec<&str> = text.lines().collect();
        let headers: Vec<usize> = (0..lines.len())
            .filter(|&at| lines[at].contains(PERF_HEADER))
            .collect();
        match headers.as_slice() {
            [] => PerfRows::Absent,
            [only] => PerfRows::Partial(window_rows(&lines, *only, &PERF_BLOCKS)),
            [.., last] => PerfRows::Full(window_rows(&lines, *last, &PERF_BLOCKS)),
        }
    }

    /// The Compilation rows of the first perf window of the device that wrote the last before `to`.
    ///
    /// Each device's encoder thread names itself in its window titles and
    /// opens its first window when the device is created, so that window holds
    /// the device's warm-up compiles, however long the warm-up took. `None`
    /// outside a `PERF=1` build.
    pub fn first_window_rows(&self, to: u64) -> Option<String> {
        let bytes = fs::read(self.path.as_deref()?).ok()?;
        let end = usize::try_from(to).map_or(bytes.len(), |at| at.min(bytes.len()));
        let text = String::from_utf8_lossy(&bytes[..end]);
        let lines: Vec<&str> = text.lines().collect();
        let encoder = |line: &str| {
            line.split_whitespace()
                .find(|word| word.starts_with("encoder="))
                .map(str::to_owned)
        };
        let last = lines.iter().rev().find(|line| line.contains(PERF_HEADER))?;
        let this = encoder(last)?;
        let first = (0..lines.len()).find(|&at| {
            lines[at].contains(PERF_HEADER) && encoder(lines[at]).as_ref() == Some(&this)
        })?;
        Some(window_rows(&lines, first, &COMPILATION_BLOCK))
    }

    /// The title and Compilation rows of every perf window written between `from` and `to`.
    ///
    /// The windows follow each other, so together they account for every
    /// compile in the span. None outside a `PERF=1` build.
    pub fn compilation_rows(&self, from: u64, to: u64) -> Vec<String> {
        let Some(bytes) = self.path.as_deref().and_then(|path| fs::read(path).ok()) else {
            return Vec::new();
        };
        let start = usize::try_from(from).map_or(bytes.len(), |at| at.min(bytes.len()));
        let end = usize::try_from(to).map_or(bytes.len(), |at| at.min(bytes.len()));
        let text = String::from_utf8_lossy(&bytes[start..end.max(start)]);
        let lines: Vec<&str> = text.lines().collect();
        (0..lines.len())
            .filter(|&at| lines[at].contains(PERF_HEADER))
            .map(|header| window_rows(&lines, header, &COMPILATION_BLOCK))
            .collect()
    }

    /// The `perf-kv` pairs of every perf window whose grid was written between `from` and `to`.
    ///
    /// One entry per window, in order, holding what follows `perf-kv v1 ` on
    /// the line the layer logs after that window's grid, or `None` for a
    /// window the line never followed (a layer older than the line). The line
    /// is looked for after its grid up to the next window's, past `to` when
    /// it has to be, since the log thread may write it after the span ended.
    /// Empty outside a `PERF=1` build.
    pub fn perf_kv(&self, from: u64, to: u64) -> Vec<Option<String>> {
        let Some(bytes) = self.path.as_deref().and_then(|path| fs::read(path).ok()) else {
            return Vec::new();
        };
        let start = usize::try_from(from).map_or(bytes.len(), |at| at.min(bytes.len()));
        let end = usize::try_from(to).map_or(bytes.len(), |at| at.min(bytes.len()));
        let mut windows: Vec<Option<String>> = Vec::new();
        let mut at = start;
        for line in bytes[start..].split(|&byte| byte == b'\n') {
            let text = String::from_utf8_lossy(line);
            if text.contains(PERF_HEADER) {
                if at >= end {
                    break;
                }
                windows.push(None);
            } else if let Some((_, pairs)) = text.split_once(PERF_KV)
                && let Some(window) = windows.last_mut()
            {
                window.get_or_insert_with(|| pairs.trim_end().to_owned());
            }
            at += line.len() + 1;
        }
        windows
    }

    /// The `perf-kv` pairs of the last window wholly inside the span, as `perf_rows` picks it.
    ///
    /// Empty when the span holds fewer than two windows: the first window
    /// written after `from` began before it.
    pub fn perf_kv_last_full(&self, from: u64, to: u64) -> Vec<Option<String>> {
        let mut windows = self.perf_kv(from, to);
        if windows.len() < 2 {
            return Vec::new();
        }
        windows.split_off(windows.len() - 1)
    }

    /// The build stamp and image ID on the layer's `d3d9.dll` load line, if the log has one.
    ///
    /// Two builds of one commit share the stamp; the image ID, which the
    /// linker derives from the binary's contents, tells them apart.
    pub fn layer_identity(&self) -> Option<(String, String)> {
        self.identity(LAYER_STAMP, LAYER_LOADED)
    }

    /// The image ID on the unix library's `mtld3d.so` line, if the log has one.
    ///
    /// The `d3d9.dll` image ID alone cannot tell two builds apart whose
    /// difference is all in the unix library.
    pub fn unix_image(&self) -> Option<String> {
        self.identity(UNIX_STAMP, UNIX_INITIALIZED)
            .map(|(_, image)| image)
    }

    /// The build stamp and image ID of the first line holding `<stamp><build> <image><after>`.
    fn identity(&self, stamp: &str, after: &str) -> Option<(String, String)> {
        let bytes = fs::read(self.path.as_deref()?).ok()?;
        String::from_utf8_lossy(&bytes).lines().find_map(|line| {
            let (_, rest) = line.split_once(stamp)?;
            let (identity, _) = rest.split_once(after)?;
            let (build, image) = identity.split_once(' ')?;
            Some((build.to_owned(), image.to_owned()))
        })
    }

    /// Where the log is, for the report; `None` when no log was found.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}

/// The perf summary rows a report carries.
pub enum PerfRows {
    /// No perf window in the measured span: not a `PERF=1` build, or too short a span.
    Absent,
    /// The only window in the span, which also covers frames from before it.
    Partial(String),
    /// A window that covers measured frames alone.
    Full(String),
}

impl PerfRows {
    /// The report section for these rows.
    pub fn section(&self) -> String {
        match self {
            Self::Absent => {
                "perf: no mtld3d::perf window in the measured span (not a PERF=1 build?)\n"
                    .to_owned()
            }
            Self::Partial(rows) => {
                format!(
                    "perf: the one window in the span, which also covers earlier frames\n{rows}"
                )
            }
            Self::Full(rows) => {
                format!("perf: last window wholly inside the measured frames\n{rows}")
            }
        }
    }
}

/// The directory the layer writes this process's log to.
///
/// `log.dir` from the suite-wide `MTLD3D_CONFIG` when it names one, and
/// otherwise the layer's default, `mtld3d-logs` beside the executable.
pub fn log_dir() -> PathBuf {
    config_value("log.dir")
        .filter(|dir| !dir.is_empty())
        .map_or_else(
            || {
                std::env::current_exe()
                    .ok()
                    .and_then(|exe| exe.parent().map(Path::to_path_buf))
                    .unwrap_or_default()
                    .join("mtld3d-logs")
            },
            PathBuf::from,
        )
}

/// Write `body` as the report `bench-<name>.txt` in the log directory, `metrics` beside it.
///
/// The header names the architecture, the build and the suite-wide
/// `MTLD3D_CONFIG`, so a report says what it measured. The name is the one
/// `metrics` was created with, and the records go to `bench-<name>.metrics`.
/// The report is printed as well.
///
/// # Panics
/// Panics if either file cannot be written.
pub fn write_report(metrics: &Metrics, log: &LayerLog, body: &str) {
    let name = &metrics.bench;
    let mut report = format!(
        "bench: {name} ({arch})\nbuild: {build}\nMTLD3D_CONFIG: {config}\nlayer log: {log}\n",
        arch = std::env::consts::ARCH,
        build = build(),
        config = config_var().unwrap_or_default(),
        log = log
            .path()
            .map_or_else(|| "not found".to_owned(), |path| path.display().to_string()),
    );
    report.push_str(body);
    let dir = log_dir();
    fs::create_dir_all(&dir).expect("the report directory can be created");
    let path = dir.join(format!("bench-{name}.txt"));
    fs::write(&path, &report).expect("the benchmark report can be written");
    let path = dir.join(format!("bench-{name}.metrics"));
    fs::write(&path, metrics.file(log)).expect("the benchmark metrics can be written");
    println!("{report}");
}

/// Which way a metric is better.
pub enum Direction {
    /// A smaller value is better.
    Lower,
    /// A larger value is better.
    Higher,
}

impl Direction {
    const fn word(self) -> &'static str {
        match self {
            Self::Lower => "lower",
            Self::Higher => "higher",
        }
    }
}

/// How a comparison of two builds reads a metric.
pub enum Class {
    /// A time a build's speed decides.
    Time,
    /// A count of frames or calls that exceed a spike limit.
    Spikes,
    /// A count that must not change: any difference is a real one.
    Exact,
    /// A number that moves between runs of the same build.
    Noisy,
    /// A memory figure, compared like a time with an absolute floor on top.
    Bytes,
    /// Reported for the reader, not compared.
    Info,
}

impl Class {
    const fn word(self) -> &'static str {
        match self {
            Self::Time => "time",
            Self::Spikes => "spikes",
            Self::Exact => "exact",
            Self::Noisy => "noisy",
            Self::Bytes => "bytes",
            Self::Info => "info",
        }
    }
}

/// A metric's value, in the unit its variant names.
pub enum Value {
    /// Milliseconds, four decimals.
    Ms(Duration),
    /// A plain count.
    Count(u64),
    /// Bytes, written as MiB with two decimals.
    Mib(u64),
    /// Nanoseconds, one decimal.
    Ns(f64),
    /// A number from the layer's `perf-kv` line, in `unit`, written with `decimals` places.
    Perf {
        value: f64,
        unit: PerfUnit,
        decimals: usize,
    },
}

impl Value {
    /// The value as the file writes it, and its unit.
    fn text(self) -> (String, &'static str) {
        match self {
            Self::Ms(duration) => (format!("{:.4}", ms(duration)), "ms"),
            Self::Count(count) => (count.to_string(), "count"),
            Self::Mib(bytes) => (mib(bytes), "mib"),
            Self::Ns(nanos) => (format!("{nanos:.1}"), "ns"),
            Self::Perf {
                value,
                unit,
                decimals,
            } => (format!("{value:.decimals$}"), unit.word()),
        }
    }
}

/// The unit of a metric read from the `perf-kv` line.
pub enum PerfUnit {
    Ms,
    Count,
    Bytes,
}

impl PerfUnit {
    const fn word(&self) -> &'static str {
        match self {
            Self::Ms => "ms",
            Self::Count => "count",
            Self::Bytes => "bytes",
        }
    }
}

/// Whether every frame of a benchmark's perf windows issues the same calls.
///
/// It decides whether a per-frame count the calls fix is compared exactly:
/// it is when every frame is alike, and only reported when the windows mix
/// frames of different kinds in proportions the machine's speed decides.
pub enum FrameWork {
    /// Every frame issues the same calls.
    Fixed,
    /// The frames differ.
    Varying,
}

/// One render pass of a frame, as the benchmark that draws it defines it.
pub struct PassShape {
    /// Width of the pass's render target, in pixels.
    pub width: u32,
    /// Height of the pass's render target, in pixels.
    pub height: u32,
    /// Draw calls in the pass each frame.
    pub draws: u32,
    /// Draws through the fixed-function vertex pipeline.
    pub ff_vs: u32,
    /// Draws through the fixed-function texture stages.
    pub ff_ps: u32,
    /// Textures the pass's draws sample, summed over the draws.
    pub textures: u32,
}

/// A benchmark's numbers in the machine-read form of `bench-<name>.metrics`.
///
/// UTF-8, one record per line, fields separated by one space, and `#`
/// opening a comment line. The records are, in this order:
///
/// - `meta <bench> <key> <value...>`, the value running to the end of the
///   line: `layer` (the build stamp the layer logged, or `unknown`),
///   `layer_image` (the image ID on the same line, or `unknown`), `arch`
///   (`i686` or `x86_64`), `profile` and `debug_assertions` (`true` or
///   `false`), and `config` (the suite-wide `MTLD3D_CONFIG`, or `none`).
///   `profile` and `debug_assertions` describe the benchmark binary, which
///   `make bench` builds with the layer's profile. `config` leaves out the
///   entries a benchmark's harness adds on top of it, such as the stutter
///   benchmark's `shaderCache.enable=false`; `config_entries` names those
///   (or `none`), so the two together are the settings the layer ran with.
///   `tsc_hz` and `tsc_granularity_ns` describe the clock the times were
///   read with (see [`TscClock`]): its calibrated rate and the smallest step
///   it was seen to take. A benchmark may add keys of its own
///   ([`Metrics::meta`]) after these.
///   `layer_unix_image` is the image ID on the unix library's `mtld3d.so`
///   line (or `unknown`), since most of the layer is in that library.
/// - `metric <bench> <name> <value> <unit> <direction> <class>`: a name of
///   `[a-z0-9_.]`, a unit of `ms`, `ns`, `count`, `mib` or `bytes`, a
///   [`Direction`] and a [`Class`]. The `perf.*` metrics come from the
///   layer's `perf-kv` lines ([`Self::perf`] has the rules); a file without
///   them says why in a `# no perf-kv line` comment.
/// - `shape <bench> pass <i> <W>x<H> draws=<n> ff_vs=<n> ff_ps=<n>
///   tex_per_draw=<x.xx>`, one per [`PassShape`] of a scene benchmark.
///
/// A program comparing a base build with a candidate reads these, so the
/// format is a contract: a record changes by adding a key or a metric, never
/// by changing one.
pub struct Metrics {
    bench: String,
    /// The benchmark's own `meta` records, written after the ones every file carries.
    extra_meta: Vec<(String, String)>,
    /// The `tsc_*` meta values: the clock's rate and its finest step.
    tsc: [String; 2],
    /// The harness's own configuration entries, `none` when it has none.
    config_entries: String,
    records: String,
    shapes: String,
}

impl Metrics {
    /// No records yet, for the benchmark `bench` measured on `h`; `bench` also names both files.
    ///
    /// # Panics
    /// Panics if `bench` is empty or holds whitespace.
    pub fn new(bench: &str, h: &Harness, clock: &TscClock) -> Self {
        Self::for_entries(bench, h.config_entries(), clock)
    }

    /// No records yet, for the benchmark `bench` whose layer ran with the entries `entries`.
    ///
    /// For a benchmark whose measured processes are not its own, such as
    /// children that each create their own device. A `log.dir` entry is a
    /// location, not a layer setting, and is left out.
    ///
    /// # Panics
    /// Panics if `bench` is empty or holds whitespace.
    pub fn for_entries(bench: &str, entries: &str, clock: &TscClock) -> Self {
        assert!(
            !bench.is_empty() && !bench.contains(char::is_whitespace),
            "a benchmark name is one word: {bench:?}"
        );
        let entries: Vec<&str> = entries
            .split(';')
            .map(str::trim)
            .filter(|entry| !entry.is_empty() && !entry.starts_with("log.dir="))
            .collect();
        Self {
            bench: bench.to_owned(),
            extra_meta: Vec::new(),
            tsc: [clock.hz.to_string(), format!("{:.2}", clock.granularity_ns)],
            config_entries: if entries.is_empty() {
                "none".to_owned()
            } else {
                entries.join(";").replace(['\r', '\n'], " ")
            },
            records: String::new(),
            shapes: String::new(),
        }
    }

    /// Add a `meta` record of the benchmark's own, such as which of two paths it took.
    ///
    /// # Panics
    /// Panics if `key` is not one word of `[a-z0-9_]`, or is a key every file carries.
    pub fn meta(&mut self, key: &str, value: &str) {
        assert!(
            !key.is_empty()
                && key
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'),
            "a meta key is [a-z0-9_]+: {key:?}"
        );
        assert!(
            !COMMON_META.contains(&key),
            "meta {key} is one every metrics file carries"
        );
        self.extra_meta
            .push((key.to_owned(), value.replace(['\r', '\n'], " ")));
    }

    /// Add one `metric` record.
    ///
    /// # Panics
    /// Panics if `name` is empty or holds a byte outside `[a-z0-9_.]`.
    pub fn metric(&mut self, name: &str, value: Value, direction: Direction, class: Class) {
        assert!(
            !name.is_empty()
                && name.bytes().all(|byte| {
                    byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"_.".contains(&byte)
                }),
            "a metric name is [a-z0-9_.]+: {name:?}"
        );
        let (value, unit) = value.text();
        let _ = writeln!(
            self.records,
            "metric {bench} {name} {value} {unit} {direction} {class}",
            bench = self.bench,
            direction = direction.word(),
            class = class.word(),
        );
    }

    /// The `<prefix>.p50`, `.mean`, `.p99` and `.max` records of a run of frame times.
    ///
    /// The three that summarise the run are compared as times; the worst
    /// frame, one sample, is reported alone.
    pub fn frame_rows(&mut self, prefix: &str, stats: &FrameStats) {
        self.percentile_rows(prefix, stats, || Class::Time);
    }

    /// The same four records as [`Self::frame_rows`], all reported for context and none compared.
    ///
    /// For a run whose times are too short, or too dependent on how the
    /// frames fall, to judge by a relative rule.
    pub fn context_rows(&mut self, prefix: &str, stats: &FrameStats) {
        self.percentile_rows(prefix, stats, || Class::Info);
    }

    /// The four percentile records, the three that summarise the run of class `summary`.
    fn percentile_rows(&mut self, prefix: &str, stats: &FrameStats, summary: impl Fn() -> Class) {
        for (row, value, class) in [
            ("p50", stats.p50, summary()),
            ("mean", stats.mean, summary()),
            ("p99", stats.p99, summary()),
            ("max", stats.max, Class::Info),
        ] {
            self.metric(
                &format!("{prefix}.{row}"),
                Value::Ms(value),
                Direction::Lower,
                class,
            );
        }
    }

    /// The `<prefix>.samples`, `.p50`, `.p99` and `.max` records of one kind of call.
    ///
    /// The percentiles, of the per-call samples and in nanoseconds, are
    /// compared as times, the p99 only from [`MIN_P99_SAMPLES`] samples up,
    /// since below that it is one of the few slowest samples; the sample
    /// count and the slowest single call are reported alone. With no sample
    /// closed there are no percentiles to write, and the count says so.
    pub fn call_rows(&mut self, prefix: &str, calls: &CallTimes) {
        let samples = u64::try_from(calls.samples.len()).expect("sample count fits u64");
        self.metric(
            &format!("{prefix}.samples"),
            Value::Count(samples),
            Direction::Higher,
            Class::Info,
        );
        if let Some(stats) = calls.stats() {
            let p99 = if stats.frames < MIN_P99_SAMPLES {
                Class::Info
            } else {
                Class::Time
            };
            for (row, value, class) in [("p50", stats.p50, Class::Time), ("p99", stats.p99, p99)] {
                self.metric(
                    &format!("{prefix}.{row}"),
                    Value::Ns(nanos(value)),
                    Direction::Lower,
                    class,
                );
            }
        }
        self.metric(
            &format!("{prefix}.max"),
            Value::Ns(nanos(calls.max)),
            Direction::Lower,
            Class::Info,
        );
    }

    /// The `mem.*` records of the samples taken after the warm-up and at the end.
    ///
    /// The peak working set is the end sample's, the peak of the whole run.
    /// All of them are of class `bytes`: a change has to clear the absolute
    /// floor as well as the relative one, since two runs of one build can
    /// differ by a MiB or so in a figure of a few MiB.
    pub fn memory(&mut self, warm: &MemorySample, end: &MemorySample) {
        for (phase, sample) in [("warm", warm), ("end", end)] {
            for (row, bytes, direction) in [
                ("committed_mib", sample.committed(), Direction::Lower),
                ("reserved_mib", sample.reserved(), Direction::Lower),
                ("largest_free_mib", sample.largest_free(), Direction::Higher),
            ] {
                self.metric(
                    &format!("mem.{phase}.{row}"),
                    Value::Mib(bytes),
                    direction,
                    Class::Bytes,
                );
            }
        }
        self.metric(
            "mem.peak_ws_mib",
            Value::Mib(end.peak_working_set()),
            Direction::Lower,
            Class::Bytes,
        );
    }

    /// One `shape` record per pass, numbered in the order the frame draws them.
    ///
    /// # Panics
    /// Panics if a pass has no draws.
    pub fn shapes(&mut self, passes: &[PassShape]) {
        for (at, pass) in passes.iter().enumerate() {
            assert!(pass.draws > 0, "pass {at} of a scene has draws");
            let _ = writeln!(
                self.shapes,
                "shape {bench} pass {at} {width}x{height} draws={draws} ff_vs={ff_vs} \
                 ff_ps={ff_ps} tex_per_draw={per_draw:.2}",
                bench = self.bench,
                width = pass.width,
                height = pass.height,
                draws = pass.draws,
                ff_vs = pass.ff_vs,
                ff_ps = pass.ff_ps,
                per_draw = f64::from(pass.textures) / f64::from(pass.draws),
            );
        }
    }

    /// The `perf.*` records of the `perf-kv` lines of `windows`, or a comment saying why not.
    ///
    /// `windows` is what [`LayerLog::perf_kv`] returned for the windows the
    /// benchmark reads. Nothing is recorded when there are none or when any
    /// of them lacks its line; the file then says so in a `# no perf-kv
    /// line` comment and the benchmark goes on. Otherwise every key but
    /// `window_s` and `frames` becomes one metric, all of them lower-is-better,
    /// by its suffix ([`perf_rule`]):
    ///
    /// - `_peak_ms`, the worst frame: `perf.<key>` in ms, `info`, the largest
    ///   of the windows.
    /// - `_ms`, a per-frame average: `perf.<key>` in ms, `time`, the windows'
    ///   mean weighted by their frames. `_avg_ms`, an average per event, is
    ///   weighted by the event's count where the line carries it
    ///   (`comp_async_latency_avg_ms` by `comp_async_installs_total`) and by
    ///   frames otherwise.
    /// - `_bytes`, a peak size: `perf.<key>` in bytes, `bytes`, the largest.
    /// - `_count`, a count gauge: `perf.<key>` in counts, the largest; `exact`
    ///   for a cache size (`cache_*_count`) when the frames are
    ///   [`FrameWork::Fixed`], `noisy` otherwise.
    /// - `_total`, a window's count: `perf.<key less _total>_pf`, the
    ///   windows' totals over their frames to three places, in bytes for a
    ///   `_bytes_total` and in counts otherwise. A count the API calls fix
    ///   ([`structural`]) is `exact` over [`FrameWork::Fixed`] frames and
    ///   `info` over varying ones; a count that depends on how the CPU and
    ///   the GPU overlap (renames, copies, retention, pools, faults, slot
    ///   waits, compiles, command buffers, whose GPU time arrives with a
    ///   later submit) is `noisy`. A window's count is divided by its frames
    ///   because a window lasts five seconds, not a number of frames, so its
    ///   totals grow with the frame rate.
    ///
    /// A key a window leaves out (`docs/ARCHITECTURE.md` names the three that
    /// can be) is aggregated over the windows that carry it.
    pub fn perf(&mut self, windows: &[Option<String>], work: &FrameWork) {
        let missing = windows.iter().filter(|window| window.is_none()).count();
        if windows.is_empty() {
            let _ = writeln!(
                self.records,
                "# no perf-kv line: no perf window in the span these metrics cover \
                 (not a PERF=1 build?)"
            );
            return;
        }
        if missing > 0 {
            let _ = writeln!(
                self.records,
                "# no perf-kv line after {missing} of the {count} perf windows in the span \
                 (a layer older than the line?)",
                count = windows.len()
            );
            return;
        }
        let mut folds: BTreeMap<&str, (PerfRule, PerfFold)> = BTreeMap::new();
        for line in windows.iter().flatten() {
            let pairs: Vec<(&str, f64)> = line
                .split_whitespace()
                .filter_map(|pair| {
                    let (key, value) = pair.split_once('=')?;
                    Some((key, value.parse::<f64>().ok().filter(|v| v.is_finite())?))
                })
                .collect();
            let find = |wanted: &str| {
                pairs
                    .iter()
                    .find_map(|&(key, value)| (key == wanted).then_some(value))
            };
            let frames = find("frames").unwrap_or(0.0);
            for &(key, value) in &pairs {
                if key == "window_s" || key == "frames" {
                    continue;
                }
                let Some(rule) = perf_rule(key, work) else {
                    continue;
                };
                let weight = match rule.fold {
                    Fold::EventMean(events) => find(events).unwrap_or(0.0),
                    Fold::FrameMean | Fold::Max | Fold::PerFrame => frames,
                };
                folds
                    .entry(key)
                    .or_insert_with(|| (rule, PerfFold::default()))
                    .1
                    .add(value, weight);
            }
        }
        for (rule, fold) in folds.into_values() {
            let value = fold.value(&rule.fold);
            self.metric(
                &rule.name,
                Value::Perf {
                    value,
                    unit: rule.unit,
                    decimals: rule.decimals,
                },
                Direction::Lower,
                rule.class,
            );
        }
    }

    /// The whole file: a comment, the `meta` records, then the metrics and the shapes.
    fn file(&self, log: &LayerLog) -> String {
        let bench = &self.bench;
        let mut file = format!("# bench-{bench}.metrics: meta, metric and shape records\n");
        let arch = match std::env::consts::ARCH {
            "x86" => "i686",
            other => other,
        };
        let (layer, layer_image) = log
            .layer_identity()
            .unwrap_or_else(|| ("unknown".to_owned(), "unknown".to_owned()));
        let layer_unix_image = log.unix_image().unwrap_or_else(|| "unknown".to_owned());
        for (key, value) in [
            ("layer", layer),
            ("layer_image", layer_image),
            ("layer_unix_image", layer_unix_image),
            ("arch", arch.to_owned()),
            ("profile", profile().unwrap_or_else(|| "unknown".to_owned())),
            ("debug_assertions", cfg!(debug_assertions).to_string()),
            (
                "config",
                config_var()
                    .filter(|config| !config.is_empty())
                    .map_or_else(
                        || "none".to_owned(),
                        |config| config.replace(['\r', '\n'], " "),
                    ),
            ),
            ("config_entries", self.config_entries.clone()),
            ("tsc_hz", self.tsc[0].clone()),
            ("tsc_granularity_ns", self.tsc[1].clone()),
        ] {
            let _ = writeln!(file, "meta {bench} {key} {value}");
        }
        for (key, value) in &self.extra_meta {
            let _ = writeln!(file, "meta {bench} {key} {value}");
        }
        file.push_str(&self.records);
        file.push_str(&self.shapes);
        file
    }
}

/// The report rows of the memory samples taken after the warm-up and at the end.
pub fn memory_section(warm: &MemorySample, end: &MemorySample) -> String {
    let row = |sample: &MemorySample| {
        format!(
            "committed {} MiB, reserved {} MiB, largest free region {} MiB",
            mib(sample.committed()),
            mib(sample.reserved()),
            mib(sample.largest_free())
        )
    };
    format!(
        "address space after the warm-up: {warm}\naddress space at the end: {end}\n\
         peak working set (under Wine the host process's peak RSS): {peak} MiB\n",
        warm = row(warm),
        end = row(end),
        peak = mib(end.peak_working_set()),
    )
}

/// An `n`x`n` grid of quads over the unit square at `z = 0`, with its 16-bit index list.
pub fn grid(n: u16) -> (Vec<TexturedVertex>, Vec<u16>) {
    let step = 1.0 / f32::from(n);
    let mut vertices = Vec::new();
    for row in 0..=n {
        for col in 0..=n {
            let (u, v) = (f32::from(col) * step, f32::from(row) * step);
            vertices.push(TexturedVertex {
                x: u,
                y: v,
                z: 0.0,
                color: 0xFFFF_FFFF,
                u,
                v,
            });
        }
    }
    let mut indices = Vec::new();
    let stride = n + 1;
    for row in 0..n {
        for col in 0..n {
            let top = row * stride + col;
            let bottom = top + stride;
            indices.extend_from_slice(&[top, bottom, top + 1, top + 1, bottom, bottom + 1]);
        }
    }
    (vertices, indices)
}

/// A managed 64x64 A8R8G8B8 checker of `color` and its inverse, one level.
pub fn pattern_texture(h: &Harness, color: u32) -> Texture<'_> {
    let texture = h.create_texture(
        TEXTURE_EDGE,
        TEXTURE_EDGE,
        1,
        0,
        D3DFMT_A8R8G8B8,
        D3DPOOL_MANAGED,
    );
    let texels: Vec<u32> = (0..TEXTURE_EDGE * TEXTURE_EDGE)
        .map(|at| {
            if (at / 8 + at / (8 * TEXTURE_EDGE)).is_multiple_of(2) {
                color
            } else {
                !color | 0xFF00_0000
            }
        })
        .collect();
    let edge = usize::try_from(TEXTURE_EDGE).expect("texture edge fits usize");
    texture.lock_rect(0, 0).write_u32_rect(edge, edge, &texels);
    texture
}

/// `n / d` for the small integers the benchmarks derive positions and shades from.
///
/// # Panics
/// Panics if either value does not fit `u16`.
pub fn ratio(n: u32, d: u32) -> f32 {
    let to_f32 = |v: u32| f32::from(u16::try_from(v).expect("benchmark ratio operand fits u16"));
    to_f32(n) / to_f32(d)
}

/// The vertex program of a textured material.
///
/// `oPos` is `c0..c3` (view-projection rows) applied to `c4..c7` (world
/// rows) applied to the position, and the texture coordinate is passed on
/// shifted by `def c95`, which is `variant` in `x`. Distinct variants are
/// distinct shaders.
#[rustfmt::skip]
pub fn material_vs(model: &Model, variant: f32) -> Vec<u32> {
    let (version, position_out, texcoord_out) = match model {
        Model::Sm2 => (0xFFFE_0200, 0xC000_0000, 0xE00F_0000), // vs_2_0, oPos, oT0
        Model::Sm3 => (0xFFFE_0300, 0xE000_0000, 0xE00F_0001), // vs_3_0, o0, o1
    };
    let mut tokens = vec![
        version,
        0x0200_001F, 0x8000_0000, 0x900F_0000, // dcl_position v0
        0x0200_001F, 0x8000_0005, 0x900F_0001, // dcl_texcoord0 v1
    ];
    if matches!(model, Model::Sm3) {
        tokens.extend_from_slice(&[
            0x0200_001F, 0x8000_0000, 0xE00F_0000, // dcl_position o0
            0x0200_001F, 0x8000_0005, 0xE00F_0001, // dcl_texcoord0 o1
        ]);
    }
    tokens.extend_from_slice(&def(0xA00F_005F, [variant, 0.0, 0.0, 0.0])); // def c95
    tokens.extend_from_slice(&transform(position_out));
    tokens.extend_from_slice(&[
        0x0300_0002, texcoord_out, 0x90E4_0001, 0xA0E4_005F, // add oT0 / o1, v1, c95
        0x0000_FFFF,
    ]);
    tokens
}

/// The pixel program of a textured material: `texld r0, s0`, then `mad oC0, r0, c0, c7`.
///
/// `def c7` is `tint`, so distinct tints are distinct shaders. A `receiver`
/// also samples `s1`, the shadow map, and multiplies it in first.
#[rustfmt::skip]
pub fn material_ps(model: &Model, tint: [f32; 4], receiver: bool) -> Vec<u32> {
    // The version, the input's dcl usage, and the input as destination and as source.
    let (version, usage, texcoord, texcoord_src) = match model {
        Model::Sm2 => (0xFFFF_0200, 0x8000_0000, 0xB00F_0000, 0xB0E4_0000), // ps_2_0, t0
        Model::Sm3 => (0xFFFF_0300, 0x8000_0005, 0x900F_0000, 0x90E4_0000), // ps_3_0, texcoord0 v0
    };
    let mut tokens = vec![
        version,
        0x0200_001F, usage, texcoord,          // dcl t0 / dcl_texcoord0 v0
        0x0200_001F, 0x9000_0000, 0xA00F_0800, // dcl_2d s0
    ];
    if receiver {
        tokens.extend_from_slice(&[0x0200_001F, 0x9000_0000, 0xA00F_0801]); // dcl_2d s1
    }
    tokens.extend_from_slice(&def(0xA00F_0007, tint)); // def c7
    tokens.extend_from_slice(&[0x0300_0042, 0x800F_0000, texcoord_src, 0xA0E4_0800]); // texld r0, s0
    if receiver {
        tokens.extend_from_slice(&[
            0x0300_0042, 0x800F_0001, texcoord_src, 0xA0E4_0801, // texld r1, s1
            0x0300_0005, 0x800F_0000, 0x80E4_0000, 0x80E4_0001, // mul r0, r0, r1
        ]);
    }
    tokens.extend_from_slice(&[
        0x0400_0004, 0x800F_0000, 0x80E4_0000, 0xA0E4_0000, 0xA0E4_0007, // mad r0, r0, c0, c7
        0x0200_0001, 0x800F_0800, 0x80E4_0000,                           // mov oC0, r0
        0x0000_FFFF,
    ]);
    tokens
}

/// `def <reg>, values`.
pub const fn def(register: u32, values: [f32; 4]) -> [u32; 6] {
    [
        0x0500_0051,
        register,
        values[0].to_bits(),
        values[1].to_bits(),
        values[2].to_bits(),
        values[3].to_bits(),
    ]
}

/// `dp4 r0.x..w, v0, c4..c7`, then `dp4 <out>.x..w, r0, c0..c3`.
///
/// `out` is the output register's token with no write mask.
#[rustfmt::skip]
pub const fn transform(out: u32) -> [u32; 32] {
    [
        0x0300_0009, 0x8001_0000, 0x90E4_0000, 0xA0E4_0004, // dp4 r0.x, v0, c4
        0x0300_0009, 0x8002_0000, 0x90E4_0000, 0xA0E4_0005, // dp4 r0.y, v0, c5
        0x0300_0009, 0x8004_0000, 0x90E4_0000, 0xA0E4_0006, // dp4 r0.z, v0, c6
        0x0300_0009, 0x8008_0000, 0x90E4_0000, 0xA0E4_0007, // dp4 r0.w, v0, c7
        0x0300_0009, out | 0x0001_0000, 0x80E4_0000, 0xA0E4_0000, // dp4 out.x, r0, c0
        0x0300_0009, out | 0x0002_0000, 0x80E4_0000, 0xA0E4_0001, // dp4 out.y, r0, c1
        0x0300_0009, out | 0x0004_0000, 0x80E4_0000, 0xA0E4_0002, // dp4 out.z, r0, c2
        0x0300_0009, out | 0x0008_0000, 0x80E4_0000, 0xA0E4_0003, // dp4 out.w, r0, c3
    ]
}

/// The world rows `c4..c7` of a mesh scaled by `scale` and placed at `(x, y, z)` in clip space.
pub const fn world_rows(scale: f32, x: f32, y: f32, z: f32) -> [f32; 16] {
    [
        scale, 0.0, 0.0, x, //
        0.0, scale, 0.0, y, //
        0.0, 0.0, 1.0, z, //
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// A stream-0 element of `type_` at `offset`, usage index 0.
pub const fn element(offset: u16, type_: u8, usage: u8) -> D3DVERTEXELEMENT9 {
    D3DVERTEXELEMENT9 {
        stream: 0,
        offset,
        type_,
        method: D3DDECLMETHOD_DEFAULT,
        usage,
        usage_index: 0,
    }
}

/// Assert that a device call a benchmark makes succeeded, naming it if not.
///
/// # Panics
/// Panics when `hr` is not `D3D_OK`.
pub fn ok(hr: i32, what: &str) {
    assert_eq!(hr, D3D_OK, "{what}: 0x{hr:08X}");
}

/// The identity rows `c0..c3`.
pub const IDENTITY_ROWS: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// How [`Metrics::perf`] records one `perf-kv` key.
struct PerfRule {
    name: String,
    fold: Fold,
    unit: PerfUnit,
    decimals: usize,
    class: Class,
}

/// How the values one key takes in several windows become one.
enum Fold {
    /// The mean of per-frame values, weighted by each window's frames.
    FrameMean,
    /// The mean of per-event values, weighted by each window's count of the event, the key named.
    EventMean(&'static str),
    /// The largest value of any window.
    Max,
    /// The windows' totals summed, over their frames summed.
    PerFrame,
}

/// The running sums one key's values fold into.
#[derive(Default)]
struct PerfFold {
    sum: f64,
    weighted: f64,
    weight: f64,
    max: f64,
}

impl PerfFold {
    fn add(&mut self, value: f64, weight: f64) {
        self.sum += value;
        self.weighted = value.mul_add(weight, self.weighted);
        self.weight += weight;
        self.max = self.max.max(value);
    }

    fn value(&self, fold: &Fold) -> f64 {
        let over = |numerator: f64| {
            if self.weight > 0.0 {
                numerator / self.weight
            } else {
                0.0
            }
        };
        match fold {
            Fold::FrameMean | Fold::EventMean(_) => over(self.weighted),
            Fold::Max => self.max,
            Fold::PerFrame => over(self.sum),
        }
    }
}

/// The metric one `perf-kv` key becomes, by its suffix; `None` for a key with no known suffix.
///
/// [`Metrics::perf`] states the rules this applies.
fn perf_rule(key: &str, work: &FrameWork) -> Option<PerfRule> {
    let fixed = matches!(work, FrameWork::Fixed);
    let rule = |name: String, fold, unit, decimals, class| {
        Some(PerfRule {
            name,
            fold,
            unit,
            decimals,
            class,
        })
    };
    let own = || format!("perf.{key}");
    if key.ends_with("_peak_ms") {
        return rule(own(), Fold::Max, PerfUnit::Ms, 3, Class::Info);
    }
    if key == "comp_async_latency_avg_ms" {
        let events = Fold::EventMean("comp_async_installs_total");
        return rule(own(), events, PerfUnit::Ms, 4, Class::Time);
    }
    if key.ends_with("_ms") {
        return rule(own(), Fold::FrameMean, PerfUnit::Ms, 4, Class::Time);
    }
    if key.ends_with("_bytes") {
        return rule(own(), Fold::Max, PerfUnit::Bytes, 0, Class::Bytes);
    }
    if key.ends_with("_count") {
        let class = if fixed && key.starts_with("cache_") {
            Class::Exact
        } else {
            Class::Noisy
        };
        return rule(own(), Fold::Max, PerfUnit::Count, 0, class);
    }
    let base = key.strip_suffix("_total")?;
    let unit = if base.ends_with("_bytes") {
        PerfUnit::Bytes
    } else {
        PerfUnit::Count
    };
    let class = match (structural(base), fixed) {
        (true, true) => Class::Exact,
        (true, false) => Class::Info,
        (false, _) => Class::Noisy,
    };
    rule(format!("perf.{base}_pf"), Fold::PerFrame, unit, 3, class)
}

/// Whether the `_total` named `<base>_total` counts work the API calls alone fix.
///
/// Draws, passes, commands, the calls of every API, device, bind, surface
/// and keys-gating row and the keys gate's skips, texture uploads and
/// dirty rects, user-pointer draws, generated fans and staging uploads: a
/// frame that repeats the calls of the one before repeats these. Anything
/// else counts events that depend on how the CPU and the GPU overlap.
fn structural(base: &str) -> bool {
    const WHOLE: [&str; 5] = [
        "draws",
        "passes",
        "commands",
        "fan_generated",
        "vbib_staging_uploads",
    ];
    const PREFIX: [&str; 4] = ["tex_uploads", "tex_dirtyrect", "up_", "keys_"];
    const CALLS: [&str; 4] = ["api_", "dev_", "bind_", "surf_"];
    WHOLE.contains(&base)
        || PREFIX.iter().any(|prefix| base.starts_with(prefix))
        || (base.ends_with("_calls") && CALLS.iter().any(|prefix| base.starts_with(prefix)))
}

/// The title and the `blocks` of the window whose header is `lines[header]`.
fn window_rows(lines: &[&str], header: usize, blocks: &[&str]) -> String {
    let mut out = String::new();
    let title = lines[header];
    let title = title.split_once("] ").map_or(title, |(_, rest)| rest);
    let _ = writeln!(out, "  {title}");
    let body = lines[header + 1..]
        .iter()
        .take_while(|line| !line.starts_with('['));
    let mut copying = false;
    for line in body {
        let row = line.trim_start();
        if row.is_empty() {
            copying = false;
            continue;
        }
        if line.len() - row.len() == BLOCK_INDENT {
            copying = blocks.iter().any(|block| row.starts_with(block)) || (copying && !opens(row));
        }
        if copying {
            let _ = writeln!(out, "{line}");
        }
    }
    out
}

/// The index of the nearest-rank `percent`-th percentile among `count` sorted values.
///
/// The smallest value with at least `percent` % of the values at or below
/// it: rank `ceil(percent / 100 * count)`, index one less.
///
/// # Panics
/// Panics if `count` is zero.
pub const fn nearest_rank(count: usize, percent: usize) -> usize {
    assert!(count > 0, "a percentile of no values");
    (count * percent).div_ceil(100).saturating_sub(1)
}

/// Whether `row`, at block indent, is the first row of any block the grid has a title for.
fn opens(row: &str) -> bool {
    PERF_BLOCKS.iter().any(|block| row.starts_with(block))
}

/// `duration` in nanoseconds, with the fraction a `Duration` carries (none below one).
pub fn nanos(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e9
}

/// `duration` in milliseconds.
pub fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

/// Set a render state, asserting that the call succeeded.
///
/// # Panics
/// Panics when `SetRenderState` fails.
pub fn rs(h: &Harness, state: u32, value: u32) {
    ok(h.set_render_state(state, value), "SetRenderState");
}

/// `bytes` in MiB with two decimals, truncated.
fn mib(bytes: u64) -> String {
    let hundredths = bytes.saturating_mul(100) >> 20;
    format!("{}.{:02}", hundredths / 100, hundredths % 100)
}

/// The cargo profile this benchmark was built with, and whether debug assertions were on.
///
/// `make bench` builds the benchmark with the layer's profile, so this is
/// the layer's build too. A binary outside a cargo target directory (a
/// stage) names no profile, and the debug-assertion state still says which
/// kind of build it is.
fn build() -> String {
    let profile = profile().map_or_else(
        || "profile unknown (not in a cargo target directory)".to_owned(),
        |profile| format!("{profile} profile"),
    );
    let assertions = if cfg!(debug_assertions) { "on" } else { "off" };
    format!("{profile}, debug assertions {assertions}")
}

/// The cargo profile, the directory the executable's `deps` sits in under `target/<triple>`.
fn profile() -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    let deps = exe.parent()?;
    (deps.file_name()? == "deps").then_some(())?;
    Some(deps.parent()?.file_name()?.to_string_lossy().into_owned())
}
