//! Support shared by the synthetic benchmarks, which `make bench` runs and `make test` skips.
//!
//! No test of its own. The benchmarks in `bench_frame_shape.rs` and
//! `bench_shader_stutter.rs` are `#[ignore]`d: they take seconds each and
//! their numbers depend on the machine and on whatever else it is running,
//! so they measure and report instead of asserting, and the ordinary suite
//! lists them as ignored. `make bench` runs them alone, one at a time in one
//! process, through the runner's `--ignored`.
//!
//! Each benchmark writes a plain-text report, `bench-<name>.txt`, into the
//! directory the layer writes its log to (`log.dir`, which `make bench`
//! points at its output directory). On a `PERF=1` build the report carries
//! the rows of the layer's five-second `mtld3d::perf` summary that cover the
//! measured frames, copied out of that log. Frame times are taken on the API
//! thread from one `Present` return to the next with `Instant`, which on
//! Windows reads `QueryPerformanceCounter`, and the device presents with
//! `D3DPRESENT_INTERVAL_IMMEDIATE` so the display does not pace it. Beside
//! that the report gives the time from a `Present` return to the next
//! `Present` call, the API thread's own work on the frame, which tells a
//! frame bound by the API thread from one bound behind `Present`.

use core::fmt::Write as _;
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime},
};

use mtld3d_tests::{Harness, Texture, TexturedVertex, config_value, config_var};
use mtld3d_types::{
    D3D_OK, D3DDECL_END, D3DDECLMETHOD_DEFAULT, D3DDECLTYPE_D3DCOLOR, D3DDECLTYPE_FLOAT2,
    D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_COLOR, D3DDECLUSAGE_POSITION, D3DDECLUSAGE_TEXCOORD,
    D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, D3DVERTEXELEMENT9,
};

/// The marker that opens one window of the `mtld3d::perf` summary in the layer log.
const PERF_HEADER: &str = "── perf  window=";

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

/// The block of a perf window that [`LayerLog::compilation_rows`] copies.
const COMPILATION_BLOCK: [&str; 1] = ["Compilation"];

/// Edge of every pattern texture, in texels.
const TEXTURE_EDGE: u32 = 64;

/// The shader model of a programmable material.
pub enum Model {
    /// `vs_2_0` with `ps_2_0`.
    Sm2,
    /// `vs_3_0` with `ps_3_0`.
    Sm3,
}

/// Present-to-Present frame times, and the API thread's work before each `Present`.
pub struct FrameClock {
    last: Instant,
    times: Vec<Duration>,
    work: Vec<Duration>,
}

impl FrameClock {
    /// A clock whose first frame ends at the next [`Self::present`].
    pub fn start(capacity: usize) -> Self {
        Self {
            last: Instant::now(),
            times: Vec::with_capacity(capacity),
            work: Vec::with_capacity(capacity),
        }
    }

    /// `Present` the frame the API thread has just finished, and time it.
    ///
    /// # Panics
    /// Panics if `Present` fails.
    pub fn present(&mut self, h: &Harness) {
        self.work.push(self.last.elapsed());
        ok(h.present(), "Present");
        let now = Instant::now();
        self.times.push(now - self.last);
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
    fn of(times: &[Duration]) -> Self {
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

/// Write `body` as the report `bench-<name>.txt` in the log directory, and print it.
///
/// The header names the architecture, the build and the suite-wide
/// `MTLD3D_CONFIG`, so a report says what it measured.
///
/// # Panics
/// Panics if the report cannot be written.
pub fn write_report(name: &str, log: &LayerLog, body: &str) {
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
    println!("{report}");
}

/// The cargo profile this benchmark was built with, and whether debug assertions were on.
///
/// `make bench` builds the benchmark with the layer's profile, so this is
/// the layer's build too. The profile is the directory cargo put the
/// executable under, `target/<triple>/<profile>/deps`; a binary anywhere
/// else (a stage) names none, and the debug-assertion state still says
/// which kind of build it is.
fn build() -> String {
    let profile = std::env::current_exe()
        .ok()
        .and_then(|exe| {
            let deps = exe.parent()?;
            (deps.file_name()? == "deps").then_some(())?;
            Some(deps.parent()?.file_name()?.to_string_lossy().into_owned())
        })
        .map_or_else(
            || "profile unknown (not in a cargo target directory)".to_owned(),
            |profile| format!("{profile} profile"),
        );
    let assertions = if cfg!(debug_assertions) { "on" } else { "off" };
    format!("{profile}, debug assertions {assertions}")
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

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}
