//! Time DXSO parsing and MSL emission over shader corpora on the host, without Wine.
//!
//! Usage: `emit_corpus [--iters N] [--metrics DIR] [mtld3d_shaders.bin ...]`, which
//! `make bench-host` runs with the production profile.
//!
//! Two synthetic corpora always run: `synthetic_ff`, about sixty fixed-function
//! vertex keys and fifty pixel keys shaped like the ones a fixed-function game
//! prewarms, and `synthetic_sm`, a handful of hand-assembled SM1/SM2/SM3 token
//! streams with the variants a draw specializes them into. A synthetic shader
//! that fails to emit fails the run, since that input is this tree's own, and
//! synthetic shaders are named by position in fixed-width names, so the MSL
//! sizes change only when the emitted code does. Each cache file named
//! on the command line adds a corpus named after its directory, holding every
//! programmable record; fixed-function records keep no emission inputs on disk,
//! so a cache contributes only its programmable half.
//!
//! A cache that is not an mtld3d shader cache, or whose container format this
//! build does not read (a stale cache next to a game install is normal), is
//! skipped with a note and writes no metrics; a torn cache is timed up to the
//! damage, with a note. A path that cannot be read fails the run.
//!
//! A corpus is timed in passes, one pass handling every shader once, and a
//! pass's per-shader time is its wall time over the shader count. The median
//! and the minimum over the passes are reported (the median of an even pass
//! count is the mean of the two middle passes). `parse` is `dxso::parse` of the
//! tokens. What the second time covers depends on the corpus, and its metric is
//! named for it:
//!
//! - `emit` in the synthetic corpora is emission alone: the fixed-function
//!   emitters for a key, and the programmable emitters over a program parsed
//!   beforehand.
//! - `parse_emit` in a cache corpus is `ShaderSource::emit`, the call the layer
//!   makes to rebuild a cached shader, which reparses the tokens before it
//!   emits, so it includes the `parse` time reported beside it.
//!
//! Without `--iters` the pass count is chosen so each measurement runs for at
//! least 200 ms.
//!
//! `--metrics DIR` writes one `bench-host_emit_<corpus>.metrics` file per corpus
//! in the bench metrics format: `meta <bench> <key> <value...>` and
//! `metric <bench> <name> <value> <unit> <direction> <class>` lines. The meta
//! lines carry `kind host`, which tells a comparison of two builds that this
//! file names a native binary of its own rather than the layer the end-to-end
//! benchmarks load, and `host_image`, this binary's Mach-O UUID (or
//! `unknown`), beside the build stamp, arch, profile and debug assertions.

use std::{
    env, fmt, fs,
    hint::black_box,
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

use mtld3d_core::{
    dxso::{
        self, DxsoProgram, FfPsKey, FfStage, FfStageFlags, FfVsFlags, FfVsKey, VariantKey,
        VsSamplerKinds,
    },
    shader_cache::{
        CACHE_FORMAT_VERSION, CacheEntry, CachedKind, SHADER_CACHE_SCHEMA_VERSION,
        SHADER_EMITTER_VERSION, ShaderSource, read_header, read_records,
    },
};
use mtld3d_types::{
    D3DCMP_GREATEREQUAL, D3DDECLUSAGE_BLENDINDICES, D3DDECLUSAGE_COLOR, D3DDECLUSAGE_FOG,
    D3DDECLUSAGE_NORMAL, D3DDECLUSAGE_POSITION, D3DDECLUSAGE_TEXCOORD, D3DFOG_LINEAR,
    D3DMCS_COLOR1, D3DMCS_COLOR2, D3DMCS_MATERIAL, D3DTA_CURRENT, D3DTA_DIFFUSE, D3DTA_TEXTURE,
    D3DTOP_ADD, D3DTOP_BLENDCURRENTALPHA, D3DTOP_BLENDTEXTUREALPHA, D3DTOP_MODULATE,
    D3DTOP_MODULATE2X, D3DTOP_MODULATEALPHA_ADDCOLOR, D3DTOP_SELECTARG1, D3DTSS_ALPHAARG1,
    D3DTSS_ALPHAARG2, D3DTSS_ALPHAOP, D3DTSS_COLORARG1, D3DTSS_COLORARG2, D3DTSS_COLOROP,
    texture_stage_state_defaults,
};

/// Wall time a measurement runs for at least when `--iters` is not given.
const TARGET: Duration = Duration::from_millis(200);

/// Fewest passes a measurement takes when `--iters` is not given.
const MIN_PASSES: u32 = 5;

/// `D3DTSS_TCI_CAMERASPACEREFLECTIONVECTOR` in the `FfVsKey::tci_modes` encoding.
const TCI_REFLECTION: u8 = 3;

/// `D3DTTFF_COUNT2` in the `FfVsKey::tt_flags` encoding.
const TT_COUNT2: u8 = 2;

/// `D3DTTFF_COUNT3` in the `FfVsKey::tt_flags` encoding.
const TT_COUNT3: u8 = 3;

// DXSO token fields for the hand-assembled synthetic shaders. Instruction
// opcodes and register types are the bytecode's own numbering, the same the
// emitter's unit-test fixtures spell out.
const VS_1_1: u32 = 0xFFFE_0101;
const PS_1_1: u32 = 0xFFFF_0101;
const PS_1_4: u32 = 0xFFFF_0104;
const VS_2_0: u32 = 0xFFFE_0200;
const PS_2_0: u32 = 0xFFFF_0200;
const VS_3_0: u32 = 0xFFFE_0300;
const PS_3_0: u32 = 0xFFFF_0300;
const END: u32 = 0x0000_FFFF;

const OP_MOV: u16 = 1;
const OP_ADD: u16 = 2;
const OP_MAD: u16 = 4;
const OP_MUL: u16 = 5;
const OP_DP3: u16 = 8;
const OP_DP4: u16 = 9;
const OP_MAX: u16 = 11;
const OP_LRP: u16 = 18;
const OP_DCL: u16 = 31;
const OP_POW: u16 = 32;
const OP_NRM: u16 = 36;
const OP_MOVA: u16 = 46;
const OP_TEXLD: u16 = 66;
const OP_DEF: u16 = 81;

const REG_TEMP: u32 = 0;
const REG_INPUT: u32 = 1;
const REG_CONST: u32 = 2;
/// The address register in a vertex shader, a texture coordinate in a `ps_2_0`.
const REG_ADDR_OR_TEXTURE: u32 = 3;
const REG_RASTOUT: u32 = 4;
const REG_ATTROUT: u32 = 5;
/// `oT#` in SM2, and the generic `o#` output an SM3 vertex shader declares.
const REG_OUTPUT: u32 = 6;
const REG_COLOROUT: u32 = 8;
const REG_SAMPLER: u32 = 10;

const XYZW: u8 = 0xF;
const XYZ: u8 = 0x7;
const X: u8 = 0x1;
const Y: u8 = 0x2;
const Z: u8 = 0x4;
const W: u8 = 0x8;
const SWIZ_IDENTITY: u8 = 0xE4;
const SWIZ_X: u8 = 0x00;
const SWIZ_Y: u8 = 0x55;
const SWIZ_Z: u8 = 0xAA;
const SWIZ_W: u8 = 0xFF;
/// `dcl_2d` sampler declaration token.
const DCL_2D: u32 = 0x9000_0000;
/// Plain `dcl` token of a `ps_2_0` input register.
const DCL_PLAIN: u32 = 0x8000_0000;
/// Destination-token shift that doubles an SM1 result (`_x2`).
const SHIFT_X2: u32 = 1 << 24;
/// Source-token flag selecting relative addressing, followed by the address token.
const RELATIVE: u32 = 1 << 13;

/// One shader and the entry-point name the layer would give it.
struct Shader {
    input: Input,
    entry: String,
}

/// What an emit starts from.
enum Input {
    FfVs(FfVsKey),
    FfPs(FfPsKey, VariantKey),
    /// A synthetic programmable shader, parsed once so its emit is timed alone.
    Synthetic {
        tokens: Vec<u32>,
        program: DxsoProgram,
        specialization: Specialization,
    },
    /// A cache record, emitted through `ShaderSource::emit`, which reparses its tokens.
    Record(CacheEntry),
}

/// The draw-time inputs a synthetic programmable shader is emitted with.
enum Specialization {
    Vertex { clip_planes: u8 },
    Pixel(VariantKey),
}

impl Shader {
    // Synthetic shaders are named by `fixed_names` once their corpus is built.
    const fn ff_vs(key: FfVsKey) -> Self {
        Self {
            input: Input::FfVs(key),
            entry: String::new(),
        }
    }

    const fn ff_ps(key: FfPsKey, variant: VariantKey) -> Self {
        Self {
            input: Input::FfPs(key, variant),
            entry: String::new(),
        }
    }

    fn synthetic(tokens: Vec<u32>, specialization: Specialization) -> Self {
        let program = dxso::parse(&tokens).expect("synthetic shader parses");
        Self {
            input: Input::Synthetic {
                tokens,
                program,
                specialization,
            },
            entry: String::new(),
        }
    }

    /// A cached record, or `None` for one that keeps no emission inputs.
    fn record(entry: CacheEntry) -> Option<Self> {
        entry.source()?;
        let name = entry.kind.entry_name(entry.key);
        Some(Self {
            input: Input::Record(entry),
            entry: name,
        })
    }

    fn kind(&self) -> CachedKind {
        match &self.input {
            Input::FfVs(_) => CachedKind::FfVs,
            Input::FfPs(..) => CachedKind::FfPs,
            Input::Synthetic { tokens, .. } => {
                let header = tokens[0];
                let major = u8::try_from((header >> 8) & 0xFF).expect("major version fits u8");
                CachedKind::from_programmable(major, header >> 16 == 0xFFFF)
                    .expect("synthetic shaders are SM1 to SM3")
            }
            Input::Record(entry) => entry.kind,
        }
    }

    /// The DXSO tokens of a programmable shader.
    fn tokens(&self) -> Option<&[u32]> {
        match &self.input {
            Input::Synthetic { tokens, .. } => Some(tokens),
            Input::Record(entry) => entry.source().map(ShaderSource::tokens),
            Input::FfVs(_) | Input::FfPs(..) => None,
        }
    }

    fn emit(&self) -> Result<String, String> {
        let entry = &self.entry;
        match &self.input {
            Input::FfVs(key) => Ok(dxso::emit_vs_ff_named(key, entry)),
            Input::FfPs(key, variant) => Ok(dxso::emit_ps_ff_named(key, *variant, entry)),
            Input::Synthetic {
                program,
                specialization,
                ..
            } => match specialization {
                Specialization::Vertex { clip_planes } => dxso::emit_vs_programmable_named(
                    program,
                    entry,
                    u16::MAX,
                    *clip_planes,
                    VsSamplerKinds::default(),
                ),
                Specialization::Pixel(variant) => {
                    dxso::emit_ps_programmable_named(program, *variant, entry)
                }
            }
            .map_err(|error| format!("MSL emission: {error:?}")),
            Input::Record(record) => record
                .source()
                .ok_or_else(|| "record keeps no DXSO".to_owned())?
                .emit(entry),
        }
    }
}

/// A named set of shaders, with the ones that failed to emit already set aside.
struct Corpus {
    name: String,
    shaders: Vec<Shader>,
    failures: usize,
    msl_total: usize,
    msl_max: usize,
}

impl Corpus {
    /// Emit every shader once, keeping those that emit and counting those that fail.
    fn new(name: String, candidates: Vec<Shader>) -> Self {
        let mut corpus = Self {
            name,
            shaders: Vec::with_capacity(candidates.len()),
            failures: 0,
            msl_total: 0,
            msl_max: 0,
        };
        for shader in candidates {
            match shader.emit() {
                Ok(msl) => {
                    corpus.msl_total += msl.len();
                    corpus.msl_max = corpus.msl_max.max(msl.len());
                    corpus.shaders.push(shader);
                }
                Err(error) => {
                    if corpus.failures == 0 {
                        eprintln!("{}: {}: {error}", corpus.name, shader.entry);
                    }
                    corpus.failures += 1;
                }
            }
        }
        corpus
    }

    fn is_programmable(&self) -> bool {
        self.shaders.iter().any(|shader| shader.tokens().is_some())
    }

    /// The metric name of the second timing: whether its emit reparses the tokens.
    fn emit_metric(&self) -> &'static str {
        if self
            .shaders
            .iter()
            .any(|shader| matches!(shader.input, Input::Record(_)))
        {
            "parse_emit"
        } else {
            "emit"
        }
    }
}

/// Median and minimum per-shader time of a measurement, and how many passes it took.
struct Timing {
    median_us: f64,
    min_us: f64,
    passes: u32,
}

impl fmt::Display for Timing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:9.3} {:9.3}", self.median_us, self.min_us)
    }
}

/// The measurements of one corpus.
struct Report {
    parse: Option<Timing>,
    emit: Timing,
}

/// The command line.
struct Args {
    iters: Option<u32>,
    metrics: Option<PathBuf>,
    caches: Vec<PathBuf>,
}

/// Which build the numbers came from, written as the `meta` lines of every metrics file.
struct BuildMeta {
    layer: &'static str,
    image: String,
    arch: &'static str,
    profile: String,
}

fn main() -> ExitCode {
    let args = match parse_args(env::args().skip(1)) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            eprintln!("usage: emit_corpus [--iters N] [--metrics DIR] [mtld3d_shaders.bin ...]");
            return ExitCode::from(2);
        }
    };
    let mut corpora = vec![
        Corpus::new("synthetic_ff".to_owned(), synthetic_ff()),
        Corpus::new("synthetic_sm".to_owned(), synthetic_sm()),
    ];
    // The synthetic inputs are this tree's own, so one that fails to emit is a bug.
    let mut failed = corpora.iter().any(|corpus| corpus.failures > 0);
    for path in &args.caches {
        match load_cache(path, &corpora) {
            Ok(Some(corpus)) => corpora.push(corpus),
            Ok(None) => {}
            Err(error) => {
                eprintln!("{}: {error}", path.display());
                failed = true;
            }
        }
    }
    let meta = BuildMeta {
        layer: mtld3d_shared::identity::BUILD,
        image: mtld3d_shared::identity::image_id().unwrap_or_else(|| "unknown".to_owned()),
        arch: env::consts::ARCH,
        profile: profile(),
    };
    println!(
        "emit_corpus: layer {} arch {} profile {} emitter {SHADER_EMITTER_VERSION:016x}",
        meta.layer, meta.arch, meta.profile
    );
    println!(
        "{:<16} {:>7} {:>5} {:>9} {:>9} {:>9} {:>9} {:>11} {:>8} {:>7}  timed",
        "corpus",
        "shaders",
        "fail",
        "parse med",
        "parse min",
        "emit med",
        "emit min",
        "msl bytes",
        "msl max",
        "passes"
    );
    for corpus in &corpora {
        if corpus.shaders.is_empty() {
            println!(
                "{:<16} {:>7} {:>5}  (nothing to time)",
                corpus.name, 0, corpus.failures
            );
            continue;
        }
        let report = measure(corpus, args.iters);
        let parse = report
            .parse
            .as_ref()
            .map_or_else(|| format!("{:>9} {:>9}", "-", "-"), ToString::to_string);
        println!(
            "{:<16} {:>7} {:>5} {parse} {} {:>11} {:>8} {:>7}  {}",
            corpus.name,
            corpus.shaders.len(),
            corpus.failures,
            report.emit,
            corpus.msl_total,
            corpus.msl_max,
            report.emit.passes,
            corpus.emit_metric()
        );
        if let Some(dir) = &args.metrics
            && let Err(error) = write_metrics(dir, corpus, &report, &meta)
        {
            eprintln!("{}: {error}", dir.display());
            failed = true;
        }
    }
    println!(
        "times are microseconds per shader, median and minimum over the passes; \
         parse_emit reparses the tokens, emit does not"
    );
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args {
        iters: None,
        metrics: None,
        caches: Vec::new(),
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--iters" => {
                let value = args.next().ok_or("--iters needs a count")?;
                let iters = value
                    .parse::<u32>()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| format!("--iters {value}: not a positive count"))?;
                parsed.iters = Some(iters);
            }
            "--metrics" => {
                parsed.metrics = Some(args.next().ok_or("--metrics needs a directory")?.into());
            }
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            _ => parsed.caches.push(arg.into()),
        }
    }
    Ok(parsed)
}

/// The cargo profile this binary was built with, from `target/<triple>/<profile>/examples`.
fn profile() -> String {
    env::current_exe()
        .ok()
        .and_then(|exe| {
            let examples = exe.parent()?;
            (examples.file_name()? == "examples").then_some(())?;
            Some(
                examples
                    .parent()?
                    .file_name()?
                    .to_string_lossy()
                    .into_owned(),
            )
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Read one shader cache into a corpus named after the directory holding it.
///
/// `Ok(None)` is a file this build cannot read as a cache, skipped with a note;
/// only a path that cannot be read at all is an error.
fn load_cache(path: &Path, taken: &[Corpus]) -> Result<Option<Corpus>, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let Ok(header) = read_header(&bytes) else {
        println!(
            "{}: skipped, not an mtld3d shader cache (no MTLD3DSH header)",
            path.display()
        );
        return Ok(None);
    };
    if header.format_version != CACHE_FORMAT_VERSION {
        println!(
            "{}: skipped, container format {} (this build reads {CACHE_FORMAT_VERSION})",
            path.display(),
            header.format_version
        );
        return Ok(None);
    }
    if header.shader_schema_version != SHADER_CACHE_SCHEMA_VERSION {
        eprintln!(
            "{}: shader schema {} (this build writes {SHADER_CACHE_SCHEMA_VERSION}); timing its \
             retained DXSO anyway",
            path.display(),
            header.shader_schema_version
        );
    }
    let records = read_records(&bytes);
    if records.valid_len() < bytes.len() {
        println!(
            "{}: torn or damaged after byte {} of {}; timing the records before it",
            path.display(),
            records.valid_len(),
            bytes.len()
        );
    }
    let records = records.shaders;
    let total = records.len();
    let shaders: Vec<Shader> = records.into_iter().filter_map(Shader::record).collect();
    println!(
        "{}: {total} shader records, {} with retained DXSO",
        path.display(),
        shaders.len()
    );
    Ok(Some(Corpus::new(corpus_name(path, taken), shaders)))
}

/// The cache's directory name as a metrics-safe identifier, unique among `taken`.
fn corpus_name(path: &Path, taken: &[Corpus]) -> String {
    let dir = fs::canonicalize(path)
        .ok()
        .and_then(|full| Some(full.parent()?.file_name()?.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "cache".to_owned());
    let base: String = dir
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let mut name = base.clone();
    let mut suffix = 2;
    while taken.iter().any(|corpus| corpus.name == name) {
        name = format!("{base}_{suffix}");
        suffix += 1;
    }
    name
}

fn measure(corpus: &Corpus, iters: Option<u32>) -> Report {
    let parse = corpus.is_programmable().then(|| {
        time(corpus.shaders.len(), iters, || {
            for tokens in corpus.shaders.iter().filter_map(Shader::tokens) {
                // Every kept shader emitted, so its tokens parse.
                let _ = black_box(dxso::parse(black_box(tokens)));
            }
        })
    });
    let emit = time(corpus.shaders.len(), iters, || {
        for shader in &corpus.shaders {
            let _ = black_box(shader.emit());
        }
    });
    Report { parse, emit }
}

/// Run `pass` repeatedly and report the per-shader median and minimum over the passes.
fn time(shaders: usize, iters: Option<u32>, mut pass: impl FnMut()) -> Timing {
    let warm = Instant::now();
    pass();
    let warm = warm.elapsed().max(Duration::from_nanos(1));
    let passes = iters.unwrap_or_else(|| {
        let needed = TARGET.as_nanos().div_ceil(warm.as_nanos());
        u32::try_from(needed).unwrap_or(u32::MAX).max(MIN_PASSES)
    });
    let mut samples: Vec<Duration> = (0..passes)
        .map(|_| {
            let start = Instant::now();
            pass();
            start.elapsed()
        })
        .collect();
    samples.sort_unstable();
    let count = f64::from(u32::try_from(shaders).expect("a corpus holds fewer than 2^32 shaders"));
    let per_shader_us = |sample: Duration| sample.as_secs_f64() * 1e6 / count;
    let middle = samples.len() / 2;
    // The mean of the two middle samples when the pass count is even.
    let median = if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2
    } else {
        samples[middle]
    };
    Timing {
        median_us: per_shader_us(median),
        min_us: per_shader_us(samples[0]),
        passes,
    }
}

fn write_metrics(
    dir: &Path,
    corpus: &Corpus,
    report: &Report,
    meta: &BuildMeta,
) -> Result<(), String> {
    use fmt::Write as _;

    let bench = format!("host_emit_{}", corpus.name);
    let emit = corpus.emit_metric();
    let mut out = String::new();
    let mut line = |args: fmt::Arguments<'_>| {
        writeln!(out, "{args}").expect("formatting into a String cannot fail");
    };
    line(format_args!("meta {bench} kind host"));
    line(format_args!("meta {bench} layer {}", meta.layer));
    line(format_args!("meta {bench} host_image {}", meta.image));
    line(format_args!("meta {bench} arch {}", meta.arch));
    line(format_args!("meta {bench} profile {}", meta.profile));
    line(format_args!(
        "meta {bench} debug_assertions {}",
        if cfg!(debug_assertions) { "on" } else { "off" }
    ));
    line(format_args!(
        "meta {bench} emitter {SHADER_EMITTER_VERSION:016x}"
    ));
    if let Some(parse) = &report.parse {
        line(format_args!(
            "metric {bench} parse.us_per_shader {:.3} us lower time",
            parse.median_us
        ));
        line(format_args!(
            "metric {bench} parse.us_per_shader_min {:.3} us lower time",
            parse.min_us
        ));
    }
    line(format_args!(
        "metric {bench} {emit}.us_per_shader {:.3} us lower time",
        report.emit.median_us
    ));
    line(format_args!(
        "metric {bench} {emit}.us_per_shader_min {:.3} us lower time",
        report.emit.min_us
    ));
    line(format_args!(
        "metric {bench} shaders {} count higher info",
        corpus.shaders.len()
    ));
    line(format_args!(
        "metric {bench} emit.failures {} count lower exact",
        corpus.failures
    ));
    line(format_args!(
        "metric {bench} msl.bytes_total {} bytes lower exact",
        corpus.msl_total
    ));
    line(format_args!(
        "metric {bench} msl.bytes_max {} bytes lower exact",
        corpus.msl_max
    ));
    fs::create_dir_all(dir).map_err(|error| error.to_string())?;
    let path = dir.join(format!("bench-{bench}.metrics"));
    fs::write(&path, out).map_err(|error| format!("{}: {error}", path.display()))
}

/// D3D enum constant at the key's narrow width.
fn narrow(value: u32) -> u8 {
    u8::try_from(value).expect("D3D9 fixed-function enum value fits u8")
}

/// The vertex layout and render state one synthetic fixed-function vertex key stands for.
struct VsShape {
    geometry: Geometry,
    /// `None` is lighting off; `Some(0)` is lighting on with no light, ambient only.
    lighting: Option<u8>,
    fog: bool,
    specular: bool,
    texturing: Texturing,
}

/// How the texture coordinates of a synthetic fixed-function vertex key are routed.
#[derive(PartialEq, Eq)]
enum Texturing {
    /// Each stage passes its own coordinate set through.
    Plain,
    /// One more stage fed by the camera-space reflection vector through a texture matrix.
    EnvMap,
    /// Stage 0's coordinates go through a `D3DTTFF_COUNT2` texture matrix.
    Scrolled,
    /// Both texture stages read coordinate set 0.
    SharedUv,
}

impl VsShape {
    const fn new(geometry: Geometry, lighting: Option<u8>, fog: bool) -> Self {
        Self {
            geometry,
            lighting,
            fog,
            specular: false,
            texturing: Texturing::Plain,
        }
    }
}

/// The vertex formats a fixed-function game draws with.
enum Geometry {
    /// Position, normal and `tex` texture coordinates: lit world geometry.
    Lit { tex: u8 },
    /// Position, diffuse colour and `tex` texture coordinates: unlit geometry.
    Unlit { tex: u8 },
    /// Pre-transformed position, diffuse colour and `tex` coordinates: interface quads.
    Rhw { tex: u8 },
}

/// Name every shader of a synthetic corpus by its position, in fixed-width names.
///
/// A name derived from the key hash would change width with the hash, and the
/// MSL sizes with it, without any change to the emitter. `u64::MAX - index`
/// always prints as sixteen hex digits.
fn fixed_names(mut shaders: Vec<Shader>) -> Vec<Shader> {
    for (index, shader) in shaders.iter_mut().enumerate() {
        let index = u64::try_from(index).expect("corpus index fits u64");
        shader.entry = shader.kind().entry_name(u64::MAX - index);
    }
    shaders
}

/// About sixty fixed-function vertex keys and fifty pixel keys.
///
/// Lit world geometry with one or two texture sets under zero to four lights
/// (the last of several a point light) or lighting on with ambient alone,
/// vertex fog on and off, specular on and off, an environment-mapped stage fed
/// by the camera-space reflection vector through a texture matrix, a scrolled
/// UV stage through a two-component texture matrix, and two stages sharing one
/// coordinate set; unlit and pre-transformed geometry beside it. The pixel keys
/// are the texture cascades such a game sets: one to three stages of modulate,
/// select, blend and add, each with and without alpha test and fog, and a few
/// with the specular add. Stage state a key does not set keeps its D3D9
/// default, as the layer's key builders read it.
fn synthetic_ff() -> Vec<Shader> {
    let mut shapes = Vec::new();
    for tex in [1, 2] {
        for lighting in [None, Some(1), Some(2), Some(4)] {
            for fog in [false, true] {
                let mut variants = vec![(Texturing::EnvMap, false), (Texturing::Plain, false)];
                if lighting.is_some() {
                    variants.push((Texturing::Plain, true));
                }
                for (texturing, specular) in variants {
                    let mut shape = VsShape::new(Geometry::Lit { tex }, lighting, fog);
                    shape.texturing = texturing;
                    shape.specular = specular;
                    shapes.push(shape);
                }
            }
        }
        for fog in [false, true] {
            shapes.push(VsShape::new(Geometry::Unlit { tex }, None, fog));
        }
    }
    for fog in [false, true] {
        shapes.push(VsShape::new(Geometry::Lit { tex: 1 }, Some(0), fog));
        let mut scrolled = VsShape::new(Geometry::Unlit { tex: 1 }, None, fog);
        scrolled.texturing = Texturing::Scrolled;
        shapes.push(scrolled);
        let mut detail = VsShape::new(Geometry::Lit { tex: 2 }, Some(1), fog);
        detail.texturing = Texturing::SharedUv;
        shapes.push(detail);
    }
    let mut scrolled_lit = VsShape::new(Geometry::Lit { tex: 1 }, Some(1), true);
    scrolled_lit.texturing = Texturing::Scrolled;
    shapes.push(scrolled_lit);
    let mut detail_unlit = VsShape::new(Geometry::Unlit { tex: 2 }, None, false);
    detail_unlit.texturing = Texturing::SharedUv;
    shapes.push(detail_unlit);
    for tex in [0, 1] {
        shapes.push(VsShape::new(Geometry::Rhw { tex }, None, false));
    }
    let mut shaders: Vec<Shader> = shapes
        .iter()
        .map(|shape| Shader::ff_vs(vs_key(shape)))
        .collect();

    for (index, cascade) in cascades().iter().enumerate() {
        for alpha_test in [false, true] {
            for fog in [false, true] {
                for specular_add in [false, true] {
                    // Only the plain modulate cascades are drawn lit with specular.
                    if specular_add && index > 1 {
                        continue;
                    }
                    let variant = VariantKey {
                        alpha_func: if alpha_test {
                            narrow(D3DCMP_GREATEREQUAL)
                        } else {
                            0
                        },
                        fog_mode: if fog { narrow(D3DFOG_LINEAR) } else { 0 },
                        ..VariantKey::default()
                    };
                    shaders.push(Shader::ff_ps(ps_key(cascade, specular_add), variant));
                }
            }
        }
    }
    fixed_names(shaders)
}

fn vs_key(shape: &VsShape) -> FfVsKey {
    let (tex, normal, rhw) = match shape.geometry {
        Geometry::Lit { tex } => (tex, true, false),
        Geometry::Unlit { tex } => (tex, false, false),
        Geometry::Rhw { tex } => (tex, false, true),
    };
    let lit = normal && shape.lighting.is_some();
    let lights = if lit { shape.lighting.unwrap_or(0) } else { 0 };
    let specular = lights > 0 && shape.specular;
    let mut flags = FfVsFlags::COLOR_VERTEX;
    flags.set(FfVsFlags::HAS_NORMAL, normal);
    flags.set(FfVsFlags::HAS_COLOR0, !normal);
    flags.set(FfVsFlags::HAS_RHW, rhw);
    flags.set(FfVsFlags::LIGHTING_ENABLED, lit);
    flags.set(FfVsFlags::SPECULAR_ENABLE, specular);
    // Canonicalized on only where the specular term reads the view vector.
    flags.set(FfVsFlags::LOCAL_VIEWER, specular);
    let active = if lights == 0 {
        0
    } else {
        0xFFu8 >> (8 - lights)
    };
    let point = if lights > 1 { 1u8 << (lights - 1) } else { 0 };
    let sets = if shape.texturing == Texturing::SharedUv {
        1
    } else {
        tex
    };
    let mut key = FfVsKey {
        flags,
        input_tex_coord_count: sets,
        tex_coord_count: tex,
        light_active_mask: active,
        light_directional_mask: active & !point,
        light_spot_mask: 0,
        diffuse_source: narrow(D3DMCS_COLOR1),
        ambient_source: narrow(D3DMCS_MATERIAL),
        specular_source: narrow(D3DMCS_COLOR2),
        emissive_source: narrow(D3DMCS_MATERIAL),
        fog_mode: if shape.fog && !rhw {
            narrow(D3DFOG_LINEAR)
        } else {
            0
        },
        tci_modes: [0; 8],
        // Every stage's `D3DTSS_TEXCOORDINDEX` defaults to its own index.
        tci_coord_indices: core::array::from_fn(|stage| {
            u8::try_from(stage).expect("stage index fits u8")
        }),
        tex_coord_dims: [0; 8],
        tt_flags: [0; 8],
        vertex_blend_count: 0,
        declared_weights_count: 0,
        clip_plane_count: 0,
    };
    for dims in &mut key.tex_coord_dims[..usize::from(sets)] {
        *dims = 2;
    }
    match shape.texturing {
        Texturing::Plain => {}
        Texturing::EnvMap => {
            let stage = usize::from(tex);
            key.tex_coord_count = tex + 1;
            key.tci_modes[stage] = TCI_REFLECTION;
            key.tt_flags[stage] = TT_COUNT3;
        }
        Texturing::Scrolled => key.tt_flags[0] = TT_COUNT2,
        Texturing::SharedUv => key.tci_coord_indices[1] = 0,
    }
    key
}

/// The texture cascades of the synthetic pixel keys, one colour op, args, alpha op, args per stage.
///
/// An argument its operation does not read keeps the D3D9 default
/// (`D3DTA_TEXTURE` for arg 1, `D3DTA_CURRENT` for arg 2).
fn cascades() -> Vec<Vec<[u32; 6]>> {
    let modulate = [
        D3DTOP_MODULATE,
        D3DTA_TEXTURE,
        D3DTA_DIFFUSE,
        D3DTOP_MODULATE,
        D3DTA_TEXTURE,
        D3DTA_DIFFUSE,
    ];
    let select_texture = [
        D3DTOP_SELECTARG1,
        D3DTA_TEXTURE,
        D3DTA_CURRENT,
        D3DTOP_SELECTARG1,
        D3DTA_TEXTURE,
        D3DTA_CURRENT,
    ];
    let modulate2x = [
        D3DTOP_MODULATE2X,
        D3DTA_TEXTURE,
        D3DTA_DIFFUSE,
        D3DTOP_SELECTARG1,
        D3DTA_DIFFUSE,
        D3DTA_CURRENT,
    ];
    let over_current = |op| {
        [
            op,
            D3DTA_TEXTURE,
            D3DTA_CURRENT,
            D3DTOP_SELECTARG1,
            D3DTA_CURRENT,
            D3DTA_CURRENT,
        ]
    };
    vec![
        vec![modulate],
        vec![modulate, over_current(D3DTOP_MODULATE)],
        vec![select_texture],
        vec![modulate2x],
        vec![[
            D3DTOP_SELECTARG1,
            D3DTA_DIFFUSE,
            D3DTA_CURRENT,
            D3DTOP_SELECTARG1,
            D3DTA_DIFFUSE,
            D3DTA_CURRENT,
        ]],
        vec![select_texture, over_current(D3DTOP_BLENDTEXTUREALPHA)],
        vec![modulate, over_current(D3DTOP_ADD)],
        vec![modulate2x, over_current(D3DTOP_MODULATE2X)],
        vec![modulate, over_current(D3DTOP_MODULATEALPHA_ADDCOLOR)],
        vec![modulate, over_current(D3DTOP_BLENDCURRENTALPHA)],
        vec![
            modulate2x,
            over_current(D3DTOP_BLENDTEXTUREALPHA),
            over_current(D3DTOP_MODULATE),
        ],
    ]
}

/// A pixel key whose stages past the cascade keep their D3D9 default state.
fn ps_key(cascade: &[[u32; 6]], specular_add: bool) -> FfPsKey {
    let stages = core::array::from_fn(|index| {
        let defaults = texture_stage_state_defaults(u8::try_from(index).expect("stage fits u8"));
        let state = |ty: u32| defaults[usize::try_from(ty).expect("state index fits usize")];
        let ops = cascade.get(index).copied().unwrap_or_else(|| {
            [
                state(D3DTSS_COLOROP),
                state(D3DTSS_COLORARG1),
                state(D3DTSS_COLORARG2),
                state(D3DTSS_ALPHAOP),
                state(D3DTSS_ALPHAARG1),
                state(D3DTSS_ALPHAARG2),
            ]
        });
        let [
            color_op,
            color_arg1,
            color_arg2,
            alpha_op,
            alpha_arg1,
            alpha_arg2,
        ] = ops;
        // A game binds a texture to the stages it samples and to no others.
        let textured = index < cascade.len()
            && [color_arg1, color_arg2, alpha_arg1, alpha_arg2].contains(&D3DTA_TEXTURE);
        FfStage {
            color_op: narrow(color_op),
            color_arg1: narrow(color_arg1),
            color_arg2: narrow(color_arg2),
            alpha_op: narrow(alpha_op),
            alpha_arg1: narrow(alpha_arg1),
            alpha_arg2: narrow(alpha_arg2),
            flags: if textured {
                FfStageFlags::HAS_TEXTURE
            } else {
                FfStageFlags::empty()
            },
        }
    });
    FfPsKey {
        stages,
        specular_add,
        tt_projected_mask: 0,
    }
}

/// Hand-assembled SM1, SM2 and SM3 shaders in the variants a draw specializes them into.
///
/// Transformed and lit world vertex shaders with vertex fog (`vs_1_1`,
/// `vs_2_0`, `vs_3_0`), with and without one user clip plane; a skinned
/// `vs_2_0` indexing its bone palette through the address register;
/// `ps_1_1`, `ps_1_4` and one- and two-texture `ps_2_0` shaders under alpha
/// test and fog; a normal-mapped `ps_3_0` shader with a specular term. The
/// SM1 streams carry no instruction-length field, as a shader compiler emits
/// them.
fn synthetic_sm() -> Vec<Shader> {
    let mut shaders = Vec::new();
    for tokens in [world_vs_1_1(), world_vs(VS_2_0), world_vs(VS_3_0)] {
        for clip_planes in [0, 1] {
            shaders.push(Shader::synthetic(
                tokens.clone(),
                Specialization::Vertex { clip_planes },
            ));
        }
    }
    shaders.push(Shader::synthetic(
        skinned_vs(),
        Specialization::Vertex { clip_planes: 0 },
    ));
    let alpha_test = narrow(D3DCMP_GREATEREQUAL);
    let fog = narrow(D3DFOG_LINEAR);
    for tokens in [
        detail_ps_1_1(),
        detail_ps_1_4(),
        single_texture_ps(),
        two_texture_ps(),
    ] {
        for (alpha_func, fog_mode) in [(0, 0), (0, fog), (alpha_test, fog)] {
            shaders.push(Shader::synthetic(
                tokens.clone(),
                Specialization::Pixel(VariantKey {
                    alpha_func,
                    fog_mode,
                    ..VariantKey::default()
                }),
            ));
        }
    }
    // SM3 pixel shaders take no automatic fog, so only alpha test varies.
    for alpha_func in [0, alpha_test] {
        shaders.push(Shader::synthetic(
            normal_mapped_ps(),
            Specialization::Pixel(VariantKey {
                alpha_func,
                ..VariantKey::default()
            }),
        ));
    }
    fixed_names(shaders)
}

/// `vs_1_1`: transformed, lit world geometry with a scrolled second coordinate set.
fn world_vs_1_1() -> Vec<u32> {
    let mut out = vec![VS_1_1];
    for (usage_code, index) in [
        (D3DDECLUSAGE_POSITION, 0),
        (D3DDECLUSAGE_NORMAL, 1),
        (D3DDECLUSAGE_TEXCOORD, 2),
    ] {
        out.extend([
            op(OP_DCL, 0),
            usage(usage_code, 0),
            dst(REG_INPUT, index, XYZW),
        ]);
    }
    out.extend([op(OP_DEF, 0), dst(REG_CONST, 8, XYZW)]);
    out.extend([0.0f32, 1.0, 0.0, 0.0].map(f32::to_bits));
    for (row, mask) in [X, Y, Z, W].into_iter().enumerate() {
        let row = u16::try_from(row).expect("row fits u16");
        out.extend([
            op(OP_DP4, 0),
            dst(REG_RASTOUT, 0, mask),
            src(REG_INPUT, 0, SWIZ_IDENTITY),
            src(REG_CONST, row, SWIZ_IDENTITY),
        ]);
    }
    out.extend([
        op(OP_DP3, 0),
        dst(REG_TEMP, 0, X),
        src(REG_INPUT, 1, SWIZ_IDENTITY),
        src(REG_CONST, 4, SWIZ_IDENTITY),
        op(OP_MAX, 0),
        dst(REG_TEMP, 0, X),
        src(REG_TEMP, 0, SWIZ_X),
        src(REG_CONST, 8, SWIZ_X),
        op(OP_MAD, 0),
        dst(REG_ATTROUT, 0, XYZW),
        src(REG_TEMP, 0, SWIZ_X),
        src(REG_CONST, 5, SWIZ_IDENTITY),
        src(REG_CONST, 6, SWIZ_IDENTITY),
        op(OP_MOV, 0),
        dst(REG_OUTPUT, 0, XYZW),
        src(REG_INPUT, 2, SWIZ_IDENTITY),
        op(OP_ADD, 0),
        dst(REG_OUTPUT, 1, XYZW),
        src(REG_INPUT, 2, SWIZ_IDENTITY),
        src(REG_CONST, 9, SWIZ_IDENTITY),
        END,
    ]);
    out
}

/// `ps_1_1`: a texture modulated by the diffuse colour and doubled by a detail texture.
fn detail_ps_1_1() -> Vec<u32> {
    vec![
        PS_1_1,
        op(OP_TEXLD, 0),
        dst(REG_ADDR_OR_TEXTURE, 0, XYZW),
        op(OP_TEXLD, 0),
        dst(REG_ADDR_OR_TEXTURE, 1, XYZW),
        op(OP_MUL, 0),
        dst(REG_TEMP, 0, XYZW),
        src(REG_ADDR_OR_TEXTURE, 0, SWIZ_IDENTITY),
        src(REG_INPUT, 0, SWIZ_IDENTITY),
        op(OP_MUL, 0),
        dst(REG_TEMP, 0, XYZ) | SHIFT_X2,
        src(REG_TEMP, 0, SWIZ_IDENTITY),
        src(REG_ADDR_OR_TEXTURE, 1, SWIZ_IDENTITY),
        END,
    ]
}

/// `ps_1_4`: two textures sampled into temporaries and blended by the second's alpha.
fn detail_ps_1_4() -> Vec<u32> {
    vec![
        PS_1_4,
        op(OP_TEXLD, 0),
        dst(REG_TEMP, 0, XYZW),
        src(REG_ADDR_OR_TEXTURE, 0, SWIZ_IDENTITY),
        op(OP_TEXLD, 0),
        dst(REG_TEMP, 1, XYZW),
        src(REG_ADDR_OR_TEXTURE, 1, SWIZ_IDENTITY),
        op(OP_MUL, 0),
        dst(REG_TEMP, 0, XYZW),
        src(REG_TEMP, 0, SWIZ_IDENTITY),
        src(REG_INPUT, 0, SWIZ_IDENTITY),
        op(OP_LRP, 0),
        dst(REG_TEMP, 0, XYZ),
        src(REG_TEMP, 1, SWIZ_W),
        src(REG_TEMP, 1, SWIZ_IDENTITY),
        src(REG_TEMP, 0, SWIZ_IDENTITY),
        END,
    ]
}

fn op(opcode: u16, operands: u32) -> u32 {
    u32::from(opcode) | (operands << 24)
}

fn reg(kind: u32, index: u16) -> u32 {
    0x8000_0000 | ((kind & 0x7) << 28) | (((kind >> 3) & 0x3) << 11) | u32::from(index)
}

fn dst(kind: u32, index: u16, mask: u8) -> u32 {
    reg(kind, index) | (u32::from(mask) << 16)
}

fn src(kind: u32, index: u16, swizzle: u8) -> u32 {
    reg(kind, index) | (u32::from(swizzle) << 16)
}

fn usage(usage: u8, index: u8) -> u32 {
    0x8000_0000 | u32::from(usage) | (u32::from(index) << 16)
}

fn def(out: &mut Vec<u32>, index: u16, value: [f32; 4]) {
    out.extend([op(OP_DEF, 5), dst(REG_CONST, index, XYZW)]);
    out.extend(value.map(f32::to_bits));
}

fn dcl(out: &mut Vec<u32>, token: u32, kind: u32, index: u16, mask: u8) {
    out.extend([op(OP_DCL, 2), token, dst(kind, index, mask)]);
}

/// Three-operand instruction: `opcode dst, a, b`.
fn alu2(out: &mut Vec<u32>, opcode: u16, dst_token: u32, a: u32, b: u32) {
    out.extend([op(opcode, 3), dst_token, a, b]);
}

/// Four-operand instruction: `opcode dst, a, b, c`.
fn alu3(out: &mut Vec<u32>, opcode: u16, dst_token: u32, [a, b, c]: [u32; 3]) {
    out.extend([op(opcode, 4), dst_token, a, b, c]);
}

/// `dp4` of `input` against the four rows of the matrix at `c[base]` into `out_reg`.
fn transform(out: &mut Vec<u32>, out_reg: (u32, u16), input: u32, base: u16) {
    for (row, mask) in [X, Y, Z, W].into_iter().enumerate() {
        let row = u16::try_from(row).expect("row fits u16");
        alu2(
            out,
            OP_DP4,
            dst(out_reg.0, out_reg.1, mask),
            input,
            src(REG_CONST, base + row, SWIZ_IDENTITY),
        );
    }
}

/// Transformed, lit, textured and fogged world geometry, `vs_2_0` or `vs_3_0`.
fn world_vs(version: u32) -> Vec<u32> {
    let sm3 = version == VS_3_0;
    let mut out = vec![version];
    dcl(
        &mut out,
        usage(D3DDECLUSAGE_POSITION, 0),
        REG_INPUT,
        0,
        XYZW,
    );
    dcl(&mut out, usage(D3DDECLUSAGE_NORMAL, 0), REG_INPUT, 1, XYZW);
    dcl(
        &mut out,
        usage(D3DDECLUSAGE_TEXCOORD, 0),
        REG_INPUT,
        2,
        XYZW,
    );
    // SM3 declares its outputs; SM2 writes the fixed output registers.
    let (position, texcoord, color, fog) = if sm3 {
        dcl(
            &mut out,
            usage(D3DDECLUSAGE_POSITION, 0),
            REG_OUTPUT,
            0,
            XYZW,
        );
        dcl(
            &mut out,
            usage(D3DDECLUSAGE_TEXCOORD, 0),
            REG_OUTPUT,
            1,
            XYZW,
        );
        dcl(&mut out, usage(D3DDECLUSAGE_COLOR, 0), REG_OUTPUT, 2, XYZW);
        dcl(&mut out, usage(D3DDECLUSAGE_FOG, 0), REG_OUTPUT, 3, X);
        (
            (REG_OUTPUT, 0),
            (REG_OUTPUT, 1),
            (REG_OUTPUT, 2),
            (REG_OUTPUT, 3),
        )
    } else {
        (
            (REG_RASTOUT, 0),
            (REG_OUTPUT, 0),
            (REG_ATTROUT, 0),
            (REG_RASTOUT, 1),
        )
    };
    def(&mut out, 8, [0.0, 1.0, 0.0, 0.0]);
    transform(&mut out, position, src(REG_INPUT, 0, SWIZ_IDENTITY), 0);
    alu2(
        &mut out,
        OP_DP3,
        dst(REG_TEMP, 0, X),
        src(REG_INPUT, 1, SWIZ_IDENTITY),
        src(REG_CONST, 4, SWIZ_IDENTITY),
    );
    alu2(
        &mut out,
        OP_MAX,
        dst(REG_TEMP, 0, X),
        src(REG_TEMP, 0, SWIZ_X),
        src(REG_CONST, 8, SWIZ_X),
    );
    alu3(
        &mut out,
        OP_MAD,
        dst(color.0, color.1, XYZW),
        [
            src(REG_TEMP, 0, SWIZ_X),
            src(REG_CONST, 5, SWIZ_IDENTITY),
            src(REG_CONST, 6, SWIZ_IDENTITY),
        ],
    );
    out.extend([
        op(OP_MOV, 2),
        dst(texcoord.0, texcoord.1, XYZW),
        src(REG_INPUT, 2, SWIZ_IDENTITY),
    ]);
    alu2(
        &mut out,
        OP_DP4,
        dst(REG_TEMP, 1, Z),
        src(REG_INPUT, 0, SWIZ_IDENTITY),
        src(REG_CONST, 2, SWIZ_IDENTITY),
    );
    alu3(
        &mut out,
        OP_MAD,
        dst(fog.0, fog.1, X),
        [
            src(REG_TEMP, 1, SWIZ_Z),
            src(REG_CONST, 7, SWIZ_X),
            src(REG_CONST, 7, SWIZ_Y),
        ],
    );
    out.push(END);
    out
}

/// A `vs_2_0` that skins through a bone palette at `c[10 + 3 * index]`.
fn skinned_vs() -> Vec<u32> {
    let mut out = vec![VS_2_0];
    dcl(
        &mut out,
        usage(D3DDECLUSAGE_POSITION, 0),
        REG_INPUT,
        0,
        XYZW,
    );
    dcl(
        &mut out,
        usage(D3DDECLUSAGE_BLENDINDICES, 0),
        REG_INPUT,
        1,
        XYZW,
    );
    dcl(
        &mut out,
        usage(D3DDECLUSAGE_TEXCOORD, 0),
        REG_INPUT,
        2,
        XYZW,
    );
    def(&mut out, 90, [3.0, 1.0, 0.0, 0.0]);
    alu2(
        &mut out,
        OP_MUL,
        dst(REG_TEMP, 0, X),
        src(REG_INPUT, 1, SWIZ_X),
        src(REG_CONST, 90, SWIZ_X),
    );
    out.extend([
        op(OP_MOVA, 2),
        dst(REG_ADDR_OR_TEXTURE, 0, X),
        src(REG_TEMP, 0, SWIZ_X),
    ]);
    for (row, mask) in [(10, X), (11, Y), (12, Z)] {
        out.extend([
            op(OP_DP4, 4),
            dst(REG_TEMP, 1, mask),
            src(REG_INPUT, 0, SWIZ_IDENTITY),
            src(REG_CONST, row, SWIZ_IDENTITY) | RELATIVE,
            src(REG_ADDR_OR_TEXTURE, 0, SWIZ_X),
        ]);
    }
    out.extend([
        op(OP_MOV, 2),
        dst(REG_TEMP, 1, W),
        src(REG_CONST, 90, SWIZ_Y),
    ]);
    transform(
        &mut out,
        (REG_RASTOUT, 0),
        src(REG_TEMP, 1, SWIZ_IDENTITY),
        0,
    );
    out.extend([
        op(OP_MOV, 2),
        dst(REG_OUTPUT, 0, XYZW),
        src(REG_INPUT, 2, SWIZ_IDENTITY),
    ]);
    out.push(END);
    out
}

/// `ps_2_0`: one texture modulated by the diffuse colour.
fn single_texture_ps() -> Vec<u32> {
    let mut out = vec![PS_2_0];
    dcl(&mut out, DCL_PLAIN, REG_ADDR_OR_TEXTURE, 0, XYZW);
    dcl(&mut out, DCL_PLAIN, REG_INPUT, 0, XYZW);
    dcl(&mut out, DCL_2D, REG_SAMPLER, 0, XYZW);
    alu2(
        &mut out,
        OP_TEXLD,
        dst(REG_TEMP, 0, XYZW),
        src(REG_ADDR_OR_TEXTURE, 0, SWIZ_IDENTITY),
        src(REG_SAMPLER, 0, SWIZ_IDENTITY),
    );
    alu2(
        &mut out,
        OP_MUL,
        dst(REG_TEMP, 0, XYZW),
        src(REG_TEMP, 0, SWIZ_IDENTITY),
        src(REG_INPUT, 0, SWIZ_IDENTITY),
    );
    out.extend([
        op(OP_MOV, 2),
        dst(REG_COLOROUT, 0, XYZW),
        src(REG_TEMP, 0, SWIZ_IDENTITY),
    ]);
    out.push(END);
    out
}

/// `ps_2_0`: two textures blended by the second one's alpha, lit and tinted.
fn two_texture_ps() -> Vec<u32> {
    let mut out = vec![PS_2_0];
    dcl(&mut out, DCL_PLAIN, REG_ADDR_OR_TEXTURE, 0, XYZW);
    dcl(&mut out, DCL_PLAIN, REG_ADDR_OR_TEXTURE, 1, XYZW);
    dcl(&mut out, DCL_PLAIN, REG_INPUT, 0, XYZW);
    dcl(&mut out, DCL_2D, REG_SAMPLER, 0, XYZW);
    dcl(&mut out, DCL_2D, REG_SAMPLER, 1, XYZW);
    for index in [0, 1] {
        alu2(
            &mut out,
            OP_TEXLD,
            dst(REG_TEMP, index, XYZW),
            src(REG_ADDR_OR_TEXTURE, index, SWIZ_IDENTITY),
            src(REG_SAMPLER, index, SWIZ_IDENTITY),
        );
    }
    alu3(
        &mut out,
        OP_LRP,
        dst(REG_TEMP, 2, XYZW),
        [
            src(REG_TEMP, 1, SWIZ_W),
            src(REG_TEMP, 1, SWIZ_IDENTITY),
            src(REG_TEMP, 0, SWIZ_IDENTITY),
        ],
    );
    alu2(
        &mut out,
        OP_MUL,
        dst(REG_TEMP, 2, XYZW),
        src(REG_TEMP, 2, SWIZ_IDENTITY),
        src(REG_INPUT, 0, SWIZ_IDENTITY),
    );
    alu2(
        &mut out,
        OP_MUL,
        dst(REG_TEMP, 2, XYZ),
        src(REG_TEMP, 2, SWIZ_IDENTITY),
        src(REG_CONST, 0, SWIZ_IDENTITY),
    );
    out.extend([
        op(OP_MOV, 2),
        dst(REG_COLOROUT, 0, XYZW),
        src(REG_TEMP, 2, SWIZ_IDENTITY),
    ]);
    out.push(END);
    out
}

/// `ps_3_0`: a normal-mapped diffuse texture with a Blinn-style specular term.
fn normal_mapped_ps() -> Vec<u32> {
    let mut out = vec![PS_3_0];
    dcl(
        &mut out,
        usage(D3DDECLUSAGE_TEXCOORD, 0),
        REG_INPUT,
        0,
        XYZW,
    );
    dcl(
        &mut out,
        usage(D3DDECLUSAGE_TEXCOORD, 1),
        REG_INPUT,
        1,
        XYZW,
    );
    dcl(&mut out, usage(D3DDECLUSAGE_COLOR, 0), REG_INPUT, 2, XYZW);
    dcl(&mut out, DCL_2D, REG_SAMPLER, 0, XYZW);
    dcl(&mut out, DCL_2D, REG_SAMPLER, 1, XYZW);
    def(&mut out, 1, [2.0, -1.0, 0.0, 16.0]);
    for sampler in [0, 1] {
        alu2(
            &mut out,
            OP_TEXLD,
            dst(REG_TEMP, sampler, XYZW),
            src(REG_INPUT, 0, SWIZ_IDENTITY),
            src(REG_SAMPLER, sampler, SWIZ_IDENTITY),
        );
    }
    alu3(
        &mut out,
        OP_MAD,
        dst(REG_TEMP, 1, XYZ),
        [
            src(REG_TEMP, 1, SWIZ_IDENTITY),
            src(REG_CONST, 1, SWIZ_X),
            src(REG_CONST, 1, SWIZ_Y),
        ],
    );
    out.extend([
        op(OP_NRM, 2),
        dst(REG_TEMP, 2, XYZ),
        src(REG_TEMP, 1, SWIZ_IDENTITY),
    ]);
    alu2(
        &mut out,
        OP_DP3,
        dst(REG_TEMP, 3, X),
        src(REG_TEMP, 2, SWIZ_IDENTITY),
        src(REG_CONST, 2, SWIZ_IDENTITY),
    );
    alu2(
        &mut out,
        OP_MAX,
        dst(REG_TEMP, 3, X),
        src(REG_TEMP, 3, SWIZ_X),
        src(REG_CONST, 1, SWIZ_Z),
    );
    out.extend([
        op(OP_NRM, 2),
        dst(REG_TEMP, 4, XYZ),
        src(REG_INPUT, 1, SWIZ_IDENTITY),
    ]);
    alu2(
        &mut out,
        OP_DP3,
        dst(REG_TEMP, 3, Y),
        src(REG_TEMP, 2, SWIZ_IDENTITY),
        src(REG_TEMP, 4, SWIZ_IDENTITY),
    );
    alu2(
        &mut out,
        OP_MAX,
        dst(REG_TEMP, 3, Y),
        src(REG_TEMP, 3, SWIZ_Y),
        src(REG_CONST, 1, SWIZ_Z),
    );
    alu2(
        &mut out,
        OP_POW,
        dst(REG_TEMP, 3, Z),
        src(REG_TEMP, 3, SWIZ_Y),
        src(REG_CONST, 1, SWIZ_W),
    );
    alu2(
        &mut out,
        OP_MUL,
        dst(REG_TEMP, 0, XYZ),
        src(REG_TEMP, 0, SWIZ_IDENTITY),
        src(REG_TEMP, 3, SWIZ_X),
    );
    alu3(
        &mut out,
        OP_MAD,
        dst(REG_TEMP, 0, XYZ),
        [
            src(REG_CONST, 3, SWIZ_IDENTITY),
            src(REG_TEMP, 3, SWIZ_Z),
            src(REG_TEMP, 0, SWIZ_IDENTITY),
        ],
    );
    alu2(
        &mut out,
        OP_MUL,
        dst(REG_COLOROUT, 0, XYZW),
        src(REG_TEMP, 0, SWIZ_IDENTITY),
        src(REG_INPUT, 2, SWIZ_IDENTITY),
    );
    out.push(END);
    out
}
