//! Check a benchmark's declared frame against a frame a game dumped, to calibrate the scene.
//!
//! Ctrl+Shift+F7 in a game makes the layer log a few consecutive frames
//! draw by draw (`[dump]` lines at info level: the frame's start and end,
//! every bind, clear and copy, and one line per draw naming its render
//! target, depth surface, shaders and textures). A benchmark that stands
//! for that game writes the frame it builds as `shape` lines in its metrics
//! file, one per pass. `bench-shape` reads the last complete dumped frame,
//! splits it into passes along the main lines of the layer's own splits,
//! and prints the two side by side.
//!
//! The splits modelled: a draw whose render target 0 or depth surface
//! differs from the draw before it, and the first draw after a
//! `StretchRect` or a `ColorFill`, both of which end the layer's pass. A
//! target set and set back with no draw in between opens no pass, which is
//! what the layer's join of two passes on the same attachments leaves. The
//! splits not modelled, since the dump does not show them or shows them
//! only indirectly: a clear inside a pass that the layer cannot draw as a
//! quad, an sRGB-write toggle, render targets 1 to 3 changing, a query or
//! readback that flushes the frame, texture-upload passes, and the cases
//! where the join keeps two passes apart. Clears with no draw after them are
//! left out, as the layer folds them into the next pass's load action or
//! drops them. The pass list is a calibration aid, not the layer's pass list.
//!
//! The game's passes and the benchmark's are paired in order, each pair of
//! one kind (drawing to a target of the back buffer's size, or offscreen)
//! and as close in size relative to its back buffer as the order allows, so
//! a pass one side lacks leaves one unpaired row rather than shifting every
//! row after it. Per pair the check compares the draw count (within 10 %),
//! the share of draws with a fixed-function vertex or pixel stage (within
//! 10 points) and the textures bound per draw (within 1.0); an unpaired
//! pass is flagged too. Render-target sizes are printed as ratios to each
//! side's back buffer, since a game's window and a benchmark's differ, and
//! are not judged.
//!
//! Each pair is also compared on its state mix (see [`StateMix`] for the
//! definitions): the shares of draws that blend, alpha-test, leave depth
//! writes off, cull nothing or write no colour to render target 0 (within
//! 10 points), and how often the shaders, the stage-0 texture, the blend
//! and alpha-test states and the cull mode switch from draw to draw and how
//! many distinct shaders and textures the pass uses (within 15 %, and never
//! closer than 5). A `shape` line without those keys, which older metrics
//! files and scenes that do not declare them write, still parses: its
//! state mix is printed as not reported and not judged. Exit code 0 when
//! everything judged is within tolerance, 1 otherwise.

use std::{collections::BTreeSet, fmt::Write as _, fs, path::Path, process::ExitCode};

use mtld3d_types::D3DCULL_NONE;

use super::{metrics, shape::log_message};

/// How far a pass's draw count may be off, as a fraction of the game's.
const DRAWS_TOLERANCE: f64 = 0.10;

/// How far a pass's fixed-function share may be off, in percentage points.
const FF_TOLERANCE: f64 = 10.0;

/// How far a pass's textures per draw may be off.
const TEX_TOLERANCE: f64 = 1.0;

/// How far a pass's state shares (blend, alpha test and the rest) may be off, in percentage points.
const STATE_SHARE_TOLERANCE: f64 = 10.0;

/// How far a pass's switch and distinct counts may be off, as a fraction of the game's.
const STATE_COUNT_TOLERANCE: f64 = 0.15;

/// The difference a switch or distinct count is always allowed, so a small pass is not judged on 1.
const STATE_COUNT_FLOOR: f64 = 5.0;

/// The keys of a pass's state mix on a `shape` line, in [`StateMix::values`] order.
const STATE_KEYS: [&str; 14] = [
    "blend",
    "atest",
    "zwrite_off",
    "cull_none",
    "cmask0",
    "vs_sw",
    "ps_sw",
    "tex_sw",
    "blend_sw",
    "atest_sw",
    "cull_sw",
    "vs_n",
    "ps_n",
    "tex_n",
];

/// How many of [`STATE_KEYS`], from the first, are counts of draws judged as a share of the pass.
const STATE_SHARES: usize = 5;

/// The keys every `shape` line carries.
const BASE_KEYS: [&str; 4] = ["draws", "ff_vs", "ff_ps", "tex_per_draw"];

/// Every key a `shape` line may carry: [`BASE_KEYS`], then [`STATE_KEYS`].
const SHAPE_KEYS: [&str; BASE_KEYS.len() + STATE_KEYS.len()] = {
    let mut keys = [""; BASE_KEYS.len() + STATE_KEYS.len()];
    let mut slot = 0;
    while slot < keys.len() {
        keys[slot] = if slot < BASE_KEYS.len() {
            BASE_KEYS[slot]
        } else {
            STATE_KEYS[slot - BASE_KEYS.len()]
        };
        slot += 1;
    }
    keys
};

/// What a state-mix cell shows for a benchmark pass that does not report its state mix.
const NOT_REPORTED: &str = "n/r";

/// The score of pairing two passes, which no difference in their sizes outweighs.
const PAIR_SCORE: f64 = 1000.0;

/// The prefix of a frame-dump message.
const DUMP_PREFIX: &str = "[dump] ";

/// The dump events that end the layer's current pass.
const PASS_ENDING_COPIES: [&str; 2] = ["StretchRect(", "ColorFill("];

/// A render-target or back-buffer size.
#[derive(Debug, PartialEq, Eq)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl Size {
    /// The first `<W>x<H>` word of `text`.
    fn find(text: &str) -> Option<Self> {
        text.split_whitespace().find_map(Self::parse)
    }

    /// `<W>x<H>`, both nonzero.
    fn parse(word: &str) -> Option<Self> {
        let (width, height) = word.split_once('x')?;
        let size = Self {
            width: width.parse().ok()?,
            height: height.parse().ok()?,
        };
        (size.width > 0 && size.height > 0).then_some(size)
    }

    /// This size relative to `whole`, as `<w>x<h>` to two places.
    fn ratio(&self, whole: &Self) -> String {
        format!(
            "{:.2}x{:.2}",
            f64::from(self.width) / f64::from(whole.width),
            f64::from(self.height) / f64::from(whole.height)
        )
    }
}

/// One pass of the game's dumped frame.
#[derive(Debug, PartialEq, Eq)]
pub struct GamePass {
    /// The render target as the draw lines name it.
    pub target: String,
    /// The depth surface as the draw lines name it.
    pub depth: String,
    pub size: Option<Size>,
    pub draws: u32,
    pub ff_vs: u32,
    pub ff_ps: u32,
    /// Textures bound over all the pass's draws.
    pub textures: u32,
    pub state: StateMix,
}

impl GamePass {
    /// The pass of `draws`, which share a render target and depth surface; `None` for no draw.
    fn of(draws: &[Draw<'_>]) -> Option<Self> {
        let first = draws.first()?;
        Some(Self {
            target: first.target.to_owned(),
            depth: first.depth.to_owned(),
            size: Size::find(first.target),
            draws: narrow(draws.len()),
            ff_vs: count_draws(draws, |d| d.vs == "ff"),
            ff_ps: count_draws(draws, |d| d.ps == "ff"),
            textures: narrow(draws.iter().map(|d| d.texture_ids().count()).sum()),
            state: StateMix::of(draws),
        })
    }
}

/// The last complete frame a game log dumped.
#[derive(Debug)]
pub struct GameFrame {
    /// How many complete frames the log dumped; the one read is the last.
    pub frames: usize,
    pub backbuffer: Size,
    pub passes: Vec<GamePass>,
    /// Log lines dropped as repeats of the line before them.
    pub repeats: usize,
}

/// One `shape` line of a benchmark: the pass it builds.
#[derive(Debug, PartialEq)]
pub struct BenchPass {
    pub size: Size,
    pub draws: u32,
    pub ff_vs: u32,
    pub ff_ps: u32,
    pub tex_per_draw: f64,
    /// `None` when the line carries none of the state keys: not reported, so not judged.
    pub state: Option<StateMix>,
}

/// A benchmark's declared frame.
#[derive(Debug)]
pub struct BenchFrame {
    pub passes: Vec<BenchPass>,
    pub backbuffer: Size,
    /// Where the back-buffer size came from: its meta line, or the last pass.
    pub backbuffer_from: &'static str,
}

/// The state mix of one pass's draws, each value a count; the field names are the `shape` keys.
///
/// These definitions are shared with the benchmarks, which write the same
/// counts for the passes they build (`PassShape` in the end-to-end suite).
/// The first five count draws and are judged as a share of the pass's
/// draws. The switch counts go over the pass's draws after its first and
/// count each draw whose value differs from the draw before it in the same
/// pass. The values are read from the `[dump]` draw line's fields: `vs=`
/// and `ps=`, `z=[enable,write,func]`, `blend=[...]`, `cull=`,
/// `cw=[rt0,rt1,rt2,rt3]`, `alpha=[enable,func,ref]` and
/// `tex=[s<stage>=<id>/<format>/<W>x<H> ...]`.
#[derive(Debug, PartialEq, Eq)]
pub struct StateMix {
    /// Draws with alpha blending enabled (the first field of `blend=` nonzero).
    pub blend: u32,
    /// Draws with alpha test enabled (the first field of `alpha=` nonzero).
    pub atest: u32,
    /// Draws whose depth-write field (the second of `z=`) is 0, whether depth is enabled or not.
    pub zwrite_off: u32,
    /// Draws with cull mode `D3DCULL_NONE`.
    pub cull_none: u32,
    /// Draws whose render target 0 colour write mask (the first of `cw=`) is 0.
    pub cmask0: u32,
    /// Draws whose vertex shader differs from the previous draw's; `ff` is one value like an id.
    pub vs_sw: u32,
    /// Draws whose pixel shader differs from the previous draw's; `ff` is one value like an id.
    pub ps_sw: u32,
    /// Draws whose stage-0 texture id differs from the previous draw's; no texture is one value.
    pub tex_sw: u32,
    /// Draws whose whole `blend=` tuple differs from the previous draw's, enabled or not.
    pub blend_sw: u32,
    /// Draws whose whole `alpha=` tuple (enable, func, ref) differs from the previous draw's.
    pub atest_sw: u32,
    /// Draws whose cull mode differs from the previous draw's.
    pub cull_sw: u32,
    /// Distinct programmable vertex shaders over the pass; fixed function is not counted.
    pub vs_n: u32,
    /// Distinct programmable pixel shaders over the pass; fixed function is not counted.
    pub ps_n: u32,
    /// Distinct texture ids over every stage of every draw of the pass.
    pub tex_n: u32,
}

impl StateMix {
    /// The values in [`STATE_KEYS`] order.
    #[must_use]
    pub const fn values(&self) -> [u32; STATE_KEYS.len()] {
        [
            self.blend,
            self.atest,
            self.zwrite_off,
            self.cull_none,
            self.cmask0,
            self.vs_sw,
            self.ps_sw,
            self.tex_sw,
            self.blend_sw,
            self.atest_sw,
            self.cull_sw,
            self.vs_n,
            self.ps_n,
            self.tex_n,
        ]
    }

    /// The state mix of `values` in [`STATE_KEYS`] order.
    const fn from_values(values: [u32; STATE_KEYS.len()]) -> Self {
        let [
            blend,
            atest,
            zwrite_off,
            cull_none,
            cmask0,
            vs_sw,
            ps_sw,
            tex_sw,
            blend_sw,
            atest_sw,
            cull_sw,
            vs_n,
            ps_n,
            tex_n,
        ] = values;
        Self {
            blend,
            atest,
            zwrite_off,
            cull_none,
            cmask0,
            vs_sw,
            ps_sw,
            tex_sw,
            blend_sw,
            atest_sw,
            cull_sw,
            vs_n,
            ps_n,
            tex_n,
        }
    }

    /// The state mix of one pass's draws, in draw order.
    fn of(draws: &[Draw<'_>]) -> Self {
        let count = |test: fn(&Draw<'_>) -> bool| count_draws(draws, test);
        let switches = |differs: &dyn Fn(&Draw<'_>, &Draw<'_>) -> bool| {
            narrow(
                draws
                    .windows(2)
                    .filter(|pair| differs(&pair[0], &pair[1]))
                    .count(),
            )
        };
        let distinct =
            |ids: &mut dyn Iterator<Item = &str>| narrow(ids.collect::<BTreeSet<_>>().len());
        let programmable = |shader: &&str| !matches!(*shader, "ff" | "none");
        Self {
            blend: count(|d| d.blend_enable != 0),
            atest: count(|d| d.alpha_enable != 0),
            zwrite_off: count(|d| d.z_write == 0),
            cull_none: count(|d| d.cull == D3DCULL_NONE),
            cmask0: count(|d| d.color_mask == 0),
            vs_sw: switches(&|a, b| a.vs != b.vs),
            ps_sw: switches(&|a, b| a.ps != b.ps),
            tex_sw: switches(&|a, b| a.stage0_texture() != b.stage0_texture()),
            blend_sw: switches(&|a, b| a.blend != b.blend),
            atest_sw: switches(&|a, b| a.alpha != b.alpha),
            cull_sw: switches(&|a, b| a.cull != b.cull),
            vs_n: distinct(&mut draws.iter().map(|d| d.vs).filter(programmable)),
            ps_n: distinct(&mut draws.iter().map(|d| d.ps).filter(programmable)),
            tex_n: distinct(&mut draws.iter().flat_map(Draw::texture_ids)),
        }
    }
}

/// Run the check of `bench-shape`: print the table, exit 0 within tolerance and 1 outside.
///
/// # Errors
///
/// Returns a message when either file cannot be read, the log holds no
/// complete dumped frame, or the metrics file's `shape` lines are malformed.
pub fn check(game_log: &Path, metrics_path: &Path) -> Result<ExitCode, String> {
    let bytes = fs::read(game_log).map_err(|e| format!("{}: {e}", game_log.display()))?;
    let game = parse_game_log(&String::from_utf8_lossy(&bytes))
        .map_err(|reason| format!("{}: {reason}", game_log.display()))?;
    let file = metrics::read(metrics_path)?;
    let bench_name = metrics_path
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(metrics::bench_of)
        .unwrap_or_default();
    let bench = parse_bench(&file.shape, file.meta.get("backbuffer").map(String::as_str))
        .map_err(|reason| format!("{}: {reason}", metrics_path.display()))?;
    let (text, within) = render(
        &game,
        &bench,
        &format!("game {}", game_log.display()),
        &format!("bench {bench_name}"),
    );
    print!("{text}");
    Ok(if within {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

/// Read the last complete dumped frame of a game log into passes.
///
/// A log that carries every line twice is read once: a dump line that
/// repeats the line before it word for word, timestamp included, is
/// dropped. That never changes a pass, since a draw line carries its own
/// sequence number and a repeated bind or copy is the same boundary twice.
///
/// # Errors
///
/// Returns a message when no frame is complete, a draw line of the frame
/// lacks a field the passes are built from, the frame's draw count
/// disagrees with the draw lines read, or nothing names the back buffer's
/// size.
pub fn parse_game_log(log: &str) -> Result<GameFrame, String> {
    let mut events = Vec::new();
    let mut previous: Option<&str> = None;
    let mut repeats = 0;
    for line in log.lines() {
        let Some(event) = log_message(line).and_then(|(_, m)| m.strip_prefix(DUMP_PREFIX)) else {
            continue;
        };
        if previous == Some(line) {
            repeats += 1;
            continue;
        }
        previous = Some(line);
        events.push(event);
    }
    let mut frames: Vec<(Vec<&str>, &str)> = Vec::new();
    let mut current: Option<Vec<&str>> = None;
    for event in events {
        if event.starts_with("frame start") {
            current = Some(Vec::new());
        } else if let Some(end) = event.strip_prefix("frame end: ") {
            if let Some(body) = current.take() {
                frames.push((body, end));
            }
        } else if let Some(body) = current.as_mut() {
            body.push(event);
        }
    }
    let count = frames.len();
    let Some((body, end)) = frames.pop() else {
        return Err(
            "no complete [dump] frame (a frame start followed by its frame end); press \
             Ctrl+Shift+F7 in the game to dump one"
                .to_owned(),
        );
    };
    let declared = end
        .split_whitespace()
        .next()
        .and_then(|draws| draws.parse::<u32>().ok())
        .ok_or_else(|| format!("frame end line {end:?} names no draw count"))?;
    let passes = game_passes(&body)?;
    let read: u32 = passes.iter().map(|pass| pass.draws).sum();
    if read != declared {
        return Err(format!(
            "frame {count} ends with {declared} draws, but {read} draw lines were read in it"
        ));
    }
    let backbuffer = body
        .iter()
        .find_map(|event| {
            let (_, rest) = event.split_once("backbuffer ")?;
            Size::parse(rest.split_whitespace().next()?)
        })
        .ok_or_else(|| format!("frame {count} names no back buffer size"))?;
    Ok(GameFrame {
        frames: count,
        backbuffer,
        passes,
        repeats,
    })
}

/// Parse a benchmark's `shape` lines into its declared frame.
///
/// Each line is `pass <i> <W>x<H> draws=<n> ff_vs=<n> ff_ps=<n> tex_per_draw=<x>`,
/// optionally followed by every key of [`STATE_KEYS`] as `<key>=<n>`: a line
/// with none of them reports no state mix, and one with some but not all is
/// an error. The back buffer is `backbuffer`, a `meta <bench> backbuffer <W>x<H>`
/// value, when the file has one, and the last pass's target otherwise:
/// a frame ends on the back buffer it presents.
///
/// # Errors
///
/// Returns a message for a line that is not that shape, a key missing or
/// unknown, state keys only in part, passes out of order, or no pass at all.
pub fn parse_bench(lines: &[String], backbuffer: Option<&str>) -> Result<BenchFrame, String> {
    let mut passes = Vec::new();
    for line in lines {
        let words: Vec<&str> = line.split_whitespace().collect();
        let [kind, index, size, rest @ ..] = words.as_slice() else {
            return Err(format!(
                "shape line {line:?} is not pass <i> <W>x<H> <key>=<value>..."
            ));
        };
        if *kind != "pass" || index.parse::<usize>().ok() != Some(passes.len()) {
            return Err(format!(
                "shape line {line:?} is not pass {} <W>x<H> ...: the passes are listed from 0 \
                 in order",
                passes.len()
            ));
        }
        let size = Size::parse(size).ok_or_else(|| format!("shape line {line:?}: bad size"))?;
        let keys = SHAPE_KEYS;
        let mut values = [None::<&str>; SHAPE_KEYS.len()];
        for word in rest {
            let (key, value) = word
                .split_once('=')
                .ok_or_else(|| format!("shape line {line:?}: {word:?} is not key=value"))?;
            let slot = keys
                .iter()
                .position(|known| *known == key)
                .ok_or_else(|| format!("shape line {line:?}: unknown key {key:?}"))?;
            if values[slot].replace(value).is_some() {
                return Err(format!("shape line {line:?}: repeats {key}="));
            }
        }
        let get = |slot: usize| {
            values[slot].ok_or_else(|| format!("shape line {line:?}: no {}=", keys[slot]))
        };
        let count = |slot: usize| {
            get(slot)?
                .parse::<u32>()
                .map_err(|_| format!("shape line {line:?}: {} is not a count", keys[slot]))
        };
        let tex_per_draw = get(3)?
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .ok_or_else(|| format!("shape line {line:?}: tex_per_draw is not a number"))?;
        let state_slots = BASE_KEYS.len()..keys.len();
        let missing: Vec<&str> = state_slots
            .clone()
            .filter(|&slot| values[slot].is_none())
            .map(|slot| keys[slot])
            .collect();
        let state = if missing.len() == STATE_KEYS.len() {
            None
        } else if missing.is_empty() {
            let mut state = [0; STATE_KEYS.len()];
            for (value, slot) in state.iter_mut().zip(state_slots) {
                *value = count(slot)?;
            }
            Some(StateMix::from_values(state))
        } else {
            return Err(format!(
                "shape line {line:?}: carries part of the state mix but not {}",
                missing.join(", ")
            ));
        };
        let draws = count(0)?;
        if let Some(state) = &state {
            let values = state.values();
            let over = values[..STATE_SHARES]
                .iter()
                .zip(STATE_KEYS)
                .find(|&(&value, _)| value > draws);
            if let Some((value, key)) = over {
                return Err(format!(
                    "shape line {line:?}: {key}={value} counts more draws than the pass's {draws}"
                ));
            }
        }
        passes.push(BenchPass {
            size,
            draws,
            ff_vs: count(1)?,
            ff_ps: count(2)?,
            tex_per_draw,
            state,
        });
    }
    let (backbuffer, backbuffer_from) = if let Some(value) = backbuffer {
        let size = Size::parse(value.trim())
            .ok_or_else(|| format!("meta backbuffer {value:?} is not <W>x<H>"))?;
        (size, "meta backbuffer")
    } else {
        let last = passes
            .last()
            .ok_or_else(|| "no shape line: the benchmark declares no pass".to_owned())?;
        let size = Size {
            width: last.size.width,
            height: last.size.height,
        };
        (size, "the last pass")
    };
    Ok(BenchFrame {
        passes,
        backbuffer,
        backbuffer_from,
    })
}

/// The side-by-side table of `game` and `bench`, and whether every check is within tolerance.
#[must_use]
pub fn render(
    game: &GameFrame,
    bench: &BenchFrame,
    game_name: &str,
    bench_name: &str,
) -> (String, bool) {
    let mut out = String::new();
    let draws = |passes: &mut dyn Iterator<Item = u32>| passes.sum::<u32>();
    let _ = writeln!(
        out,
        "bench-shape: {game_name}: frame {} of {}, {} draws in {} passes, back buffer {}x{}{}",
        game.frames,
        game.frames,
        draws(&mut game.passes.iter().map(|p| p.draws)),
        game.passes.len(),
        game.backbuffer.width,
        game.backbuffer.height,
        if game.repeats > 0 {
            format!(" ({} repeated log lines read once)", game.repeats)
        } else {
            String::new()
        }
    );
    let _ = writeln!(
        out,
        "bench-shape: {bench_name}: {} draws in {} passes, back buffer {}x{} (from {})",
        draws(&mut bench.passes.iter().map(|p| p.draws)),
        bench.passes.len(),
        bench.backbuffer.width,
        bench.backbuffer.height,
        bench.backbuffer_from
    );
    let _ = writeln!(
        out,
        "tolerance: draws {:.0} %, ff share {FF_TOLERANCE:.0} points, tex/draw {TEX_TOLERANCE:.1}, \
         state shares {STATE_SHARE_TOLERANCE:.0} points, switch and distinct counts {:.0} % but \
         never under {STATE_COUNT_FLOOR:.0}; sizes are ratios to the back buffer and not judged",
        DRAWS_TOLERANCE * 100.0,
        STATE_COUNT_TOLERANCE * 100.0
    );
    let header = [
        "game/bench",
        "game rt",
        "size",
        "bench size",
        "draws",
        "ff_vs %",
        "ff_ps %",
        "tex/draw",
        "flags",
    ]
    .map(str::to_owned);
    let state_header = |keys: &[&str], suffix: &str| {
        std::iter::once("game/bench".to_owned())
            .chain(keys.iter().map(|key| format!("{key}{suffix}")))
            .collect::<Vec<_>>()
    };
    let (share_keys, count_keys) = STATE_KEYS.split_at(STATE_SHARES);
    let mut cells: Vec<Vec<String>> = Vec::new();
    let mut share_cells: Vec<Vec<String>> = Vec::new();
    let mut count_cells: Vec<Vec<String>> = Vec::new();
    let mut flagged = 0;
    let mut not_reported = 0;
    let pairs = pair_passes(game, bench);
    for &(game_index, bench_index) in &pairs {
        let game_pass = game_index.map(|index| &game.passes[index]);
        let bench_pass = bench_index.map(|index| &bench.passes[index]);
        let flags = flags(game_pass, bench_pass);
        if !flags.is_empty() {
            flagged += 1;
        }
        if bench_pass.is_some_and(|p| p.state.is_none()) {
            not_reported += 1;
        }
        let pair = |g: Option<String>, b: Option<String>| {
            format!(
                "{} / {}",
                g.unwrap_or_else(|| "-".to_owned()),
                b.unwrap_or_else(|| "-".to_owned())
            )
        };
        let shown = |index: Option<usize>| index.map_or_else(|| "-".to_owned(), |i| i.to_string());
        let label = format!("{}/{}", shown(game_index), shown(bench_index));
        // A state value of each side as its cell shows it: a share of the
        // pass's draws for the first `STATE_SHARES` keys, a count after.
        let state_cell = |slot: usize| {
            let value_cell = |value: u32, draws: u32| {
                if slot < STATE_SHARES {
                    format!("{:.0}", share(value, draws))
                } else {
                    value.to_string()
                }
            };
            pair(
                game_pass.map(|p| value_cell(p.state.values()[slot], p.draws)),
                bench_pass.map(|p| {
                    p.state.as_ref().map_or_else(
                        || NOT_REPORTED.to_owned(),
                        |state| value_cell(state.values()[slot], p.draws),
                    )
                }),
            )
        };
        share_cells.push(
            std::iter::once(label.clone())
                .chain((0..STATE_SHARES).map(state_cell))
                .collect(),
        );
        count_cells.push(
            std::iter::once(label.clone())
                .chain((STATE_SHARES..STATE_KEYS.len()).map(state_cell))
                .collect(),
        );
        cells.push(vec![
            label,
            game_pass.map_or_else(|| "-".to_owned(), |p| target_kind(&p.target).to_owned()),
            game_pass
                .and_then(|p| p.size.as_ref())
                .map_or_else(|| "-".to_owned(), |size| size.ratio(&game.backbuffer)),
            bench_pass.map_or_else(|| "-".to_owned(), |p| p.size.ratio(&bench.backbuffer)),
            pair(
                game_pass.map(|p| p.draws.to_string()),
                bench_pass.map(|p| p.draws.to_string()),
            ),
            pair(
                game_pass.map(|p| format!("{:.0}", share(p.ff_vs, p.draws))),
                bench_pass.map(|p| format!("{:.0}", share(p.ff_vs, p.draws))),
            ),
            pair(
                game_pass.map(|p| format!("{:.0}", share(p.ff_ps, p.draws))),
                bench_pass.map(|p| format!("{:.0}", share(p.ff_ps, p.draws))),
            ),
            pair(
                game_pass.map(|p| format!("{:.2}", tex_per_draw(p))),
                bench_pass.map(|p| format!("{:.2}", p.tex_per_draw)),
            ),
            flags.join(","),
        ]);
    }
    write_table(&mut out, &header, &cells);
    let _ = writeln!(
        out,
        "state shares, % of the pass's draws (game / bench; flags in the table above):"
    );
    write_table(&mut out, &state_header(share_keys, " %"), &share_cells);
    let _ = writeln!(
        out,
        "state switches and distinct counts (game / bench; flags in the table above):"
    );
    write_table(&mut out, &state_header(count_keys, ""), &count_cells);
    if not_reported > 0 {
        let _ = writeln!(
            out,
            "{NOT_REPORTED}: not reported; the bench's shape line for that pass carries no state \
             keys (a metrics file older than them, or a scene that does not declare them), so its \
             state mix is not judged"
        );
    }
    let count_matches = game.passes.len() == bench.passes.len();
    let within = count_matches && flagged == 0;
    let _ = writeln!(
        out,
        "bench-shape: {}: {} game passes, {} bench passes; {flagged} of {} rows flagged{}",
        if within {
            "WITHIN TOLERANCE"
        } else {
            "OUT OF TOLERANCE"
        },
        game.passes.len(),
        bench.passes.len(),
        pairs.len(),
        if not_reported > 0 {
            format!("; state mix not reported for {not_reported} bench passes, not judged")
        } else {
            String::new()
        }
    );
    (out, within)
}

/// The rows of the table: the game's and the benchmark's passes paired, in order.
///
/// An alignment in the manner of a longest common subsequence: two passes
/// may pair when both draw to a target of their back buffer's size or both
/// draw offscreen, a pair scores [`PAIR_SCORE`] less how far apart their
/// sizes are relative to their back buffers, and the alignment with the
/// highest total wins. Of equal alignments, the one that pairs earlier
/// passes first wins.
fn pair_passes(game: &GameFrame, bench: &BenchFrame) -> Vec<(Option<usize>, Option<usize>)> {
    let (n, m) = (game.passes.len(), bench.passes.len());
    let pair = |i: usize, j: usize| pair_score(&game.passes[i], game, &bench.passes[j], bench);
    let mut best = vec![vec![0.0_f64; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            let skip = best[i + 1][j].max(best[i][j + 1]);
            best[i][j] = pair(i, j).map_or(skip, |score| skip.max(score + best[i + 1][j + 1]));
        }
    }
    let mut rows = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n || j < m {
        let paired = (i < n && j < m)
            .then(|| pair(i, j))
            .flatten()
            .is_some_and(|score| score + best[i + 1][j + 1] >= best[i + 1][j].max(best[i][j + 1]));
        if paired {
            rows.push((Some(i), Some(j)));
            i += 1;
            j += 1;
        } else if j == m || (i < n && best[i + 1][j] >= best[i][j + 1]) {
            rows.push((Some(i), None));
            i += 1;
        } else {
            rows.push((None, Some(j)));
            j += 1;
        }
    }
    rows
}

/// The score of pairing a game pass with a benchmark pass; `None` when they are of two kinds.
fn pair_score(
    game_pass: &GamePass,
    game: &GameFrame,
    bench_pass: &BenchPass,
    bench: &BenchFrame,
) -> Option<f64> {
    let game_full = game_pass.target.starts_with("backbuffer")
        || game_pass.size.as_ref() == Some(&game.backbuffer);
    let bench_full = bench_pass.size == bench.backbuffer;
    if game_full != bench_full {
        return None;
    }
    let distance = game_pass.size.as_ref().map_or(0.0, |size| {
        let relative = |part: u32, whole: u32| f64::from(part) / f64::from(whole);
        let width = relative(size.width, game.backbuffer.width)
            / relative(bench_pass.size.width, bench.backbuffer.width);
        let height = relative(size.height, game.backbuffer.height)
            / relative(bench_pass.size.height, bench.backbuffer.height);
        width.ln().abs() + height.ln().abs()
    });
    Some(PAIR_SCORE - distance.min(PAIR_SCORE / 10.0))
}

/// The passes of one dumped frame's events, see the module doc for where one ends.
///
/// # Errors
///
/// Returns a message for a draw line that lacks a field a pass is built from.
fn game_passes(events: &[&str]) -> Result<Vec<GamePass>, String> {
    let mut passes: Vec<Vec<Draw<'_>>> = Vec::new();
    let mut split = false;
    for event in events {
        if PASS_ENDING_COPIES
            .iter()
            .any(|copy| event.starts_with(copy))
        {
            split = true;
            continue;
        }
        let Some(draw) = Draw::parse(event)? else {
            continue;
        };
        let current = passes.last_mut().filter(|pass| {
            !split
                && pass
                    .last()
                    .is_some_and(|last| last.target == draw.target && last.depth == draw.depth)
        });
        split = false;
        match current {
            Some(pass) => pass.push(draw),
            None => passes.push(vec![draw]),
        }
    }
    Ok(passes
        .iter()
        .filter_map(|draws| GamePass::of(draws))
        .collect())
}

/// What one `draw <n>: ...` line of the dump says about its pass.
struct Draw<'a> {
    target: &'a str,
    depth: &'a str,
    /// `ff`, `none` or the program's id, as the line prints it.
    vs: &'a str,
    /// `ff`, `none` or the program's id, as the line prints it.
    ps: &'a str,
    z_write: u32,
    /// The text between the brackets of `blend=[...]`.
    blend: &'a str,
    blend_enable: u32,
    cull: u32,
    /// Render target 0's colour write mask.
    color_mask: u32,
    /// The text between the brackets of `alpha=[...]`.
    alpha: &'a str,
    alpha_enable: u32,
    /// The text between the brackets of `tex=[...]`: `s<stage>=<id>/...` and `vt<slot>=<id>/...`.
    textures: &'a str,
}

impl<'a> Draw<'a> {
    /// The draw of a `draw <n>: rt=... ds=.../<bits> vs=... ps=... z=[...] ... tex=[...]` event.
    ///
    /// `Ok(None)` for any other event. `draw <n> psc: ...` lines, the shader
    /// constants printed beside a draw that fetches depth, are no draw.
    ///
    /// # Errors
    ///
    /// Returns a message for a draw line that lacks one of the fields read
    /// here or carries one in a form the dump does not print.
    fn parse(event: &'a str) -> Result<Option<Self>, String> {
        let Some((seq, rest)) = event
            .strip_prefix("draw ")
            .and_then(|draw| draw.split_once(": "))
        else {
            return Ok(None);
        };
        if seq.is_empty() || !seq.bytes().all(|b| b.is_ascii_digit()) {
            return Ok(None);
        }
        // The keys carry the space before them so a longer key ending in the
        // same name never matches; the messages print them without it.
        let missing = |what: &str| format!("draw line {event:?}: no {}", what.trim_start());
        let target = between(rest, "rt=", " ds=").ok_or_else(|| missing("rt= ... ds="))?;
        let depth = between(rest, " ds=", " vs=").ok_or_else(|| missing("ds= ... vs="))?;
        let depth = depth.rsplit_once('/').map_or(depth, |(label, _)| label);
        // The other fields come after the shaders; reading them from there
        // keeps a target label from ever being taken for one.
        let state = rest
            .find(" vs=")
            .map(|at| &rest[at..])
            .ok_or_else(|| missing("vs="))?;
        let word = |key: &str| first_word(state, key).ok_or_else(|| missing(key));
        let list = |key: &str| between(state, key, "]").ok_or_else(|| missing(key));
        let number = |list: &str, key: &str, index: usize| {
            list_number(list, index).ok_or_else(|| {
                format!(
                    "draw line {event:?}: {}...] has no number at {index}",
                    key.trim_start()
                )
            })
        };
        let z = list(" z=[")?;
        let blend = list(" blend=[")?;
        let cw = list(" cw=[")?;
        let alpha = list(" alpha=[")?;
        let cull = word(" cull=")?;
        let textures = state
            .rsplit_once(" tex=[")
            .map(|(_, list)| list.trim_end().trim_end_matches(']'))
            .ok_or_else(|| missing(" tex=["))?;
        Ok(Some(Self {
            target,
            depth,
            vs: word(" vs=")?,
            ps: word(" ps=")?,
            z_write: number(z, " z=[", 1)?,
            blend,
            blend_enable: number(blend, " blend=[", 0)?,
            cull: list_number(cull, 0).ok_or_else(|| missing(" cull=<n>"))?,
            color_mask: number(cw, " cw=[", 0)?,
            alpha,
            alpha_enable: number(alpha, " alpha=[", 0)?,
            textures,
        }))
    }

    /// The id of every texture the draw binds, stage by stage, as the line prints it.
    fn texture_ids(&self) -> impl Iterator<Item = &'a str> {
        self.textures
            .split_whitespace()
            .map(|bound| bound.split_once('=').map_or(bound, |(_, texture)| texture))
            .map(|texture| texture.split_once('/').map_or(texture, |(id, _)| id))
    }

    /// The id of the texture on stage 0, `None` when it has none.
    fn stage0_texture(&self) -> Option<&'a str> {
        self.textures
            .split_whitespace()
            .find_map(|bound| bound.strip_prefix("s0="))
            .map(|texture| texture.split_once('/').map_or(texture, |(id, _)| id))
    }
}

/// The text of `text` between the first `start` and the `end` after it.
fn between<'a>(text: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let (_, after) = text.split_once(start)?;
    Some(after.split_once(end)?.0)
}

/// `n` as a count of the dump, which no frame comes near overflowing; `u32::MAX` if one did.
fn narrow(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// How many of `draws` pass `test`.
fn count_draws(draws: &[Draw<'_>], test: impl Fn(&Draw<'_>) -> bool) -> u32 {
    narrow(draws.iter().filter(|d| test(d)).count())
}

/// The first word of `text` after the first `key`.
fn first_word<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.split_once(key)?.1.split_whitespace().next()
}

/// The `index`th number of a dump list such as `1,7,1` or `0x0,0xf`, decimal or `0x` hex.
///
/// Commas and spaces both separate, so `0,5,6,1 sep=0 2,1,1` has `0` at 0.
fn list_number(list: &str, index: usize) -> Option<u32> {
    let value = list.split([',', ' ']).nth(index)?;
    value.strip_prefix("0x").map_or_else(
        || value.parse().ok(),
        |hex| u32::from_str_radix(hex, 16).ok(),
    )
}

/// Write `rows` under `header` as left-aligned columns two spaces apart.
fn write_table(out: &mut String, header: &[String], rows: &[Vec<String>]) {
    let mut widths: Vec<usize> = header.iter().map(String::len).collect();
    for row in rows {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.len());
        }
    }
    for row in std::iter::once(header).chain(rows.iter().map(Vec::as_slice)) {
        let mut line = String::new();
        for (cell, width) in row.iter().zip(&widths) {
            let _ = write!(line, "{cell:<width$}  ");
        }
        let _ = writeln!(out, "{}", line.trim_end());
    }
}

/// The kind of a render target as a draw line names it: `backbuffer`, `texture`, `surface`, `none`.
fn target_kind(target: &str) -> &str {
    target.split_whitespace().next().unwrap_or("-")
}

/// `part` as a percentage of `whole`, 0 for an empty whole.
fn share(part: u32, whole: u32) -> f64 {
    if whole == 0 {
        0.0
    } else {
        f64::from(part) * 100.0 / f64::from(whole)
    }
}

/// The textures a game pass binds per draw.
fn tex_per_draw(pass: &GamePass) -> f64 {
    if pass.draws == 0 {
        0.0
    } else {
        f64::from(pass.textures) / f64::from(pass.draws)
    }
}

/// What is out of tolerance in one pass pair: `pass` when one side lacks it.
///
/// A state value out of tolerance is flagged by its key; a benchmark pass
/// that reports no state mix is judged on the rest alone.
fn flags(game: Option<&GamePass>, bench: Option<&BenchPass>) -> Vec<&'static str> {
    let (Some(game), Some(bench)) = (game, bench) else {
        return vec!["pass"];
    };
    let mut flags = Vec::new();
    if f64::from(game.draws.abs_diff(bench.draws)) > DRAWS_TOLERANCE * f64::from(game.draws) {
        flags.push("draws");
    }
    if (share(game.ff_vs, game.draws) - share(bench.ff_vs, bench.draws)).abs() > FF_TOLERANCE {
        flags.push("ff_vs");
    }
    if (share(game.ff_ps, game.draws) - share(bench.ff_ps, bench.draws)).abs() > FF_TOLERANCE {
        flags.push("ff_ps");
    }
    if (tex_per_draw(game) - bench.tex_per_draw).abs() > TEX_TOLERANCE {
        flags.push("tex");
    }
    let Some(bench_state) = &bench.state else {
        return flags;
    };
    let game_values = game.state.values();
    let bench_values = bench_state.values();
    for (slot, key) in STATE_KEYS.iter().enumerate() {
        let (g, b) = (game_values[slot], bench_values[slot]);
        let off = if slot < STATE_SHARES {
            (share(g, game.draws) - share(b, bench.draws)).abs() > STATE_SHARE_TOLERANCE
        } else {
            f64::from(g.abs_diff(b)) > (STATE_COUNT_TOLERANCE * f64::from(g)).max(STATE_COUNT_FLOOR)
        };
        if off {
            flags.push(*key);
        }
    }
    flags
}

#[cfg(test)]
mod tests;
