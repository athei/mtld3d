//! Check a benchmark's declared frame against a frame a game dumped, to calibrate the scene.
//!
//! F12 in a game makes the layer log a few consecutive frames draw by draw
//! (`[dump]` lines at info level: the frame's start and end, every bind,
//! clear and copy, and one line per draw naming its render target, depth
//! surface, shaders and textures). A benchmark that stands for that game
//! writes the frame it builds as `shape` lines in its metrics file, one per
//! pass. `bench-shape` reads the last complete dumped frame, splits it into
//! passes along the main lines of the layer's own splits, and prints the two
//! side by side.
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
//! are not judged. Exit code 0 when everything is within tolerance, 1
//! otherwise.

use std::{fmt::Write as _, fs, path::Path, process::ExitCode};

use super::{metrics, shape::log_message};

/// How far a pass's draw count may be off, as a fraction of the game's.
const DRAWS_TOLERANCE: f64 = 0.10;

/// How far a pass's fixed-function share may be off, in percentage points.
const FF_TOLERANCE: f64 = 10.0;

/// How far a pass's textures per draw may be off.
const TEX_TOLERANCE: f64 = 1.0;

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
}

/// A benchmark's declared frame.
#[derive(Debug)]
pub struct BenchFrame {
    pub passes: Vec<BenchPass>,
    pub backbuffer: Size,
    /// Where the back-buffer size came from: its meta line, or the last pass.
    pub backbuffer_from: &'static str,
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
/// Returns a message when no frame is complete, the frame's draw count
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
            "no complete [dump] frame (a frame start followed by its frame end); press F12 in \
             the game to dump one"
                .to_owned(),
        );
    };
    let declared = end
        .split_whitespace()
        .next()
        .and_then(|draws| draws.parse::<u32>().ok())
        .ok_or_else(|| format!("frame end line {end:?} names no draw count"))?;
    let passes = game_passes(&body);
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
/// Each line is `pass <i> <W>x<H> draws=<n> ff_vs=<n> ff_ps=<n> tex_per_draw=<x>`.
/// The back buffer is `backbuffer`, a `meta <bench> backbuffer <W>x<H>`
/// value, when the file has one, and the last pass's target otherwise:
/// a frame ends on the back buffer it presents.
///
/// # Errors
///
/// Returns a message for a line that is not that shape, a key missing or
/// unknown, passes out of order, or no pass at all.
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
        let mut values = [None::<&str>; 4];
        let keys = ["draws", "ff_vs", "ff_ps", "tex_per_draw"];
        for word in rest {
            let (key, value) = word
                .split_once('=')
                .ok_or_else(|| format!("shape line {line:?}: {word:?} is not key=value"))?;
            let slot = keys
                .iter()
                .position(|known| *known == key)
                .ok_or_else(|| format!("shape line {line:?}: unknown key {key:?}"))?;
            values[slot] = Some(value);
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
        passes.push(BenchPass {
            size,
            draws: count(0)?,
            ff_vs: count(1)?,
            ff_ps: count(2)?,
            tex_per_draw,
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
        "tolerance: draws {:.0} %, ff share {FF_TOLERANCE:.0} points, tex/draw {TEX_TOLERANCE:.1}; \
         sizes are ratios to the back buffer and not judged",
        DRAWS_TOLERANCE * 100.0
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
    ];
    let mut cells: Vec<[String; 9]> = Vec::new();
    let mut flagged = 0;
    let pairs = pair_passes(game, bench);
    for &(game_index, bench_index) in &pairs {
        let game_pass = game_index.map(|index| &game.passes[index]);
        let bench_pass = bench_index.map(|index| &bench.passes[index]);
        let flags = flags(game_pass, bench_pass);
        if !flags.is_empty() {
            flagged += 1;
        }
        let pair = |g: Option<String>, b: Option<String>| {
            format!(
                "{} / {}",
                g.unwrap_or_else(|| "-".to_owned()),
                b.unwrap_or_else(|| "-".to_owned())
            )
        };
        let shown = |index: Option<usize>| index.map_or_else(|| "-".to_owned(), |i| i.to_string());
        cells.push([
            format!("{}/{}", shown(game_index), shown(bench_index)),
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
    let mut widths = header.map(str::len);
    for row in &cells {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.len());
        }
    }
    let rows = std::iter::once(header.map(str::to_owned)).chain(cells);
    for row in rows {
        let mut line = String::new();
        for (cell, width) in row.iter().zip(widths) {
            let _ = write!(line, "{cell:<width$}  ");
        }
        let _ = writeln!(out, "{}", line.trim_end());
    }
    let count_matches = game.passes.len() == bench.passes.len();
    let within = count_matches && flagged == 0;
    let _ = writeln!(
        out,
        "bench-shape: {}: {} game passes, {} bench passes; {flagged} of {} rows flagged",
        if within {
            "WITHIN TOLERANCE"
        } else {
            "OUT OF TOLERANCE"
        },
        game.passes.len(),
        bench.passes.len(),
        pairs.len()
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
fn game_passes(events: &[&str]) -> Vec<GamePass> {
    let mut passes: Vec<GamePass> = Vec::new();
    let mut split = false;
    for event in events {
        if PASS_ENDING_COPIES
            .iter()
            .any(|copy| event.starts_with(copy))
        {
            split = true;
            continue;
        }
        let Some(draw) = Draw::parse(event) else {
            continue;
        };
        let continues = !split
            && passes
                .last()
                .is_some_and(|pass| pass.target == draw.target && pass.depth == draw.depth);
        split = false;
        if !continues {
            passes.push(GamePass {
                target: draw.target.to_owned(),
                depth: draw.depth.to_owned(),
                size: Size::find(draw.target),
                draws: 0,
                ff_vs: 0,
                ff_ps: 0,
                textures: 0,
            });
        }
        if let Some(pass) = passes.last_mut() {
            pass.draws += 1;
            pass.ff_vs += u32::from(draw.ff_vs);
            pass.ff_ps += u32::from(draw.ff_ps);
            pass.textures += draw.textures;
        }
    }
    passes
}

/// What one `draw <n>: ...` line of the dump says about its pass.
struct Draw<'a> {
    target: &'a str,
    depth: &'a str,
    ff_vs: bool,
    ff_ps: bool,
    textures: u32,
}

impl<'a> Draw<'a> {
    /// The draw of a `draw <n>: rt=... ds=.../<bits> vs=... ps=... ... tex=[...]` event.
    ///
    /// `draw <n> psc: ...` lines, the shader constants printed beside a draw
    /// that fetches depth, are no draw.
    fn parse(event: &'a str) -> Option<Self> {
        let (seq, rest) = event.strip_prefix("draw ")?.split_once(": ")?;
        if seq.is_empty() || !seq.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let target = between(rest, "rt=", " ds=")?;
        let depth = between(rest, " ds=", " vs=")?;
        let depth = depth.rsplit_once('/').map_or(depth, |(label, _)| label);
        let stage = |key: &str| {
            let (_, after) = rest.split_once(key)?;
            after.split_whitespace().next()
        };
        let textures = rest.rsplit_once("tex=[").map_or(0, |(_, list)| {
            list.trim_end()
                .trim_end_matches(']')
                .split_whitespace()
                .count()
        });
        Some(Self {
            target,
            depth,
            ff_vs: stage(" vs=")? == "ff",
            ff_ps: stage(" ps=")? == "ff",
            textures: u32::try_from(textures).unwrap_or(u32::MAX),
        })
    }
}

/// The text of `text` between the first `start` and the `end` after it.
fn between<'a>(text: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let (_, after) = text.split_once(start)?;
    Some(after.split_once(end)?.0)
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
    flags
}

#[cfg(test)]
mod tests;
