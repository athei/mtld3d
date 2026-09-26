//! How busy the machine was before each benchmark process, kept beside its round and flagged.
//!
//! Nothing else may run while `bench-ab` measures, and nothing enforced it:
//! a game played during a run moved the numbers with no trace in the
//! report. So before every round process the runner records the 1-minute
//! load average and measures, over [`INTERVAL`], the CPU `kernel_task` takes
//! (which rises when macOS holds the CPUs back for heat) and the CPU of the
//! processes that are not the run's own, and keeps them in the round's
//! directory, one file per process ([`file_name`]: `machine-<binary>.txt`
//! before a test binary's benchmarks, `machine-host.txt` before the host
//! emitter). The report then warns about each round that started on a busy
//! machine; a warning never changes a verdict, and such rounds are for
//! running again.
//!
//! A file holds a `load1 <x>` line, a `kernel_task <cpu%>` line, a `foreign
//! <cpu%>` line (every foreign process together) and up to [`TOP`] `top
//! <cpu%> <pid> <command line>` lines, CPU in percent of one core. A round
//! is busy when one foreign process held [`HEAVY_CPU`] %, the foreign
//! processes together [`FOREIGN_CPU`] %, or `kernel_task`
//! [`KERNEL_TASK_CPU`] %. The load average is recorded and never warned on:
//! it lags a minute behind, so it still holds the run's own builds and the
//! round before.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

/// How many of the busiest foreign processes a sample keeps.
pub const TOP: usize = 3;

/// How long a sample measures the processes' CPU over: the time between two reads of it.
///
/// Long enough for `ps`'s hundredths of a second to give a share to 2 %,
/// short enough that a sample before each of a run's twenty-odd processes
/// costs seconds, not minutes.
pub const INTERVAL: Duration = Duration::from_millis(500);

/// The CPU share, in percent of one core, from which one foreign process makes a round busy.
///
/// A quarter of a core held while the run is between processes is a game,
/// a build or an indexer, not a background daemon's blip.
pub const HEAVY_CPU: f64 = 25.0;

/// The CPU share, in percent of one core, from which the foreign processes together do.
pub const FOREIGN_CPU: f64 = 50.0;

/// The CPU share, in percent of one core, from which `kernel_task` says the machine is throttled.
///
/// Idle it takes a few percent; macOS raises it to keep the CPUs from
/// running when it holds them back for heat, so half a core is throttling.
pub const KERNEL_TASK_CPU: f64 = 50.0;

/// The least CPU share, in percent of one core, a process counts with.
///
/// Below it a process is neither in the top nor in the sum, which spares
/// asking `lsof` about every idle Wine process whether it is the run's.
const COUNTED_CPU: f64 = 1.0;

/// The processes the run's own work keeps busy outside its process tree.
///
/// Metal's shader compiler service and the window server that composites
/// the benchmarks' windows.
const SIDE_EFFECTS: [&str; 2] = ["MTLCompilerService", "WindowServer"];

/// The commands between the runner and whoever started the run: cargo, make and their shells.
///
/// The ancestors the run leaves out of the foreign processes are the chain
/// of these above the runner, up to the first that is none of them (a
/// terminal's shell, tmux, an IDE, launchd), which is foreign like any
/// other process.
const STARTERS: [&str; 5] = ["cargo", "make", "gmake", "gnumake", "sh"];

/// The name `ps` gives the kernel's own process.
const KERNEL_TASK: &str = "kernel_task";

/// The line of a round's file that sums the foreign processes' CPU.
const FOREIGN: &str = "foreign";

/// The longest command line a sample keeps, in characters.
const COMMAND_CHARS: usize = 200;

/// One process from a sample.
#[derive(Debug, PartialEq)]
pub struct Process {
    pub cpu: f64,
    pub pid: u32,
    /// Its command line, or its executable when the command line could not be read.
    pub command: String,
}

/// What a sample saw: the load average, `kernel_task`'s share and the foreign processes'.
#[derive(Debug, Default, PartialEq)]
pub struct Sample {
    /// The 1-minute load average, `None` when it could not be read.
    pub load1: Option<f64>,
    /// `kernel_task`'s CPU share, in percent of one core, `None` when `ps` did not list it.
    pub kernel_task: Option<f64>,
    /// Every foreign process's CPU share together, in percent of one core.
    pub foreign: f64,
    pub top: Vec<Process>,
}

impl Sample {
    /// The sample as a round's file holds it.
    #[must_use]
    pub fn text(&self) -> String {
        let mut out = String::new();
        if let Some(load) = self.load1 {
            let _ = writeln!(out, "load1 {load:.2}");
        }
        if let Some(cpu) = self.kernel_task {
            let _ = writeln!(out, "{KERNEL_TASK} {cpu:.1}");
        }
        let _ = writeln!(out, "{FOREIGN} {:.1}", self.foreign);
        for process in &self.top {
            let _ = writeln!(
                out,
                "top {:.1} {} {}",
                process.cpu, process.pid, process.command
            );
        }
        out
    }

    /// Read a sample back from the text of a round's file; lines it does not know are skipped.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let value = |line: &str, key: &str| {
            line.strip_prefix(key)?
                .strip_prefix(' ')?
                .trim()
                .parse::<f64>()
                .ok()
        };
        let mut sample = Self::default();
        for line in text.lines() {
            if let Some(load) = value(line, "load1") {
                sample.load1 = Some(load);
            } else if let Some(cpu) = value(line, KERNEL_TASK) {
                sample.kernel_task = Some(cpu);
            } else if let Some(cpu) = value(line, FOREIGN) {
                sample.foreign = cpu;
            } else if let Some(rest) = line.strip_prefix("top ") {
                sample.top.extend(top_line(rest));
            }
        }
        sample
    }

    /// Why the round this sample precedes started on a busy machine; `None` when it did not.
    ///
    /// The load average is not a reason: it lags a minute behind the run's
    /// own work.
    #[must_use]
    pub fn busy(&self) -> Option<String> {
        let mut reasons = Vec::new();
        if let Some(cpu) = self.kernel_task.filter(|cpu| *cpu >= KERNEL_TASK_CPU) {
            reasons.push(format!(
                "{KERNEL_TASK} at {cpu:.0} % of a core (macOS holding the CPUs back for heat)"
            ));
        }
        let heavy: Vec<&Process> = self
            .top
            .iter()
            .filter(|process| process.cpu >= HEAVY_CPU)
            .collect();
        for process in &heavy {
            reasons.push(format!(
                "{} (pid {}) at {:.0} % of a core",
                process.command, process.pid, process.cpu
            ));
        }
        if heavy.is_empty() && self.foreign >= FOREIGN_CPU {
            reasons.push(format!(
                "other processes at {:.0} % of a core together",
                self.foreign
            ));
        }
        (!reasons.is_empty()).then(|| reasons.join("; "))
    }
}

/// What [`classify`] makes of a listing: `kernel_task`'s share and the foreign processes'.
#[derive(Debug, Default, PartialEq)]
pub struct Classified {
    pub kernel_task: Option<f64>,
    /// Every counted foreign process's share together.
    pub foreign: f64,
    pub top: Vec<Process>,
}

/// Sample the machine: its load average, `kernel_task` and the processes that are not the run's.
///
/// `legs` are the Wine installs the run's two legs boot from (the isolated
/// SDK clones); a Wine process that maps its image from one of them is the
/// run's, any other Wine process (a game under another Wine) is foreign.
/// The CPU is measured over [`INTERVAL`] ([`measured_listing`]); when that
/// cannot be read the sample falls back to `ps`'s own `%cpu`, a decaying
/// average of the last minute. A sample that cannot be taken is empty
/// rather than an error: the numbers the run measures do not depend on it.
#[must_use]
pub fn sample(legs: &[PathBuf]) -> Sample {
    let mut averages = [0.0f64; 3];
    // SAFETY: `averages` has room for the three averages asked for, and
    // getloadavg writes at most that many.
    let read = unsafe { libc::getloadavg(averages.as_mut_ptr(), 3) };
    let load1 = (read >= 1).then_some(averages[0]);
    let listing =
        measured_listing().unwrap_or_else(|| ps(&["-A", "-r", "-o", "pcpu=,pid=,ppid=,comm="]));
    let mut classified = classify(&listing, std::process::id(), legs, |pid| {
        image_origin(pid, legs)
    });
    for process in &mut classified.top {
        let line = ps(&["-o", "command=", "-p", &process.pid.to_string()]);
        let line = line.trim();
        if !line.is_empty() {
            process.command = line.chars().take(COMMAND_CHARS).collect();
        }
    }
    Sample {
        load1,
        kernel_task: classified.kernel_task,
        foreign: classified.foreign,
        top: classified.top,
    }
}

/// Sort a `<cpu%> <pid> <ppid> <command>` listing, busiest first, into `kernel_task` and the rest.
///
/// The run's own processes are left out: the process `own`, the chain of
/// [`STARTERS`] above it (the cargo, make and shells that started it) and
/// its children (`ps`), the [`SIDE_EFFECTS`] of its Metal work, a process
/// whose executable lies in one of `legs`, and a Wine process (a Windows
/// path, an `.exe`, or `wine` in its name) whose image `origin` finds in one
/// of them (`Some(true)`): the legs' wineservers and their resident Windows
/// processes. A Wine process `origin` cannot tell (`None`: it exited, or
/// `lsof` could not read it) is dropped rather than called foreign. Every
/// other process at [`COUNTED_CPU`] or more, another Wine's included, is
/// foreign: all of them are summed and the busiest [`TOP`] kept, in the
/// listing's order.
#[must_use]
pub fn classify(
    listing: &str,
    own: u32,
    legs: &[PathBuf],
    mut origin: impl FnMut(u32) -> Option<bool>,
) -> Classified {
    let rows: Vec<Row> = listing.lines().filter_map(ps_row).collect();
    let parents: BTreeMap<u32, (u32, &str)> = rows
        .iter()
        .map(|row| (row.pid, (row.ppid, command_name(&row.command))))
        .collect();
    let mut run = BTreeSet::from([own]);
    let mut at = own;
    while let Some(&(parent, _)) = parents.get(&at) {
        let Some(&(_, name)) = parents.get(&parent) else {
            break;
        };
        if !STARTERS.contains(&name) || !run.insert(parent) {
            break;
        }
        at = parent;
    }
    let mut classified = Classified::default();
    for row in rows {
        let name = command_name(&row.command);
        if name == KERNEL_TASK {
            classified.kernel_task.get_or_insert(row.cpu);
            continue;
        }
        if row.cpu < COUNTED_CPU
            || run.contains(&row.pid)
            || row.ppid == own
            || SIDE_EFFECTS.contains(&name)
            || legs
                .iter()
                .any(|leg| Path::new(&row.command).starts_with(leg))
            || (wine_like(&row.command) && origin(row.pid) != Some(false))
        {
            continue;
        }
        classified.foreign += row.cpu;
        if classified.top.len() < TOP {
            classified.top.push(Process {
                cpu: row.cpu,
                pid: row.pid,
                command: row.command,
            });
        }
    }
    classified
}

/// The name of the file a round keeps the sample taken before the process of `what` in.
#[must_use]
pub fn file_name(what: &str) -> String {
    format!("machine-{what}.txt")
}

/// Write `sample` into the round directory `dir`, as the file of `what` ([`file_name`]).
///
/// # Errors
///
/// Returns a message when the file cannot be written.
pub fn keep(dir: &Path, what: &str, sample: &Sample) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(file_name(what));
    fs::write(&path, sample.text()).map_err(|e| format!("{}: {e}", path.display()))
}

/// The warnings of an A/B directory: one per round and process that started on a busy machine.
#[must_use]
pub fn warnings(dir: &Path, legs: &[&str], rounds: usize) -> Vec<String> {
    let mut out = Vec::new();
    for round in 0..rounds {
        for leg in legs {
            let round_dir = dir.join(leg).join(round.to_string());
            let Ok(entries) = fs::read_dir(&round_dir) else {
                continue;
            };
            let mut files: Vec<(String, PathBuf)> = entries
                .flatten()
                .filter_map(|entry| {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    let what = name.strip_prefix("machine-")?.strip_suffix(".txt")?;
                    Some((what.to_owned(), entry.path()))
                })
                .collect();
            files.sort();
            for (what, path) in files {
                let Ok(text) = fs::read_to_string(&path) else {
                    continue;
                };
                if let Some(reason) = Sample::parse(&text).busy() {
                    out.push(format!(
                        "{leg} round {} ({what}) started on a busy machine: {reason}; run it \
                         again",
                        round + 1
                    ));
                }
            }
        }
    }
    out
}

/// One process of a listing, before it is classified.
struct Row {
    cpu: f64,
    pid: u32,
    ppid: u32,
    command: String,
}

/// One `<cpu> <pid> <ppid> <command>` line of a listing, the command running to the end.
fn ps_row(line: &str) -> Option<Row> {
    let (cpu, rest) = line.trim_start().split_once(char::is_whitespace)?;
    let (pid, rest) = rest.trim_start().split_once(char::is_whitespace)?;
    let (parent, command) = rest.trim_start().split_once(char::is_whitespace)?;
    let command = command.trim();
    Some(Row {
        cpu: cpu.parse().ok().filter(|cpu: &f64| cpu.is_finite())?,
        pid: pid.parse().ok()?,
        ppid: parent.parse().ok()?,
        command: (!command.is_empty()).then(|| command.to_owned())?,
    })
}

/// One `<cpu> <pid> <command>` line of a round's file.
fn top_line(line: &str) -> Option<Process> {
    let (cpu, rest) = line.trim_start().split_once(char::is_whitespace)?;
    let (pid, command) = rest.trim_start().split_once(char::is_whitespace)?;
    let command = command.trim();
    Some(Process {
        cpu: cpu.parse().ok().filter(|cpu: &f64| cpu.is_finite())?,
        pid: pid.parse().ok()?,
        command: (!command.is_empty()).then(|| command.to_owned())?,
    })
}

/// Whether a process's executable looks like Wine's: a Windows path, an `.exe`, or `wine`.
fn wine_like(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    command.as_bytes().get(1..3) == Some(b":\\".as_slice())
        || lower.contains("wine")
        || Path::new(&lower)
            .extension()
            .is_some_and(|extension| extension == "exe")
}

/// The executable's name in a command: what follows its last `/` or `\`.
fn command_name(command: &str) -> &str {
    command.rsplit(['/', '\\']).next().unwrap_or(command)
}

/// Whether the process `pid` maps its images from one of the Wine installs `legs`.
///
/// `None` when `lsof` names no image for it: the process has exited, it
/// cannot be read, or `lsof` itself failed. `-b` keeps `lsof` from calls
/// that can block in the kernel.
fn image_origin(pid: u32, legs: &[PathBuf]) -> Option<bool> {
    let listing = Command::new("lsof")
        .args([
            "-b",
            "-n",
            "-P",
            "-w",
            "-a",
            "-p",
            &pid.to_string(),
            "-d",
            "txt",
            "-F",
            "n",
        ])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default();
    let images: Vec<&str> = listing
        .lines()
        .filter_map(|line| line.strip_prefix('n'))
        .collect();
    (!images.is_empty()).then(|| {
        images
            .iter()
            .any(|path| legs.iter().any(|leg| Path::new(path).starts_with(leg)))
    })
}

/// Every process's CPU over about [`INTERVAL`], busiest first, as a listing [`classify`] reads.
///
/// Two reads of the CPU time each process has used rather than `ps`'s own
/// `%cpu`, a decaying average of the last minute that still holds a load
/// that has gone ([`shares`]); the share divides by the time measured
/// between the reads returning. `None` when either read gives nothing, so
/// the caller falls back to `%cpu` instead of reading a quiet machine.
fn measured_listing() -> Option<String> {
    let before: BTreeMap<u32, f64> = ps(&["-A", "-o", "pid=,cputime="])
        .lines()
        .filter_map(|line| {
            let (pid, time) = line.trim_start().split_once(char::is_whitespace)?;
            Some((pid.parse().ok()?, cpu_seconds(time.trim())?))
        })
        .collect();
    let first = Instant::now();
    if before.is_empty() {
        return None;
    }
    thread::sleep(INTERVAL);
    let after = cputime_rows(&ps(&["-A", "-o", "pid=,ppid=,cputime=,comm="]));
    let elapsed = first.elapsed();
    if after.is_empty() {
        return None;
    }
    Some(shares(&before, &after, elapsed))
}

/// One process of the second CPU-time read: its pid, its parent's, its CPU seconds, its command.
pub struct CpuTime {
    pub pid: u32,
    pub ppid: u32,
    pub seconds: f64,
    pub command: String,
}

/// The `<pid> <ppid> <cputime> <command>` lines of a `ps` listing.
#[must_use]
pub fn cputime_rows(listing: &str) -> Vec<CpuTime> {
    listing
        .lines()
        .filter_map(|line| {
            let (pid, rest) = line.trim_start().split_once(char::is_whitespace)?;
            let (parent, rest) = rest.trim_start().split_once(char::is_whitespace)?;
            let (time, command) = rest.trim_start().split_once(char::is_whitespace)?;
            Some(CpuTime {
                pid: pid.parse().ok()?,
                ppid: parent.parse().ok()?,
                seconds: cpu_seconds(time)?,
                command: command.trim().to_owned(),
            })
        })
        .collect()
}

/// The listing [`classify`] reads, from CPU seconds `before` and `after` `elapsed` apart.
///
/// Each process's share is the CPU time it used between the reads over
/// `elapsed`, in percent of one core, busiest first. A process the first
/// read did not see (it started between them) counts all its time; one
/// whose time went down (its pid was reused by a process younger than the
/// first read) counts none; one only the first read saw has exited and is
/// not listed.
#[must_use]
pub fn shares(before: &BTreeMap<u32, f64>, after: &[CpuTime], elapsed: Duration) -> String {
    let secs = elapsed.as_secs_f64().max(f64::EPSILON);
    let mut rows: Vec<(f64, String)> = after
        .iter()
        .map(|process| {
            let used = process.seconds - before.get(&process.pid).copied().unwrap_or(0.0);
            let cpu = used.max(0.0) / secs * 100.0;
            (
                cpu,
                format!(
                    "{cpu:.1} {} {} {}",
                    process.pid, process.ppid, process.command
                ),
            )
        })
        .collect();
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    rows.into_iter().fold(String::new(), |mut out, (_, line)| {
        let _ = writeln!(out, "{line}");
        out
    })
}

/// The seconds of a `ps` CPU time, `[[dd-]hh:]mm:ss.ss`.
fn cpu_seconds(time: &str) -> Option<f64> {
    let (days, clock) = time.split_once('-').unwrap_or(("0", time));
    let mut seconds = days.parse::<f64>().ok()? * 86_400.0;
    let mut scale = 1.0;
    for part in clock.rsplit(':') {
        seconds = part.parse::<f64>().ok()?.mul_add(scale, seconds);
        scale *= 60.0;
    }
    Some(seconds)
}

/// What `ps` prints with `args`, empty when it cannot run.
fn ps(args: &[&str]) -> String {
    Command::new("ps")
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests;
