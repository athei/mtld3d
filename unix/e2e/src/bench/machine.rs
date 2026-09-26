//! How busy the machine was before each benchmark process, kept beside its round and flagged.
//!
//! Nothing else may run while `bench-ab` measures, and nothing enforced it:
//! a game played during a run moved the numbers with no trace in the
//! report. So before every round process the runner samples the 1-minute
//! load average, `kernel_task`'s CPU share (which rises when macOS holds the
//! CPUs back for heat) and the processes using the most CPU that are not
//! the run's own, and keeps them in the round's directory, one file per
//! process ([`file_name`]: `machine-<binary>.txt` before a test binary's
//! benchmarks, `machine-host.txt` before the host emitter). The report then
//! warns about each round that started on a busy machine; a warning never
//! changes a verdict, and such rounds are for running again.
//!
//! A file is a `load1 <x>` line, a `kernel_task <cpu%>` line and up to
//! [`TOP`] `top <cpu%> <pid> <command line>` lines. A round is busy when the
//! load was over [`LOAD_THRESHOLD`], `kernel_task` held [`KERNEL_TASK_CPU`]
//! % of a core, or a foreign process held [`HEAVY_CPU`] %.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

/// How many of the busiest foreign processes a sample keeps.
pub const TOP: usize = 3;

/// The 1-minute load average above which a round started on a busy machine.
///
/// The run itself keeps the load near two or three: a benchmark keeps its
/// API, encoder and submit threads busy for most of every round, and the
/// average still holds the round before when the next one starts. Anything
/// past four is something else.
pub const LOAD_THRESHOLD: f64 = 4.0;

/// The CPU share, in percent of one core, from which a foreign process counts as heavy.
///
/// `ps` reports a decaying average; a quarter of a core held over the last
/// minute is a game, a build or an indexer, not a background daemon's blip.
pub const HEAVY_CPU: f64 = 25.0;

/// The CPU share, in percent of one core, from which `kernel_task` says the machine is throttled.
///
/// Idle it takes a few percent; macOS raises it to keep the CPUs from
/// running when it holds them back for heat, so half a core is throttling.
pub const KERNEL_TASK_CPU: f64 = 50.0;

/// The processes the run's own work starts outside its process tree: Metal's shader compiler
/// service and the window server that composites its windows.
const SIDE_EFFECTS: [&str; 2] = ["MTLCompilerService", "WindowServer"];

/// The name `ps` gives the kernel's own process.
const KERNEL_TASK: &str = "kernel_task";

/// The longest command line a sample keeps, in characters.
const COMMAND_CHARS: usize = 200;

/// One process from a `ps` sample.
#[derive(Debug, PartialEq)]
pub struct Process {
    pub cpu: f64,
    pub pid: u32,
    /// Its command line, or its executable when the command line could not be read.
    pub command: String,
}

/// What a sample saw: the load average, `kernel_task`'s share and the busiest foreign processes.
#[derive(Debug, Default, PartialEq)]
pub struct Sample {
    /// The 1-minute load average, `None` when it could not be read.
    pub load1: Option<f64>,
    /// `kernel_task`'s CPU share, in percent of one core, `None` when `ps` did not list it.
    pub kernel_task: Option<f64>,
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
        let mut sample = Self::default();
        for line in text.lines() {
            if let Some(load) = line.strip_prefix("load1 ") {
                sample.load1 = load.trim().parse().ok();
            } else if let Some(cpu) = line
                .strip_prefix(KERNEL_TASK)
                .and_then(|rest| rest.strip_prefix(' '))
            {
                sample.kernel_task = cpu.trim().parse().ok();
            } else if let Some(rest) = line.strip_prefix("top ") {
                sample.top.extend(top_line(rest));
            }
        }
        sample
    }

    /// Why the round this sample precedes started on a busy machine; `None` when it did not.
    #[must_use]
    pub fn busy(&self) -> Option<String> {
        let mut reasons = Vec::new();
        if let Some(load) = self.load1.filter(|load| *load > LOAD_THRESHOLD) {
            reasons.push(format!(
                "1-minute load {load:.2} (over {LOAD_THRESHOLD:.1})"
            ));
        }
        if let Some(cpu) = self.kernel_task.filter(|cpu| *cpu >= KERNEL_TASK_CPU) {
            reasons.push(format!(
                "{KERNEL_TASK} at {cpu:.0} % of a core (macOS holding the CPUs back for heat)"
            ));
        }
        for process in self.top.iter().filter(|process| process.cpu >= HEAVY_CPU) {
            reasons.push(format!(
                "{} (pid {}) at {:.0} % of a core",
                process.command, process.pid, process.cpu
            ));
        }
        (!reasons.is_empty()).then(|| reasons.join("; "))
    }
}

/// What [`classify`] makes of a `ps` listing: `kernel_task`'s share and the foreign top.
#[derive(Debug, Default, PartialEq)]
pub struct Classified {
    pub kernel_task: Option<f64>,
    pub top: Vec<Process>,
}

/// Sample the machine: its load average, `kernel_task` and the busiest processes not the run's.
///
/// `legs` are the Wine installs the run's two legs boot from (the isolated
/// SDK clones); a Wine process that maps its image from one of them is the
/// run's, any other Wine process (a game under another Wine) is foreign. A
/// sample that cannot be taken is empty rather than an error: the numbers
/// the run measures do not depend on it.
#[must_use]
pub fn sample(legs: &[PathBuf]) -> Sample {
    let mut averages = [0.0f64; 3];
    // SAFETY: `averages` has room for the three averages asked for, and
    // getloadavg writes at most that many.
    let read = unsafe { libc::getloadavg(averages.as_mut_ptr(), 3) };
    let load1 = (read >= 1).then_some(averages[0]);
    let listing = ps(&["-A", "-r", "-o", "pcpu=,pid=,ppid=,comm="]);
    let mut classified = classify(&listing, std::process::id(), legs, |pid| {
        maps_a_leg(pid, legs)
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
        top: classified.top,
    }
}

/// Sort a `ps -r -o pcpu=,pid=,ppid=,comm=` listing into `kernel_task` and the foreign top.
///
/// The run's own processes are left out: the process `own` and its
/// ancestors (the cargo and make that started it) and children (`ps`), the
/// [`SIDE_EFFECTS`] of its Metal work, a process whose executable lies in one
/// of `legs`, and a Wine process (a Windows path, an `.exe`, or `wine` in
/// its name) that `maps_leg` says maps its image from one of them: the legs'
/// wineservers and their resident Windows processes. Every other process,
/// another Wine's included, is foreign. `ps -r` lists by CPU already; the
/// order is kept.
#[must_use]
pub fn classify(
    listing: &str,
    own: u32,
    legs: &[PathBuf],
    mut maps_leg: impl FnMut(u32) -> bool,
) -> Classified {
    let rows: Vec<Row> = listing.lines().filter_map(ps_row).collect();
    let parents: BTreeMap<u32, u32> = rows.iter().map(|row| (row.pid, row.ppid)).collect();
    let mut run = BTreeSet::from([own]);
    let mut at = own;
    while let Some(&parent) = parents.get(&at) {
        if parent == 0 || !run.insert(parent) {
            break;
        }
        at = parent;
    }
    let mut classified = Classified::default();
    for row in rows {
        let name = row
            .command
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(&row.command);
        if name == KERNEL_TASK {
            classified.kernel_task.get_or_insert(row.cpu);
            continue;
        }
        if classified.top.len() == TOP
            || run.contains(&row.pid)
            || row.ppid == own
            || SIDE_EFFECTS.contains(&name)
            || legs
                .iter()
                .any(|leg| Path::new(&row.command).starts_with(leg))
            || (wine_like(&row.command) && maps_leg(row.pid))
        {
            continue;
        }
        classified.top.push(Process {
            cpu: row.cpu,
            pid: row.pid,
            command: row.command,
        });
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

/// One process of a `ps` listing, before it is classified.
struct Row {
    cpu: f64,
    pid: u32,
    ppid: u32,
    command: String,
}

/// One `<cpu> <pid> <ppid> <command>` line of `ps`, the command running to the end.
fn ps_row(line: &str) -> Option<Row> {
    let (cpu, rest) = line.trim_start().split_once(char::is_whitespace)?;
    let (pid, rest) = rest.trim_start().split_once(char::is_whitespace)?;
    let (ppid, command) = rest.trim_start().split_once(char::is_whitespace)?;
    let command = command.trim();
    Some(Row {
        cpu: cpu.parse().ok().filter(|cpu: &f64| cpu.is_finite())?,
        pid: pid.parse().ok()?,
        ppid: ppid.parse().ok()?,
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

/// Whether the process `pid` maps an image from one of the Wine installs `legs`.
fn maps_a_leg(pid: u32, legs: &[PathBuf]) -> bool {
    let listing = Command::new("lsof")
        .args([
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
    listing
        .lines()
        .filter_map(|line| line.strip_prefix('n'))
        .any(|path| legs.iter().any(|leg| Path::new(path).starts_with(leg)))
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
