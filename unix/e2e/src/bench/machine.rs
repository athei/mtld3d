//! How busy the machine was before each benchmark process, kept beside its round and flagged.
//!
//! Nothing else may run while `bench-ab` measures, and nothing enforced it:
//! a game played during a run moved the numbers with no trace in the
//! report. So before every round process the runner samples the 1-minute
//! load average and the processes using the most CPU that are not the
//! run's own, and keeps them in the round's directory ([`MACHINE_FILES`]:
//! `machine.txt` before the benchmarks, `machine-host.txt` before the host
//! emitter). The report
//! then warns about each round that started on a busy machine; a warning
//! never changes a verdict, and such rounds are for running again.
//!
//! The file is one `load1 <x>` line and up to [`TOP`] `top <cpu%> <pid>
//! <command>` lines. A round is busy when the load was over
//! [`LOAD_THRESHOLD`] or a foreign process used at least [`HEAVY_CPU`] % of
//! a core.

use std::{
    fmt::Write as _,
    fs,
    path::Path,
    process::{Command, Stdio},
};

/// The files in a round directory that keep the machine's state before each of its processes.
///
/// One before the end-to-end benchmarks' process of the leg's round, one
/// before the host emitter's, each with how the warnings name it.
pub const MACHINE_FILES: [(&str, &str); 2] = [
    ("machine.txt", "the benchmarks"),
    ("machine-host.txt", "the host emitter"),
];

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

/// Command names that belong to the run: the runner, what started it, and `ps` itself.
const OWN_COMMANDS: [&str; 5] = ["mtld3d-e2e", "cargo", "make", "ps", "sh"];

/// One process from a `ps` sample.
#[derive(Debug, PartialEq)]
pub struct Process {
    pub cpu: f64,
    pub pid: u32,
    pub command: String,
}

/// What a sample saw: the load average and the busiest foreign processes.
#[derive(Debug, Default, PartialEq)]
pub struct Sample {
    /// The 1-minute load average, `None` when it could not be read.
    pub load1: Option<f64>,
    pub top: Vec<Process>,
}

impl Sample {
    /// The sample as `machine.txt` holds it.
    #[must_use]
    pub fn text(&self) -> String {
        let mut out = String::new();
        if let Some(load) = self.load1 {
            let _ = writeln!(out, "load1 {load:.2}");
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

    /// Read a sample back from the text of `machine.txt`; lines it does not know are skipped.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let mut sample = Self::default();
        for line in text.lines() {
            if let Some(load) = line.strip_prefix("load1 ") {
                sample.load1 = load.trim().parse().ok();
            } else if let Some(rest) = line.strip_prefix("top ") {
                sample.top.extend(ps_line(rest));
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
        for process in self.top.iter().filter(|process| process.cpu >= HEAVY_CPU) {
            reasons.push(format!(
                "{} (pid {}) at {:.0} % of a core",
                process.command, process.pid, process.cpu
            ));
        }
        (!reasons.is_empty()).then(|| reasons.join("; "))
    }
}

/// Sample the machine now: its load average and the busiest processes that are not the run's.
///
/// A sample that cannot be taken is empty rather than an error: the numbers
/// the run measures do not depend on it.
#[must_use]
pub fn sample() -> Sample {
    let mut averages = [0.0f64; 3];
    // SAFETY: `averages` has room for the three averages asked for, and
    // getloadavg writes at most that many.
    let read = unsafe { libc::getloadavg(averages.as_mut_ptr(), 3) };
    let load1 = (read >= 1).then_some(averages[0]);
    let own = std::process::id();
    let listing = Command::new("ps")
        .args(["-A", "-r", "-o", "pcpu=,pid=,comm="])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default();
    Sample {
        load1,
        top: foreign_top(&listing, own),
    }
}

/// The [`TOP`] busiest processes of a `ps -r -o pcpu=,pid=,comm=` listing that are not the run's.
///
/// The run's own are the process `own`, the commands in [`OWN_COMMANDS`],
/// and Wine's: a path naming `wine` (the loader, the server, the preloader)
/// or a Windows path (`C:\windows\system32\winedevice.exe`, the residents
/// of the legs' sessions). `ps -r` lists by CPU already; the order is kept.
#[must_use]
pub fn foreign_top(listing: &str, own: u32) -> Vec<Process> {
    listing
        .lines()
        .filter_map(ps_line)
        .filter(|process| process.pid != own && !run_command(&process.command))
        .take(TOP)
        .collect()
}

/// Write `sample` into the round directory `dir`, as `file`, one of [`MACHINE_FILES`].
///
/// # Errors
///
/// Returns a message when the file cannot be written.
pub fn keep(dir: &Path, file: &str, sample: &Sample) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(file);
    fs::write(&path, sample.text()).map_err(|e| format!("{}: {e}", path.display()))
}

/// The warnings of an A/B directory: one per leg and round whose sample says the machine was busy.
#[must_use]
pub fn warnings(dir: &Path, legs: &[&str], rounds: usize) -> Vec<String> {
    let mut out = Vec::new();
    for round in 0..rounds {
        for leg in legs {
            for (file, what) in MACHINE_FILES {
                let path = dir.join(leg).join(round.to_string()).join(file);
                let Ok(text) = fs::read_to_string(&path) else {
                    continue;
                };
                if let Some(reason) = Sample::parse(&text).busy() {
                    out.push(format!(
                        "{leg} round {} of {what} started on a busy machine: {reason}; run it \
                         again",
                        round + 1
                    ));
                }
            }
        }
    }
    out
}

/// One `<cpu> <pid> <command>` line, the command running to the end; `None` for anything else.
fn ps_line(line: &str) -> Option<Process> {
    let (cpu, rest) = line.trim_start().split_once(char::is_whitespace)?;
    let (pid, command) = rest.trim_start().split_once(char::is_whitespace)?;
    let command = command.trim();
    (!command.is_empty())
        .then(|| Process {
            cpu: cpu.parse().unwrap_or(f64::NAN),
            pid: pid.parse().unwrap_or(0),
            command: command.to_owned(),
        })
        .filter(|process| process.cpu.is_finite() && process.pid != 0)
}

/// Whether `command` is one of the run's own processes.
fn run_command(command: &str) -> bool {
    let name = command.rsplit(['/', '\\']).next().unwrap_or(command);
    let windows_path = command.as_bytes().get(1..3) == Some(b":\\".as_slice());
    OWN_COMMANDS.contains(&name) || command.to_ascii_lowercase().contains("wine") || windows_path
}

#[cfg(test)]
mod tests;
