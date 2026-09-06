//! A test binary on disk: its name, the launcher that runs it, and the accounts of its processes.
//!
//! Two accounts survive a process the runner lost: the stderr it kept in a
//! file of its own, and the log the layer wrote, which is where the layer's
//! crash report goes and the only place it goes.

use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

use crate::{
    attribute::{Launcher, ProcessEnd},
    libtest::{self, Event, Parser},
    run::{self, ExitKind},
};

/// How many kept stderr files a directory holds, as many as the layer keeps logs.
const KEEP: usize = 10;

/// How many of the process's own stderr lines a report shows.
const TAIL_LINES: usize = 15;

/// How many layer-log lines a report shows once a fatal line anchors them.
///
/// The crash report is a banner, the faulting thread, the program counter,
/// the registers and a symbolised stack: enough lines that the plain tail
/// would cut the banner off, and few enough to quote whole.
const FATAL_LINES: usize = 40;

/// The classes Wine's debug channels print, the first field of every line they emit.
const WINE_CLASSES: [&str; 4] = ["err", "warn", "fixme", "trace"];

/// The name a test binary is reported under: its file stem without cargo's `-<hash>`.
#[must_use]
pub fn binary_name(exe: &Path) -> String {
    let stem = exe
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    match stem.rsplit_once('-') {
        Some((name, hash)) if !hash.is_empty() && hash.chars().all(|c| c.is_ascii_hexdigit()) => {
            name.to_owned()
        }
        _ => stem.to_owned(),
    }
}

/// Runs one test binary's processes under Wine.
pub struct WineLauncher {
    wine: PathBuf,
    exe: PathBuf,
    /// Where a dead process's whole stderr goes, beside the layer's own logs.
    log_dir: PathBuf,
    timeout: Duration,
    /// A line as the process printed it, for the progress a caller shows.
    on_line: Box<dyn FnMut(&str)>,
}

impl WineLauncher {
    /// A launcher for `exe`, keeping dead processes' stderr in `log_dir`.
    ///
    /// `log_dir` is where the layer writes its own per-process logs, so the
    /// two accounts of one process sit together; `None` means the layer's
    /// default, `mtld3d-logs` beside the executable.
    pub fn new(
        wine: &Path,
        exe: &Path,
        log_dir: Option<&Path>,
        timeout: Duration,
        on_line: Box<dyn FnMut(&str)>,
    ) -> Self {
        let log_dir = log_dir.map_or_else(
            || {
                exe.parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join("mtld3d-logs")
            },
            Path::to_path_buf,
        );
        Self {
            wine: wine.to_path_buf(),
            exe: exe.to_path_buf(),
            log_dir,
            timeout,
            on_line,
        }
    }
}

impl Launcher for WineLauncher {
    fn run(
        &mut self,
        names: Option<&[String]>,
        threads: u32,
        on_event: &mut dyn FnMut(Event),
    ) -> Result<ProcessEnd, String> {
        let mut args = vec![
            format!("--test-threads={threads}"),
            "--nocapture".to_owned(),
        ];
        if let Some(names) = names {
            args.push("--exact".to_owned());
            args.extend(names.iter().cloned());
        }
        let mut parser = Parser::default();
        let mut stdout = String::new();
        let exit = run::run(&self.wine, &self.exe, &args, self.timeout, &mut |line| {
            (self.on_line)(line);
            stdout.push_str(line);
            stdout.push('\n');
            if let Some(event) = parser.line(line) {
                on_event(event);
            }
        })?;
        Ok(ProcessEnd {
            pid: exit.pid,
            kind: exit.kind,
            stdout,
            stderr: exit.stderr,
        })
    }

    fn list(&mut self) -> Result<Vec<String>, String> {
        let mut stdout = String::new();
        let exit = run::run(
            &self.wine,
            &self.exe,
            &["--list".to_owned()],
            self.timeout,
            &mut |line| {
                stdout.push_str(line);
                stdout.push('\n');
            },
        )?;
        if exit.kind != ExitKind::Code(0) {
            return Err(format!(
                "{} --list ended with {}:\n{}",
                self.exe.display(),
                exit.kind.describe(),
                stderr_tail(&exit.stderr)
            ));
        }
        let names = libtest::listed_tests(&stdout);
        if names.is_empty() {
            return Err(format!("{} --list named no test", self.exe.display()));
        }
        Ok(names)
    }

    fn keep_stderr(&self, pid: u32, stderr: &str) -> Result<PathBuf, String> {
        keep_stderr(&self.log_dir, &binary_name(&self.exe), pid, stderr)
    }

    fn layer_log(&self, pid: u32) -> Result<(PathBuf, String), String> {
        // The layer names the file after the executable's whole stem, cargo
        // hash and all, and after the host pid, which is the one the runner
        // spawned the process under.
        let stem = self
            .exe
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default();
        let path = self
            .log_dir
            .join(mtld3d_shared::log_paths::log_file_name(stem, pid));
        let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok((path, layer_tail(&text)))
    }
}

/// Write one dead process's whole stderr into `dir`, and name the file.
///
/// The file is `<binary>-<pid>.stderr` beside the layer's per-process logs,
/// under the pid the runner spawned. The extension is not `log`, so the
/// layer's own retention (which keeps the newest ten of those) never
/// removes an account of how a process died; the newest ten of these stay in
/// the directory instead.
///
/// # Errors
///
/// Returns the reason when the directory or the file cannot be written.
pub fn keep_stderr(dir: &Path, binary: &str, pid: u32, stderr: &str) -> Result<PathBuf, String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    prune(dir, KEEP - 1);
    let path = dir.join(format!("{binary}-{pid}.stderr"));
    fs::write(&path, stderr).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// The last lines of a process's stderr that the process itself printed.
///
/// Wine's own chatter is dropped first: dozens of `fixme:dbghelp` lines
/// surround every backtrace, so a fixed count of raw lines rarely reaches
/// back to the message that says what failed. Nothing is lost by it, since
/// the whole stderr goes to the file [`Launcher::keep_stderr`] names.
#[must_use]
pub fn stderr_tail(stderr: &str) -> String {
    let lines: Vec<&str> = stderr
        .lines()
        .filter(|line| !is_wine_chatter(line))
        .collect();
    let start = lines.len().saturating_sub(TAIL_LINES);
    lines[start..].join("\n")
}

/// Whether Wine printed the line rather than the process running under it.
///
/// A channel line is `<class>:<channel>:<function> <message>`. Every
/// `fixme` is an unimplemented notice, and `dbghelp` fills the lines around
/// a backtrace whatever its class; neither ever carries the failure.
fn is_wine_chatter(line: &str) -> bool {
    let mut fields = line.splitn(3, ':');
    let (Some(class), Some(channel), Some(_)) = (fields.next(), fields.next(), fields.next())
    else {
        return false;
    };
    WINE_CLASSES.contains(&class) && (class == "fixme" || channel == "dbghelp")
}

/// The part of a layer log that accounts for the process's end.
///
/// From the fatal banner when the layer wrote one, since everything after it
/// is the crash report and everything before it is ordinary work; the last
/// lines otherwise, which are what the layer was doing when it stopped.
#[must_use]
pub fn layer_tail(log: &str) -> String {
    let lines: Vec<&str> = log.lines().collect();
    let fatal = lines
        .iter()
        .position(|line| line.contains(mtld3d_shared::fatal::BANNER));
    let (start, len) = fatal.map_or_else(
        || (lines.len().saturating_sub(TAIL_LINES), TAIL_LINES),
        |at| (at, FATAL_LINES),
    );
    let end = start.saturating_add(len).min(lines.len());
    lines[start..end].join("\n")
}

/// Remove the oldest kept stderr files in `dir` beyond the newest `keep`.
///
/// Age is the modification time; an entry whose metadata cannot be read is
/// left alone, and so is one whose removal fails, which the next process to
/// die tries again. Nothing here reports, because a directory that cannot
/// be tidied is not a reason to say less about the process that died.
fn prune(dir: &Path, keep: usize) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut aged: Vec<(SystemTime, PathBuf)> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "stderr"))
        .filter_map(|path| Some((fs::metadata(&path).ok()?.modified().ok()?, path)))
        .collect();
    if aged.len() <= keep {
        return;
    }
    // Newest last; a tie keeps the order stable by name.
    aged.sort();
    let doomed = aged.len() - keep;
    for (_, path) in aged.into_iter().take(doomed) {
        let _ = fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests;
