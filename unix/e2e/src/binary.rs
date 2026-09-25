//! A test binary on disk: its name, the launcher that runs it, and the accounts of its processes.
//!
//! Two accounts survive a process the runner lost: the stderr it kept in a
//! file of its own, and the log the layer wrote, which is where the layer's
//! crash report goes and the only place it goes. The layer keeps only the
//! newest ten of its logs and removes the rest as later processes create
//! theirs, so the runner moves the log of a lost process to a name that
//! retention never matches, beside the stderr, and caps those itself.

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

/// How many kept files of each kind a directory holds, as many as the layer keeps logs.
const KEEP: usize = 10;

/// A completed abnormal process's captured streams and exit status.
const PROCESS_LOG_EXT: &str = "process-log";

/// Windows command-line capacity in UTF-16 units, including the terminating NUL.
const COMMAND_LINE_UNITS: usize = 32_767;

/// The extension of a kept stderr: `<binary>-<pid>.stderr`.
const STDERR_EXT: &str = "stderr";

/// The extension of a kept layer log: `<binary>-<pid>.layer-log`.
///
/// Anything but `log`: that is the extension the layer's own retention
/// matches, and a file under it is gone once ten newer processes have
/// logged, which one run of the suite nearly does.
const LAYER_LOG_EXT: &str = "layer-log";

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
    /// Whether every process runs the `#[ignore]` tests only (libtest's `--ignored`).
    ignored: bool,
    /// A line as the process printed it, for the progress a caller shows.
    on_line: Box<dyn FnMut(&str)>,
}

impl WineLauncher {
    /// A launcher for `exe`, keeping dead processes' stderr in `log_dir`.
    ///
    /// `log_dir` is where the layer writes its own per-process logs, so the
    /// two accounts of one process sit together; `None` means the layer's
    /// default, `mtld3d-logs` beside the executable.
    ///
    /// # Errors
    ///
    /// Returns a message if the executable path cannot be made absolute.
    pub fn new(
        wine: &Path,
        exe: &Path,
        log_dir: Option<&Path>,
        timeout: Duration,
        on_line: Box<dyn FnMut(&str)>,
    ) -> Result<Self, String> {
        let exe = std::path::absolute(exe)
            .map_err(|e| format!("could not resolve {}: {e}", exe.display()))?;
        let log_dir = log_dir.map_or_else(
            || {
                exe.parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join("mtld3d-logs")
            },
            Path::to_path_buf,
        );
        Ok(Self {
            wine: wine.to_path_buf(),
            exe,
            log_dir,
            timeout,
            ignored: false,
            on_line,
        })
    }

    /// Run the tests marked `#[ignore]` instead of the others, listing included.
    #[must_use]
    pub const fn ignored_only(mut self, ignored: bool) -> Self {
        self.ignored = ignored;
        self
    }
}

impl Launcher for WineLauncher {
    fn batch_len(&self, names: &[String], threads: u32) -> Result<usize, String> {
        fitting_prefix(&self.exe, names, threads, self.ignored)
    }

    fn run(
        &mut self,
        names: Option<&[String]>,
        threads: u32,
        on_event: &mut dyn FnMut(Event),
    ) -> Result<ProcessEnd, String> {
        let args = test_arguments(names, threads, self.ignored);
        let mut parser = Parser::default();
        let mut stdout = String::new();
        let exit = run::run(&self.wine, &self.exe, &args, self.timeout, &mut |line| {
            (self.on_line)(line);
            stdout.push_str(line);
            stdout.push('\n');
            for event in parser.line(line) {
                on_event(event);
            }
        })?;
        let layer_gpu_hang = self.layer_reported_gpu_hang(exit.pid)?;
        Ok(ProcessEnd {
            pid: exit.pid,
            kind: exit.kind,
            stdout,
            stderr: exit.stderr,
            gpu_hang: exit.gpu_hang || layer_gpu_hang,
        })
    }

    fn list(&mut self) -> Result<Vec<String>, String> {
        let mut stdout = String::new();
        let mut args = vec!["--list".to_owned()];
        if self.ignored {
            args.push("--ignored".to_owned());
        }
        let exit = run::run(&self.wine, &self.exe, &args, self.timeout, &mut |line| {
            stdout.push_str(line);
            stdout.push('\n');
        })?;
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

    fn keep_process(&self, end: &ProcessEnd) -> Result<PathBuf, String> {
        keep_process(&self.log_dir, &binary_name(&self.exe), end)
    }

    fn keep_layer_log(&self, pid: u32) -> Result<LayerLog, String> {
        // The layer names the file after the executable's whole stem, cargo
        // hash and all, and after the host pid, which is the one the runner
        // spawned the process under.
        let stem = self.exe_stem();
        keep_layer_log(&self.log_dir, stem, &binary_name(&self.exe), pid)
    }
}

impl WineLauncher {
    /// The executable stem the layer uses in its per-process log name.
    fn exe_stem(&self) -> &str {
        self.exe
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
    }

    /// The layer's live log path for `pid`, before the runner preserves it.
    fn layer_log_path(&self, pid: u32) -> PathBuf {
        self.log_dir.join(mtld3d_shared::log_paths::log_file_name(
            self.exe_stem(),
            pid,
        ))
    }

    /// Whether the layer's completed log reports a GPU hang.
    ///
    /// A process that never loaded mtld3d has no log. Any other read error
    /// ends the runner rather than letting another process use a GPU whose
    /// state could not be checked.
    fn layer_reported_gpu_hang(&self, pid: u32) -> Result<bool, String> {
        let path = self.layer_log_path(pid);
        match fs::read_to_string(&path) {
            Ok(text) => Ok(run::is_gpu_hang_report(&text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(format!(
                "could not read {} after process exit: {e}",
                path.display()
            )),
        }
    }
}

/// The layer's log of a dead process: where it is now, and its account of the end.
pub struct LayerLog {
    /// The kept file, or the layer's own when the move failed.
    pub path: PathBuf,
    /// The lines that account for the end, see [`layer_tail`].
    pub tail: String,
    /// Why the file is still the layer's own, when it is.
    pub not_kept: Option<String>,
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
    keep_text(dir, binary, pid, STDERR_EXT, stderr)
}

/// Keep full captured text and exit metadata together after abnormal completion.
///
/// Lengths frame the captured strings even when they contain section headings.
/// stdout has already been decoded line by line and stderr lossily, so this is
/// the runner's captured text, not a byte-exact recording of the original pipes.
///
/// # Errors
///
/// Returns the reason when the directory or the file cannot be written.
pub fn keep_process(dir: &Path, binary: &str, end: &ProcessEnd) -> Result<PathBuf, String> {
    let text = format!(
        "binary: {binary}\npid: {}\nexit: {}\nstdout-bytes: {}\nstderr-bytes: {}\n\nstdout:\n{}\nstderr:\n{}",
        end.pid,
        end.kind.describe(),
        end.stdout.len(),
        end.stderr.len(),
        end.stdout,
        end.stderr,
    );
    keep_text(dir, binary, end.pid, PROCESS_LOG_EXT, &text)
}

/// Write one retained account using the same per-kind file budget.
fn keep_text(dir: &Path, binary: &str, pid: u32, ext: &str, text: &str) -> Result<PathBuf, String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    prune(dir, ext, KEEP - 1);
    let path = dir.join(format!("{binary}-{pid}.{ext}"));
    fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// Move the layer's log of the dead process `pid` out of the layer's retention.
///
/// The layer's file is `<stem>-<pid>.log` in `dir`; it becomes
/// `<binary>-<pid>.layer-log` beside the kept stderr, the newest ten of
/// which stay. The account of the end is read before the move, so a move
/// that fails still quotes it and names the file where it is.
///
/// # Errors
///
/// Returns the reason when the layer's file cannot be read, which for a
/// process that logged nothing is that it was never created.
pub fn keep_layer_log(dir: &Path, stem: &str, binary: &str, pid: u32) -> Result<LayerLog, String> {
    let from = dir.join(mtld3d_shared::log_paths::log_file_name(stem, pid));
    let text = fs::read_to_string(&from).map_err(|e| format!("{}: {e}", from.display()))?;
    let tail = layer_tail(&text);
    prune(dir, LAYER_LOG_EXT, KEEP - 1);
    let to = dir.join(format!("{binary}-{pid}.{LAYER_LOG_EXT}"));
    Ok(match fs::rename(&from, &to) {
        Ok(()) => LayerLog {
            path: to,
            tail,
            not_kept: None,
        },
        Err(e) => LayerLog {
            path: from,
            tail,
            not_kept: Some(format!("could not be moved to {}: {e}", to.display())),
        },
    })
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

/// Build the arguments shared by command sizing and process launch.
///
/// `ignored` adds libtest's `--ignored`, so the process runs the tests marked
/// `#[ignore]` among those it is given and no others.
fn test_arguments(names: Option<&[String]>, threads: u32, ignored: bool) -> Vec<String> {
    let mut args = vec![
        format!("--test-threads={threads}"),
        "--nocapture".to_owned(),
    ];
    if ignored {
        args.push("--ignored".to_owned());
    }
    if let Some(names) = names {
        args.push("--exact".to_owned());
        args.extend(names.iter().cloned());
    }
    args
}

/// Fit a selection after allowing for Wine's executable-path mapping.
fn fitting_prefix(
    exe: &Path,
    names: &[String],
    threads: u32,
    ignored: bool,
) -> Result<usize, String> {
    // Wine replaces an absolute Unix path's leading mapped directory with a
    // drive prefix, or uses the longer \\?\unix prefix. Keeping the whole
    // absolute path plus that eight-unit prefix bounds either spelling.
    // argv[0] is always quoted; path separators become backslashes before
    // quoting, so a separator before a literal quote must be counted too.
    let image = format!(r"\\?\unix{}", exe.to_string_lossy().replace('/', "\\"));
    let mut units = argument_units(&image, true).saturating_add(1);
    for arg in test_arguments(Some(&[]), threads, ignored) {
        units = units.saturating_add(1 + argument_units(&arg, false));
    }
    let mut count = 0;
    for name in names {
        let next = units.saturating_add(1 + argument_units(name, false));
        if next > COMMAND_LINE_UNITS {
            break;
        }
        units = next;
        count += 1;
    }
    if count == 0 && !names.is_empty() {
        return Err(format!(
            "test {:?} does not fit the Windows command-line limit of {COMMAND_LINE_UNITS} \
             UTF-16 units including flags, NUL and the executable-path allowance for {}",
            names[0],
            exe.display()
        ));
    }
    Ok(count)
}

/// Size one argument after Wine's Windows command-line quoting.
fn argument_units(argument: &str, force_quotes: bool) -> usize {
    let quoted = force_quotes || argument.is_empty() || argument.contains([' ', '\t']);
    let mut units = usize::from(quoted) * 2;
    let mut backslashes = 0;
    for character in argument.chars() {
        units += character.len_utf16();
        if character == '\\' {
            backslashes += 1;
        } else {
            if character == '"' {
                units += backslashes + 1;
            }
            backslashes = 0;
        }
    }
    units + if quoted { backslashes } else { 0 }
}

/// Remove the oldest kept files with extension `ext` in `dir` beyond the newest `keep`.
///
/// Age is the modification time; an entry whose metadata cannot be read is
/// left alone, and so is one whose removal fails, which the next process to
/// die tries again. Nothing here reports, because a directory that cannot
/// be tidied is not a reason to say less about the process that died.
fn prune(dir: &Path, ext: &str, keep: usize) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut aged: Vec<(SystemTime, PathBuf)> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|found| found == ext))
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
