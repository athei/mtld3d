//! A test binary on disk: its name, and the launcher that runs it under Wine.

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use crate::{
    attribute::{Launcher, ProcessEnd},
    libtest::{self, Event, Parser},
    run::{self, ExitKind},
};

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
    timeout: Duration,
    /// A line as the process printed it, for the progress a caller shows.
    on_line: Box<dyn FnMut(&str)>,
}

impl WineLauncher {
    pub fn new(wine: &Path, exe: &Path, timeout: Duration, on_line: Box<dyn FnMut(&str)>) -> Self {
        Self {
            wine: wine.to_path_buf(),
            exe: exe.to_path_buf(),
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
}

/// The last lines of a process's stderr, for a report of how it died.
#[must_use]
pub fn stderr_tail(stderr: &str) -> String {
    const LINES: usize = 15;
    let lines: Vec<&str> = stderr.lines().collect();
    let start = lines.len().saturating_sub(LINES);
    lines[start..].join("\n")
}

#[cfg(test)]
mod tests;
