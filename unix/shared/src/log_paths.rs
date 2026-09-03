//! Names of the files one process writes into its log directory.
//!
//! Both sides agree on them: the PE side names the directory, the unix side
//! opens the files. One log per process, so a launch never overwrites the
//! log of the one before it, and the GPU traces of a process sit next to
//! its log under the same prefix.

/// The log file of process `pid` running executable `stem`: `<stem>-<pid>.log`.
#[must_use]
pub fn log_file_name(stem: &str, pid: u32) -> String {
    format!("{stem}-{pid}.log")
}

/// The `index`-th GPU trace of that process: `<stem>-<pid>-<index>.gputrace`.
#[must_use]
pub fn trace_file_name(stem: &str, pid: u32, index: u32) -> String {
    format!("{stem}-{pid}-{index}.gputrace")
}

#[cfg(test)]
mod tests;
