//! The banner the unix side writes when a fatal signal ends the process.
//!
//! It goes to the process's log file and nowhere else, because the log file
//! is where every line of both sides goes and a signal handler cannot pick a
//! second destination. A parent that reads only the process's pipes
//! therefore sees a bare exit status for a death the layer reported in full,
//! so anything that has to recognise such a death (the e2e runner, reading
//! the log of a process it lost) looks for this prefix, and the two sides
//! name it once.

/// The first thing the crash handler writes, followed by the signal's name.
pub const BANNER: &str = "[mtld3d::unix] FATAL: ";
