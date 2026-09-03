//! File names of one process's log and traces.
//!
//! They share the `<stem>-<pid>` prefix, so a process's traces sort next to its log; the
//! trace index is the only difference.

use super::{log_file_name, trace_file_name};

#[test]
fn log_and_trace_names_share_the_process_prefix() {
    assert_eq!(log_file_name("hl2", 1234), "hl2-1234.log");
    assert_eq!(trace_file_name("hl2", 1234, 1), "hl2-1234-1.gputrace");
    assert!(trace_file_name("hl2", 1234, 7).starts_with("hl2-1234-"));
}
