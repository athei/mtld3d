//! Unit tests for the binary naming and for the accounts a dead process leaves behind.

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use super::{
    FATAL_LINES, KEEP, LAYER_LOG_EXT, WineLauncher, binary_name, keep_layer_log, keep_stderr,
    layer_tail, stderr_tail,
};
use crate::attribute::Launcher as _;

/// A fresh directory under the temp dir, named after the test using it.
fn dir(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtld3d-e2e-stderr-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("temp dir");
    path
}

/// Dozens of the `fixme:dbghelp` lines Wine prints around a backtrace.
fn chatter(count: usize) -> String {
    (0..count)
        .map(|i| format!("fixme:dbghelp:elf_search_auxv can't find symbol {i}\n"))
        .collect::<Vec<_>>()
        .concat()
}

#[test]
fn the_name_drops_cargos_hash_and_nothing_else() {
    assert_eq!(
        binary_name(Path::new("/t/deps/e2e-1030f4ab05278ecb.exe")),
        "e2e"
    );
    assert_eq!(
        binary_name(Path::new("snmalloc_drift-0a1b2c3d4e5f6071.exe")),
        "snmalloc_drift"
    );
    assert_eq!(binary_name(Path::new("unload.exe")), "unload");
    assert_eq!(
        binary_name(Path::new("multi-device.exe")),
        "multi-device",
        "a dash followed by a non-hex word is part of the name"
    );
}

#[test]
fn the_tail_keeps_the_last_lines() {
    let text = (0..20)
        .map(|i| format!("line {i}\n"))
        .collect::<Vec<_>>()
        .concat();
    let tail = stderr_tail(&text);
    assert!(tail.starts_with("line 5\n"));
    assert!(tail.ends_with("line 19"));
    assert_eq!(stderr_tail("a\nb"), "a\nb");
}

#[test]
fn the_tail_reaches_past_wines_chatter_to_the_failure() {
    let stderr = format!(
        "IOSurfaceClientCreate failed\n{}wine = 929;\nerr:dbghelp:elf_map_file cannot read\n",
        chatter(40)
    );
    let tail = stderr_tail(&stderr);
    assert_eq!(tail, "IOSurfaceClientCreate failed\nwine = 929;");
}

#[test]
fn a_wine_channel_line_is_chatter_only_for_fixme_and_dbghelp() {
    let stderr = "err:module:import_dll library not found\nfixme:d3d:unimplemented\ntrace:dbghelp:noise\nplain: text: line\n";
    assert_eq!(
        stderr_tail(stderr),
        "err:module:import_dll library not found\nplain: text: line"
    );
}

#[test]
fn the_whole_stderr_lands_in_a_file_the_report_can_name() {
    let dir = dir("kept");
    let stderr = format!("IOSurfaceClientCreate failed\n{}", chatter(40));
    let path = keep_stderr(&dir, "e2e", 4242, &stderr).expect("kept");
    assert_eq!(path, dir.join("e2e-4242.stderr"));
    assert_eq!(std::fs::read_to_string(&path).expect("read"), stderr);
}

#[test]
fn the_kept_files_are_capped_per_kind_and_never_touch_the_layers_logs() {
    let dir = dir("capped");
    std::fs::write(dir.join("e2e-1.log"), "the layer's own").expect("log");
    for pid in 0..u32::try_from(KEEP).expect("small") + 5 {
        keep_stderr(&dir, "e2e", pid, "stderr").expect("kept");
        std::fs::write(dir.join(format!("e2e-abcd-{pid}.log")), "dead").expect("layer log");
        keep_layer_log(&dir, "e2e-abcd", "e2e", pid).expect("kept");
    }
    let kept = |ext: &str| {
        std::fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == ext))
            .count()
    };
    assert_eq!(kept("stderr"), KEEP);
    assert_eq!(kept(LAYER_LOG_EXT), KEEP);
    assert_eq!(kept("log"), 1, "a live process's log is not the runner's");
}

#[test]
fn a_layer_log_is_read_from_its_fatal_line_on() {
    let work: String = (0..30)
        .map(|i| format!("[2026-01-01T00:00:00Z INFO  mtld3d::unix] work {i}\n"))
        .collect::<Vec<_>>()
        .concat();
    let log = format!(
        "{work}[mtld3d::unix] FATAL: SIGSEGV fault=0x0\n[mtld3d::unix] thread=mtld3d-encoder\n"
    );
    let tail = layer_tail(&log);
    assert!(tail.starts_with("[mtld3d::unix] FATAL: SIGSEGV"), "{tail}");
    assert!(tail.ends_with("thread=mtld3d-encoder"), "{tail}");
    assert!(!tail.contains("work 29"), "{tail}");
}

#[test]
fn a_layer_log_without_a_fatal_line_is_tailed_like_stderr() {
    let log = (0..20)
        .map(|i| format!("line {i}\n"))
        .collect::<Vec<_>>()
        .concat();
    assert_eq!(layer_tail(&log), stderr_tail(&log));
    assert!(layer_tail(&log).starts_with("line 5\n"));
}

#[test]
fn a_fatal_line_early_in_a_long_log_is_quoted_to_a_bound() {
    let log = format!(
        "[mtld3d::unix] FATAL: SIGABRT\n{}",
        (0..200)
            .map(|i| format!("frame {i}\n"))
            .collect::<Vec<_>>()
            .concat()
    );
    assert_eq!(layer_tail(&log).lines().count(), FATAL_LINES);
}

#[test]
fn the_layers_log_of_a_dead_process_is_moved_out_of_the_layers_retention() {
    let dir = dir("layer");
    let exe = Path::new("/t/deps/e2e-1030f4ab05278ecb.exe");
    let launcher = WineLauncher::new(
        Path::new("/usr/bin/true"),
        exe,
        Some(&dir),
        Duration::from_secs(1),
        Box::new(|_| {}),
    );
    assert!(
        launcher.keep_layer_log(4242).is_err(),
        "a process that logged nothing has no file"
    );
    let log = "ordinary line\n[mtld3d::unix] FATAL: SIGSEGV fault=0x8\n";
    let own = dir.join("e2e-1030f4ab05278ecb-4242.log");
    std::fs::write(&own, log).expect("layer log");
    let kept = launcher.keep_layer_log(4242).expect("the layer log");
    assert_eq!(kept.path, dir.join("e2e-4242.layer-log"));
    assert_eq!(kept.tail, "[mtld3d::unix] FATAL: SIGSEGV fault=0x8");
    assert!(kept.not_kept.is_none());
    assert!(
        kept.path.extension().is_none_or(|ext| ext != "log"),
        "the layer prunes by the `log` extension"
    );
    assert!(!own.exists(), "moved, not copied");
    assert_eq!(std::fs::read_to_string(&kept.path).expect("read"), log);
}

#[test]
fn a_layer_log_that_cannot_be_moved_is_still_quoted_where_it_is() {
    let dir = dir("unmovable");
    let own = dir.join("e2e-abcd-7.log");
    std::fs::write(&own, "[mtld3d::unix] FATAL: SIGBUS\n").expect("layer log");
    // A directory in the way of the kept name makes the move fail.
    std::fs::create_dir(dir.join("e2e-7.layer-log")).expect("blocker");
    let kept = keep_layer_log(&dir, "e2e-abcd", "e2e", 7).expect("read");
    assert_eq!(kept.path, own);
    assert_eq!(kept.tail, "[mtld3d::unix] FATAL: SIGBUS");
    assert!(
        kept.not_kept
            .is_some_and(|why| why.contains("e2e-7.layer-log"))
    );
}
