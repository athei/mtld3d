//! Unit tests for the binary naming and for what a dead process's stderr leaves behind.

use std::path::{Path, PathBuf};

use super::{KEEP, binary_name, keep_stderr, stderr_tail};

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
fn the_kept_files_are_capped_and_never_touch_the_layers_logs() {
    let dir = dir("capped");
    std::fs::write(dir.join("e2e-1.log"), "the layer's own").expect("log");
    for pid in 0..u32::try_from(KEEP).expect("small") + 5 {
        keep_stderr(&dir, "e2e", pid, "stderr").expect("kept");
    }
    let kept = |ext: &str| {
        std::fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == ext))
            .count()
    };
    assert_eq!(kept("stderr"), KEEP);
    assert_eq!(kept("log"), 1);
}
