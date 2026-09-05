//! Unit tests for the binary naming.

use std::path::Path;

use super::{binary_name, stderr_tail};

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
