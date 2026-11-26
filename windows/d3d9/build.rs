use std::fmt::Write;

fn main() {
    // The export list is the PE link's; a native build exports its two
    // factories by name.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        pe_link_args();
    } else {
        native_d3d_set();
    }

    // Diagnostic-only: `MTLD3D_CRUMB=1` enables the mmap breadcrumb in
    // `crate::crumb`. Routed through a build-script cfg instead of a
    // `RUSTFLAGS --cfg` flag so it composes with the xwin-link-path
    // rustflags set by `.cargo/config.toml` (env RUSTFLAGS *replaces*
    // those, it does not append).
    println!("cargo::rustc-check-cfg=cfg(mtld3d_crumb)");
    println!("cargo:rerun-if-env-changed=MTLD3D_CRUMB");
    if std::env::var("MTLD3D_CRUMB").is_ok_and(|v| !v.is_empty() && v != "0") {
        println!("cargo:rustc-cfg=mtld3d_crumb");
    }

    println!("cargo::rustc-check-cfg=cfg(perf_tracking)");
    println!("cargo:rerun-if-env-changed=MTLD3D_PERF");
    if std::env::var("MTLD3D_PERF").is_ok_and(|v| !v.is_empty() && v != "0") {
        println!("cargo:rustc-cfg=perf_tracking");
    }
}

/// Fails a native build of `d3d9` whose `mtld3d-unix` would take the Wine host.
///
/// `d3d9` picks its native side by target and `mtld3d-unix` its host by
/// `MTLD3D_NATIVE_D3D`, so a native target needs the variable too.
fn native_d3d_set() {
    println!("cargo:rerun-if-env-changed=MTLD3D_NATIVE_D3D");
    let set = std::env::var("MTLD3D_NATIVE_D3D").is_ok_and(|v| !v.is_empty() && v != "0");
    assert!(
        set,
        "a non-Windows d3d9 is the native build: set MTLD3D_NATIVE_D3D=1 (see `make native-d3d`)"
    );
}

/// The PE link's export list, from `d3d9.def`.
fn pe_link_args() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let def = format!("{manifest_dir}/d3d9.def");
    println!("cargo:rerun-if-changed={def}");
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86") {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let i686_def = format!("{out_dir}/d3d9.i686.def");
        let text = std::fs::read_to_string(&def).expect("read d3d9.def");
        std::fs::write(&i686_def, decorate_def(&text)).expect("write the i686 d3d9.def");
        println!("cargo:rustc-cdylib-link-arg=/DEF:{i686_def}");
    } else {
        println!("cargo:rustc-cdylib-link-arg=/DEF:{def}");
    }
}

/// `d3d9.def` for the i686 link: every export entry pointed at its stdcall symbol.
///
/// rustc exports each entry point itself, under its decorated symbol
/// (`_Name@N`). lld keeps an entry of the `.def` only when it names that same
/// symbol, so an undecorated entry is dropped, ordinal and `NONAME` with it.
/// The argument bytes `N` come from the comment after each entry; an entry
/// without them, or without an `@ordinal`, fails the build. Lines outside
/// `EXPORTS` are copied as they are.
fn decorate_def(text: &str) -> String {
    let mut out = String::new();
    let mut in_exports = false;
    for line in text.lines() {
        let (entry, comment) = line.split_once(';').unwrap_or((line, ""));
        let mut fields = entry.split_whitespace();
        let (true, Some(name)) = (in_exports, fields.next()) else {
            in_exports |= entry.trim() == "EXPORTS";
            out.push_str(line);
            out.push('\n');
            continue;
        };
        let ordinal = fields
            .next()
            .filter(|ordinal| ordinal.starts_with('@'))
            .unwrap_or_else(|| panic!("d3d9.def: the entry {name} has no `@ordinal`"));
        let bytes: u32 = comment.trim().parse().unwrap_or_else(|_| {
            panic!("d3d9.def: the entry {name} names no i386 argument bytes after `;`")
        });
        let rest: Vec<&str> = fields.collect();
        writeln!(
            out,
            "    {name}=_{name}@{bytes} {ordinal} {}",
            rest.join(" ")
        )
        .expect("formatting into a String cannot fail");
    }
    out
}
