fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-cdylib-link-arg=/DEF:{manifest_dir}/d3d9.def");

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
