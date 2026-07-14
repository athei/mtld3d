fn main() {
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
