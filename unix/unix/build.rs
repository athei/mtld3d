const DELEGATE_FORWARD: &str = "src/metal/macdrv/wine/delegate_forward.m";

fn main() {
    println!("cargo::rustc-check-cfg=cfg(perf_tracking)");
    println!("cargo:rerun-if-env-changed=MTLD3D_PERF");
    if std::env::var("MTLD3D_PERF").is_ok_and(|value| !value.is_empty() && value != "0") {
        println!("cargo:rustc-cfg=perf_tracking");
    }
    println!("cargo::rustc-check-cfg=cfg(native_d3d)");
    println!("cargo:rerun-if-env-changed=MTLD3D_NATIVE_D3D");
    let native_d3d = std::env::var("MTLD3D_NATIVE_D3D").is_ok_and(|v| !v.is_empty() && v != "0");
    if native_d3d {
        println!("cargo:rustc-cfg=native_d3d");
    }

    // The delegate forward is the Wine host's, so a native build skips it.
    let target = std::env::var("TARGET").unwrap();
    if !target.contains("apple") || native_d3d {
        return;
    }

    // Every Rust frame of this crate aborts on unwind (`panic = "abort"`),
    // so a message send whose Objective-C exception has to be caught keeps
    // its `@try` and the send together in Objective-C.
    println!("cargo:rerun-if-changed={DELEGATE_FORWARD}");
    cc::Build::new()
        .file(DELEGATE_FORWARD)
        .flag("-fobjc-exceptions")
        .flag("-fno-objc-arc")
        .warnings_into_errors(true)
        .compile("mtld3d_delegate_forward");
}
