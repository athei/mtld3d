fn main() {
    let target = std::env::var("TARGET").unwrap();
    if !target.contains("windows") {
        return;
    }

    // Wine's builtin directory for this target, which is also the path
    // `libwinecrt0.a` files its per-arch objects under. Spelled out rather than
    // defaulted, so a target we have not thought about fails here instead of
    // silently linking another arch's `unix_lib.o`. arm64ec builds are ARM64
    // code and take the ARM64 tree, matching Wine's own loader (`get_pe_dir`
    // maps `IMAGE_FILE_MACHINE_ARM64` to `aarch64-windows`, and a hybrid module
    // requested as AMD64 is redirected there).
    let wine_arch = match target.split('-').next().expect("target triple has an arch") {
        "x86_64" => "x86_64-windows",
        "i686" => "i386-windows",
        "aarch64" | "arm64ec" => "aarch64-windows",
        arch => panic!("no Wine builtin directory known for target arch `{arch}`"),
    };

    let wine_sdk = std::env::var("WINE_SDK").expect("WINE_SDK must be set");
    let lib_dir = format!("{wine_sdk}/lib/wine/{wine_arch}");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Homebrew installs llvm keg-only (not on PATH), so resolve `llvm-ar`
    // explicitly: honour an `LLVM_AR` override, else ask Homebrew for its llvm
    // prefix (covers Apple-Silicon `/opt/homebrew` and Intel `/usr/local`),
    // else fall back to the default Apple-Silicon keg path.
    let llvm_ar = std::env::var("LLVM_AR").unwrap_or_else(|_| resolve_llvm_ar());

    // Extract just unix_lib.o from winecrt0.a - avoids TLS symbol conflicts
    // with MSVC's CRT (both define __tls_index, __tls_start, etc.)
    let output = std::process::Command::new(&llvm_ar)
        .args([
            "p",
            &format!("{lib_dir}/libwinecrt0.a"),
            &format!("libs/winecrt0/{wine_arch}/unix_lib.o"),
        ])
        .output()
        .expect("ar failed");
    assert!(
        output.status.success(),
        "could not extract libs/winecrt0/{wine_arch}/unix_lib.o from \
         {lib_dir}/libwinecrt0.a: is WINE_SDK a Wine build tree that carries \
         this architecture?"
    );

    let unix_lib_path = format!("{out_dir}/unix_lib.o");
    std::fs::write(&unix_lib_path, &output.stdout).expect("failed to write unix_lib.o");
    println!("cargo:rustc-link-arg-cdylib={unix_lib_path}");

    // Wine's libntdll.a must be found before xwin's ntdll.lib for RtlFindExportedRoutineByName
    println!("cargo:rustc-link-arg-cdylib=-L{lib_dir}");

    println!("cargo:rerun-if-env-changed=WINE_SDK");
    println!("cargo:rerun-if-env-changed=LLVM_AR");
}

/// Resolve the `llvm-ar` binary.
///
/// Homebrew installs llvm keg-only (not on PATH), so query `brew --prefix
/// llvm` and fall back to the default Apple-Silicon keg location if Homebrew
/// is unavailable.
fn resolve_llvm_ar() -> String {
    std::process::Command::new("brew")
        .args(["--prefix", "llvm"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .filter(|prefix| !prefix.is_empty())
        .map_or_else(
            || "/opt/homebrew/opt/llvm/bin/llvm-ar".to_owned(),
            |prefix| format!("{prefix}/bin/llvm-ar"),
        )
}
