fn main() {
    let target = std::env::var("TARGET").unwrap();
    if !target.contains("windows") {
        return;
    }

    // Where Wine keeps this target's libraries, and the path its per-arch
    // objects carry inside `libwinecrt0.a`. The two differ for arm64ec: an
    // ARM64X build (`--enable-archs=arm64ec,aarch64`) installs one archive
    // under the native arch and packs BOTH arches' objects into it
    // (`output_static_lib` in Wine's makedep adds `object_files[hybrid_arch]`),
    // so the EC build takes the EC object out of the aarch64 archive. That also
    // matches the loader, whose `get_pe_dir` knows no arm64ec directory and
    // redirects a hybrid module requested as AMD64 to `aarch64-windows`.
    //
    // Spelled out rather than defaulted, so a target we have not thought about
    // fails here instead of silently linking another arch's `unix_lib.o`.
    let (lib_arch, object_arch) = match target.split('-').next().expect("target triple has an arch")
    {
        "x86_64" => ("x86_64-windows", "x86_64-windows"),
        "i686" => ("i386-windows", "i386-windows"),
        "aarch64" => ("aarch64-windows", "aarch64-windows"),
        "arm64ec" => ("aarch64-windows", "arm64ec-windows"),
        arch => panic!("no Wine builtin directory known for target arch `{arch}`"),
    };

    let wine_sdk = std::env::var("WINE_SDK").expect("WINE_SDK must be set");
    let lib_dir = format!("{wine_sdk}/lib/wine/{lib_arch}");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // `llvm-ar` is never on PATH under either of the names it can arrive as, so
    // resolve it explicitly: an `LLVM_AR` override first, then the toolchain's
    // own `llvm-tools` copy, then Homebrew's keg for a toolchain without that
    // component. It has to be `llvm-ar` and not the `llvm-lib` symlink beside it,
    // because the extraction below speaks ar syntax, not lib.exe syntax.
    let llvm_ar = std::env::var("LLVM_AR").unwrap_or_else(|_| resolve_llvm_ar());

    // Take just unix_lib.o out of winecrt0.a, which avoids the TLS symbol
    // conflicts with MSVC's CRT (both define __tls_index, __tls_start, etc.).
    // Absolute, because the extraction runs with its working directory set to
    // `OUT_DIR`; canonicalizing here also turns a missing archive into an error
    // naming the file rather than an `ar` failure.
    let archive = std::fs::canonicalize(format!("{lib_dir}/libwinecrt0.a"))
        .unwrap_or_else(|err| panic!("{lib_dir}/libwinecrt0.a: {err}"));
    let unix_lib_path =
        extract_unix_lib(&llvm_ar, &archive.to_string_lossy(), object_arch, &out_dir);
    println!("cargo:rustc-link-arg-cdylib={unix_lib_path}");

    // Wine's libntdll.a must be found before xwin's ntdll.lib for RtlFindExportedRoutineByName
    println!("cargo:rustc-link-arg-cdylib=-L{lib_dir}");

    println!("cargo:rerun-if-env-changed=WINE_SDK");
    println!("cargo:rerun-if-env-changed=LLVM_AR");
}

/// Extract `unix_lib.o` for `object_arch` from `archive` into `out_dir`.
///
/// Returns the path of the extracted object. `llvm-ar` matches members by
/// BASENAME, so the directory part of a member name selects nothing: an ARM64X
/// `libwinecrt0.a` carries two `unix_lib.o` members (the ARM64 one and the EC
/// one) and a plain `ar p <path>` would hand back whichever comes first. The
/// stored names are read first, then the wanted one is pulled out by instance
/// number, which is the only selector `ar` offers for a repeated name. Reading
/// them also survives Wine moving winecrt0 between `dlls/` and `libs/`, which
/// it has done before.
///
/// # Panics
///
/// If the archive is unreadable or carries no `unix_lib.o` for this arch, which
/// is what a `WINE_SDK` built without this architecture looks like.
fn extract_unix_lib(llvm_ar: &str, archive: &str, object_arch: &str, out_dir: &str) -> String {
    let listing = std::process::Command::new(llvm_ar)
        .args(["t", archive])
        .output()
        .expect("ar failed");
    assert!(
        listing.status.success(),
        "could not read {archive}: is WINE_SDK a Wine build tree carrying this architecture?"
    );
    let listing = String::from_utf8(listing.stdout).expect("ar member names are UTF-8");

    // 1-based instance number among the members sharing this basename, which is
    // what `ar`'s `N` modifier counts.
    let instance = listing
        .lines()
        .filter(|member| member.ends_with("unix_lib.o"))
        .position(|member| member.contains(&format!("{object_arch}/")))
        .map_or_else(
            || panic!("{archive} carries no {object_arch}/unix_lib.o"),
            |index| index + 1,
        );

    let status = std::process::Command::new(llvm_ar)
        .current_dir(out_dir)
        .args(["xN", &instance.to_string(), archive, "unix_lib.o"])
        .status()
        .expect("ar failed");
    assert!(
        status.success(),
        "extracting unix_lib.o from {archive} failed"
    );
    format!("{out_dir}/unix_lib.o")
}

/// Resolve the `llvm-ar` binary.
///
/// Homebrew installs llvm keg-only (not on PATH), so query `brew --prefix
/// llvm` and fall back to the default Apple-Silicon keg location if Homebrew
/// is unavailable.
fn resolve_llvm_ar() -> String {
    toolchain_llvm_ar().unwrap_or_else(homebrew_llvm_ar)
}

/// `llvm-ar` from the active toolchain's `llvm-tools` component.
///
/// The first choice, because it needs nothing installed beyond the toolchain
/// this build is already running under, and `make setup-rust` installs that
/// component for exactly this (and for the `lld-link`/`llvm-lib` symlinks it
/// creates from the same directory). `None` when the component is absent, which
/// is what a toolchain installed without it looks like.
fn toolchain_llvm_ar() -> Option<String> {
    let rustc = std::env::var("RUSTC").ok()?;
    let host = std::env::var("HOST").ok()?;
    let out = std::process::Command::new(rustc)
        .args(["--print", "sysroot"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sysroot = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    let path = format!("{sysroot}/lib/rustlib/{host}/bin/llvm-ar");
    std::path::Path::new(&path).is_file().then_some(path)
}

/// `llvm-ar` from the Homebrew llvm keg, which is keg-only and off PATH.
///
/// The fallback for a toolchain with no `llvm-tools`. Asking `brew` covers both
/// prefixes (Apple-Silicon `/opt/homebrew`, Intel `/usr/local`); the literal
/// path is the last resort when `brew` itself is not on PATH.
fn homebrew_llvm_ar() -> String {
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
