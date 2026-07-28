use std::{path::PathBuf, process::Command};

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

    // Stamp the release identity into every cdylib that links this crate, so a
    // captured log names the release it came from. The exact binary is named by
    // the linker-assigned image ID the same log line carries (`crate::image`);
    // this half only has to say which release the source is from.
    println!("cargo:rustc-env=MTLD3D_BUILD={}", build_id());
}

/// Release identity: `git describe`, or the manifest version outside a checkout.
///
/// Deliberately no `--dirty`. Keeping that flag honest would mean watching every
/// crate's sources from here, and this crate sits upstream of all three cdylibs,
/// so each source edit would rebuild the whole tree. The image ID in the same log
/// line already changes whenever the binary's contents change, which is what
/// `--dirty` was approximating.
///
/// Falls back to the manifest version rather than `unknown` so a build from an
/// exported source tree still names a version.
fn build_id() -> String {
    // Tags matter as much as commits: cutting a release changes the identity
    // without touching a single source file or moving HEAD, and a release
    // artifact stamped with the previous version is the one mistake this line
    // exists to prevent.
    if let Some(dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        let dir = PathBuf::from(dir);
        let mut watch = vec![
            dir.join("HEAD"),
            dir.join("packed-refs"),
            dir.join("refs").join("tags"),
        ];
        if let Some(head_ref) = git(&["symbolic-ref", "--quiet", "HEAD"]) {
            watch.push(dir.join(head_ref));
        }
        for path in watch.iter().filter(|p| p.exists()) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    git(&["describe", "--tags", "--always"])
        .unwrap_or_else(|| format!("v{}", env!("CARGO_PKG_VERSION")))
}

/// Run `git` in the manifest directory, returning trimmed stdout on success.
///
/// Any failure (no git, no checkout, non-zero exit, empty output) yields `None`
/// so the caller can fall back.
fn git(args: &[&str]) -> Option<String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")?;
    Command::new("git")
        .current_dir(manifest_dir)
        .args(args)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}
