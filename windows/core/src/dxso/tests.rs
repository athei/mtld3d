use std::{
    fs,
    path::{Path, PathBuf},
};

/// Module paths an emitter source may name, and where their source files live.
const MODULE_ROOTS: [(&str, &str); 2] = [
    ("crate::", "src/"),
    ("mtld3d_shared::", "../../unix/shared/src/"),
];

/// The emitter sources `build.rs` hashes: `src/dxso.rs` and everything under `src/dxso`.
fn emitter_sources(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).expect("read emitter source directory") {
            let path = entry.expect("emitter source entry").path();
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("source name");
            if stem == "tests" || stem.ends_with("_tests") {
                continue;
            }
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }
    let mut out = vec![root.join("src/dxso.rs")];
    walk(&root.join("src/dxso"), &mut out);
    out
}

/// Every module outside `dxso` that the emitter names is in the emitter fingerprint.
///
/// A cached MSL library is reused while the fingerprint is unchanged, so a
/// value the emitter writes into MSL from a source `build.rs` does not hash
/// leaves stale libraries in every cache when only that source changes.
/// `mtld3d_types` is not checked: it holds D3D9 ABI constants, whose values
/// are fixed.
#[test]
fn every_module_the_emitter_names_is_fingerprinted() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let build = fs::read_to_string(root.join("build.rs")).expect("read build.rs");
    let mut missing = Vec::new();
    for path in emitter_sources(root) {
        let text = fs::read_to_string(&path).expect("read emitter source");
        let code = text
            .lines()
            .map(str::trim_start)
            .filter(|line| !line.starts_with("//"));
        for line in code {
            for (prefix, dir) in MODULE_ROOTS {
                for (at, _) in line.match_indices(prefix) {
                    let rest = &line[at + prefix.len()..];
                    assert!(
                        !rest.starts_with('{'),
                        "{}: name each `{prefix}` module on its own path so this check sees it",
                        path.display()
                    );
                    let module: String = rest
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    // A macro or a root item names no module source.
                    if module == "dxso" || !rest[module.len()..].starts_with("::") {
                        continue;
                    }
                    if !build.contains(&format!("\"{dir}{module}.rs\"")) {
                        missing.push(format!("{}: {prefix}{module}", path.display()));
                    }
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "emitter sources name modules the fingerprint does not hash: {missing:#?}"
    );
}
