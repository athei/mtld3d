//! Command-line argument parsing for the conformance runner.

use std::path::PathBuf;

use crate::model::{Arch, Subtest};

/// Parsed invocation options.
#[derive(Debug)]
pub struct Config {
    /// `--update-baseline`: record a fresh baseline instead of diffing.
    pub update: bool,
    /// `--wine-build`: the Wine build tree holding `d3d9_test.exe` per arch.
    pub wine_build: PathBuf,
    /// `--assets`: the directory holding `baseline.txt` / `CONFORMANCE.md`.
    ///
    /// `None` means "use the crate directory" (resolved by the caller).
    pub assets: Option<PathBuf>,
    /// `--only <subtest>`: restrict the run to one subtest (`None` = all four).
    pub only: Option<Subtest>,
    /// `--arch <arch>`: restrict the run to one PE arch (`None` = both).
    pub arch: Option<Arch>,
    /// `--repeat <N>`: run each selected `(arch, subtest)` N times.
    ///
    /// Prints a flap report instead of diffing. `1` (the default) keeps the
    /// normal gate.
    pub repeat: u32,
}

/// Parse CLI args (excluding `argv[0]`).
///
/// Recognised flags: `--update-baseline`, `--wine-build <path>`,
/// `--assets <dir>`, `--only <subtest>`, `--arch <arch>`, `--repeat <N>`.
/// `--wine-build` falls back to `wine_build_env` (the `WINE_BUILD` env var) when
/// the flag is absent. `--assets` is optional and defaults to the crate
/// directory. `--only`/`--arch`/`--repeat>1` are mutually exclusive with
/// `--update-baseline` (a filtered re-baseline would drop the unselected
/// `(arch, subtest)` entries from `baseline.txt`).
///
/// # Errors
///
/// Returns a message on an unknown flag, a flag missing or mis-parsing its
/// value, no `--wine-build` and no env fallback, or a filter combined with
/// `--update-baseline`.
pub fn parse_args(
    mut args: impl Iterator<Item = String>,
    wine_build_env: Option<String>,
) -> Result<Config, String> {
    let mut update = false;
    let mut wine_build: Option<PathBuf> = None;
    let mut assets: Option<PathBuf> = None;
    let mut only: Option<Subtest> = None;
    let mut arch: Option<Arch> = None;
    let mut repeat: u32 = 1;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--update-baseline" => update = true,
            "--wine-build" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--wine-build needs a path".to_owned())?;
                wine_build = Some(PathBuf::from(value));
            }
            "--assets" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--assets needs a path".to_owned())?;
                assets = Some(PathBuf::from(value));
            }
            "--only" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--only needs a subtest".to_owned())?;
                only = Some(value.parse::<Subtest>()?);
            }
            "--arch" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--arch needs an arch".to_owned())?;
                arch = Some(value.parse::<Arch>()?);
            }
            "--repeat" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--repeat needs a count".to_owned())?;
                repeat = value
                    .parse::<u32>()
                    .map_err(|_| format!("--repeat not an integer: {value:?}"))?;
                if repeat == 0 {
                    return Err("--repeat must be >= 1".to_owned());
                }
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if update && (only.is_some() || arch.is_some() || repeat > 1) {
        return Err(
            "--update-baseline cannot be combined with --only/--arch/--repeat \
             (a filtered re-baseline would drop the unselected entries)"
                .to_owned(),
        );
    }
    let wine_build = wine_build
        .or_else(|| wine_build_env.map(PathBuf::from))
        .ok_or_else(|| "missing --wine-build (or WINE_BUILD env)".to_owned())?;
    Ok(Config {
        update,
        wine_build,
        assets,
        only,
        arch,
        repeat,
    })
}

#[cfg(test)]
mod tests {
    use super::parse_args;
    use crate::model::{Arch, Subtest};

    fn args(tokens: &[&str]) -> std::vec::IntoIter<String> {
        tokens
            .iter()
            .map(|t| (*t).to_owned())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn parses_flags_and_update() {
        let config = parse_args(
            args(&["--update-baseline", "--wine-build", "/wb", "--assets", "/a"]),
            None,
        )
        .unwrap();
        assert!(config.update);
        assert_eq!(config.wine_build.to_str(), Some("/wb"));
        assert_eq!(
            config.assets.as_deref().and_then(std::path::Path::to_str),
            Some("/a")
        );
    }

    #[test]
    fn wine_build_falls_back_to_env() {
        let config = parse_args(args(&["--assets", "/a"]), Some("/env-wb".to_owned())).unwrap();
        assert_eq!(config.wine_build.to_str(), Some("/env-wb"));
        assert!(!config.update);
    }

    #[test]
    fn assets_is_none_when_absent() {
        let config = parse_args(args(&["--wine-build", "/wb"]), None).unwrap();
        assert!(config.assets.is_none());
    }

    #[test]
    fn missing_wine_build_errors() {
        let err = parse_args(args(&["--update-baseline"]), None).unwrap_err();
        assert!(err.contains("wine-build"), "{err}");
    }

    #[test]
    fn unknown_flag_errors() {
        let err = parse_args(args(&["--nope"]), None).unwrap_err();
        assert!(err.contains("unknown argument"), "{err}");
    }

    #[test]
    fn filters_default_to_unset_and_repeat_one() {
        let config = parse_args(args(&["--wine-build", "/wb"]), None).unwrap();
        assert!(config.only.is_none());
        assert!(config.arch.is_none());
        assert_eq!(config.repeat, 1);
    }

    #[test]
    fn parses_only_arch_and_repeat() {
        let config = parse_args(
            args(&[
                "--wine-build",
                "/wb",
                "--only",
                "device",
                "--arch",
                "i686",
                "--repeat",
                "20",
            ]),
            None,
        )
        .unwrap();
        assert_eq!(config.only, Some(Subtest::Device));
        assert_eq!(config.arch, Some(Arch::I686));
        assert_eq!(config.repeat, 20);
    }

    #[test]
    fn bad_subtest_and_arch_error() {
        let err = parse_args(args(&["--wine-build", "/wb", "--only", "nope"]), None).unwrap_err();
        assert!(err.contains("unknown subtest"), "{err}");
        let err = parse_args(args(&["--wine-build", "/wb", "--arch", "arm"]), None).unwrap_err();
        assert!(err.contains("unknown arch"), "{err}");
    }

    #[test]
    fn repeat_zero_errors() {
        let err = parse_args(args(&["--wine-build", "/wb", "--repeat", "0"]), None).unwrap_err();
        assert!(err.contains(">= 1"), "{err}");
    }

    #[test]
    fn update_baseline_rejects_filters() {
        let err = parse_args(
            args(&[
                "--update-baseline",
                "--wine-build",
                "/wb",
                "--only",
                "device",
            ]),
            None,
        )
        .unwrap_err();
        assert!(err.contains("cannot be combined"), "{err}");
    }
}
