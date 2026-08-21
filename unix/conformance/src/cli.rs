//! Command-line argument parsing for the conformance runner.

use std::path::PathBuf;

use crate::model::{Arch, Subtest};

/// Parsed invocation options.
#[derive(Debug)]
pub struct Config {
    /// `--update-baseline`: record a fresh baseline instead of diffing.
    pub update: bool,
    /// `--wine`: the Wine loader to spawn the test binary with.
    pub wine: PathBuf,
    /// `--exe`: the `d3d9_test.exe` to run.
    pub exe: PathBuf,
    /// `--arch`: which architecture `exe` is, i.e. the label results record under.
    pub arch: Arch,
    /// `--assets`: the directory holding `baseline.txt` / `CONFORMANCE.md`.
    ///
    /// `None` means "use the crate directory" (resolved by the caller).
    pub assets: Option<PathBuf>,
    /// `--only <subtest>`: restrict the run to one subtest (`None` = all four).
    pub only: Option<Subtest>,
    /// `--repeat <N>`: run each selected subtest N times.
    ///
    /// Prints a flap report instead of diffing. `1` (the default) keeps the
    /// normal gate.
    pub repeat: u32,
}

/// Parse CLI args (excluding `argv[0]`).
///
/// Recognised flags: `--update-baseline`, `--wine <path>`, `--exe <path>`,
/// `--arch <arch>`, `--assets <dir>`, `--only <subtest>`, `--repeat <N>`.
/// `--wine`, `--exe` and `--arch` are mandatory: the runner resolves no paths of
/// its own, so the caller (the Makefile) owns every Wine location. One
/// invocation runs one test binary, which is what lets a 32-bit and a 64-bit CI
/// job be two independent processes. `--assets` defaults to the crate
/// directory. `--only`/`--repeat>1` are mutually exclusive with
/// `--update-baseline` (a filtered re-baseline would drop the unselected
/// subtests from `baseline.txt`).
///
/// # Errors
///
/// Returns a message on an unknown flag, a flag missing or mis-parsing its
/// value, a missing mandatory flag, or a filter combined with
/// `--update-baseline`.
pub fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Config, String> {
    let mut update = false;
    let mut wine: Option<PathBuf> = None;
    let mut exe: Option<PathBuf> = None;
    let mut arch: Option<Arch> = None;
    let mut assets: Option<PathBuf> = None;
    let mut only: Option<Subtest> = None;
    let mut repeat: u32 = 1;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--update-baseline" => update = true,
            "--wine" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--wine needs a path".to_owned())?;
                wine = Some(PathBuf::from(value));
            }
            "--exe" => {
                let value = args.next().ok_or_else(|| "--exe needs a path".to_owned())?;
                exe = Some(PathBuf::from(value));
            }
            "--arch" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--arch needs an arch".to_owned())?;
                arch = Some(value.parse::<Arch>()?);
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
    if update && (only.is_some() || repeat > 1) {
        return Err("--update-baseline cannot be combined with --only/--repeat \
             (a filtered re-baseline would drop the unselected entries)"
            .to_owned());
    }
    let wine = wine.ok_or_else(|| "missing --wine <path to the wine loader>".to_owned())?;
    let exe = exe.ok_or_else(|| "missing --exe <path to d3d9_test.exe>".to_owned())?;
    let arch = arch.ok_or_else(|| "missing --arch <i686|x86_64>".to_owned())?;
    Ok(Config {
        update,
        wine,
        exe,
        arch,
        assets,
        only,
        repeat,
    })
}

#[cfg(test)]
mod tests;
