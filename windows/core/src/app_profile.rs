//! Built-in per-application configuration profiles.
//!
//! A few games need an option set no user should have to discover: an adapter
//! identity their engine branches on, a capability they must not be told about,
//! an era-driver behaviour their renderer was written against. A profile carries
//! those values for one executable so the game runs correctly out of the box.
//!
//! A profile is the weakest configuration layer. `mtld3d.conf` and
//! `MTLD3D_CONFIG` both beat it key by key, so a user can take one option back
//! without losing the rest of the profile.
//!
//! A profile is keyed by the version resource the vendor linked into the image,
//! not by a path, so it follows the game wherever it is installed and cannot
//! catch an unrelated program that happens to share a file name.
//!
//! Pure logic: the PE side supplies the executable's name and the raw
//! `VS_VERSIONINFO` blob, so everything here is host-testable.

use log::info;

use super::LOG_TARGET;

/// The profiles mtld3d ships.
///
/// Every entry pins at least one version-resource field, so a rule can never
/// fire on a same-named program from someone else.
static PROFILES: &[AppProfile] = &[
    // Grand Theft Auto IV. Its renderer branches on the reported adapter
    // vendor, and the ATI identity is the one whose paths it completes; the
    // NVIDIA identity stalls in the game's own identifier parsing. It also
    // picks a mixed DF24 plus INTZ depth path when the DF fourccs are
    // advertised, which no hardware of its era offered together, so the depth
    // formats stay hidden. Its occlusion culling needs real pixel counts
    // rather than an immediate answer, otherwise every query reads as fully
    // visible. And its late alpha, sky and glow passes z-test one INTZ depth
    // texture against scene depth rendered into a same-size sibling, which
    // only works where equal-size depth surfaces share one allocation.
    AppProfile {
        name: "gta-iv",
        exe: "GTAIV.exe",
        company: Some("Rockstar Games"),
        product: Some("Grand Theft Auto IV"),
        original_filename: None,
        settings: "adapter.spoof=amd;caps.dfFormats=false;\
                   query.flushImmediate=false;depth.aliasSameSize=true",
    },
];

/// One built-in profile: what it matches, and the options it sets.
///
/// The options are held in `MTLD3D_CONFIG` syntax rather than as typed fields,
/// so a profile flows through the same per-entry decode as a user's file and
/// every present and future key is available to one without new code.
pub struct AppProfile {
    name: &'static str,
    exe: &'static str,
    company: Option<&'static str>,
    product: Option<&'static str>,
    original_filename: Option<&'static str>,
    settings: &'static str,
}

impl AppProfile {
    /// The profile's name, as the log prints it.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// The profile's options, semicolon-separated `key=value` entries.
    #[must_use]
    pub const fn settings(&self) -> &'static str {
        self.settings
    }

    /// Whether this profile applies to `id`.
    ///
    /// The executable name has to match, case-insensitively. Each version field
    /// the profile pins is an additional case-insensitive substring test, so
    /// `Rockstar Games` also matches a `CompanyName` of `Rockstar Games, Inc.`.
    fn matches(&self, id: &AppIdentity) -> bool {
        id.exe.eq_ignore_ascii_case(self.exe)
            && field_matches(self.company, &id.company)
            && field_matches(self.product, &id.product)
            && field_matches(self.original_filename, &id.original_filename)
    }
}

/// Who the running application is.
///
/// The executable's file name, plus the three `VS_VERSIONINFO` strings a vendor
/// sets at link time. The version strings are empty when the image carries no
/// version resource, which leaves it matchable by name alone.
pub struct AppIdentity {
    exe: String,
    company: String,
    product: String,
    original_filename: String,
}

impl AppIdentity {
    /// Build an identity from the executable's name and its version resource.
    ///
    /// `version_blob` is the raw `RT_VERSION` resource as UTF-16 units, or
    /// `None` for an image that has none.
    #[must_use]
    pub fn new(exe: String, version_blob: Option<&[u16]>) -> Self {
        let runs = version_blob.map(utf16_runs).unwrap_or_default();
        Self {
            exe,
            company: value_after(&runs, "CompanyName"),
            product: value_after(&runs, "ProductName"),
            original_filename: value_after(&runs, "OriginalFilename"),
        }
    }
}

/// The built-in profile for this application, if it has one.
///
/// Logs the outcome either way: which profile took effect, or that none did.
/// Both answers are the first thing to establish when a game behaves
/// differently than its options say it should.
#[must_use]
pub fn lookup(id: &AppIdentity) -> Option<&'static AppProfile> {
    let profile = PROFILES.iter().find(|p| p.matches(id));
    match profile {
        Some(p) => info!(
            target: LOG_TARGET,
            "app profile: {} matched {} ({}, {})", p.name, id.exe, id.company, id.product
        ),
        None => info!(target: LOG_TARGET, "app profile: none for {}", id.exe),
    }
    profile
}

/// Whether an optional substring constraint holds, case-insensitively.
///
/// A profile that leaves the field unset places no constraint at all.
fn field_matches(constraint: Option<&str>, actual: &str) -> bool {
    constraint.is_none_or(|want| {
        actual
            .to_ascii_lowercase()
            .contains(&want.to_ascii_lowercase())
    })
}

/// The value a `VS_VERSIONINFO` string entry carries under `key`.
///
/// A `String` entry stores a NUL-terminated key immediately followed, after
/// padding, by its value, so the value is the run after the key's. The key is
/// matched as a run *suffix* because each entry starts with three header words
/// (`wLength`, `wValueLength`, `wType`) that sit right against the key text and
/// decode as part of that run; the value that follows has no such header and so
/// is a clean run.
fn value_after(runs: &[String], key: &str) -> String {
    runs.iter()
        .position(|run| run.ends_with(key))
        .and_then(|i| runs.get(i + 1))
        .cloned()
        .unwrap_or_default()
}

/// Split a UTF-16 buffer into its NUL-separated runs, dropping empty ones.
///
/// Decoding is lossy: the blob is vendor data, and one bad unit in an unrelated
/// entry must not cost us the fields we came for.
fn utf16_runs(blob: &[u16]) -> Vec<String> {
    blob.split(|&unit| unit == 0)
        .filter(|run| !run.is_empty())
        .map(String::from_utf16_lossy)
        .collect()
}

#[cfg(test)]
mod tests;
