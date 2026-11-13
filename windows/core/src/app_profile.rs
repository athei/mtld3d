//! Built-in per-application configuration profiles.
//!
//! A few games need an option set no user should have to discover: an adapter
//! identity their engine branches on, a capability they must not be told about,
//! an era-driver behaviour their renderer was written against. A profile carries
//! those values for the executables of one game, its launcher included where
//! the launcher loads D3D9 too, so the game runs correctly out of the box.
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
    // Fallout 3.
    //
    // Its renderer turns "3.0 Lighting" on only for an adapter whose `Driver`
    // string starts with `ati` or `nv`, and only when
    // `PS20Caps.NumInstructionSlots` is above 255; the game counts the slots
    // as 96 when `PS20Caps.NumTemps` is under 32. The layer reports 512 and
    // 32, so a caps change past either threshold turns the flag off again.
    // The game creates its type 0x1d shader properties whether the flag is on
    // or not, and the shader lookup for that type returns null while it is
    // off, which the game dereferences: New Game crashes without an `ati` or
    // `nv` prefix.
    //
    // Either prefix sets the flag, and each runs a multi-GPU probe. The `ati`
    // one calls `AtiQueryMgpuCount` in the `atimgpud.dll` the game ships, so
    // it always runs. The `nv` one goes through NVAPI (`nvapi_QueryInterface`
    // in `nvapi.dll`) and has no effect when the DLL is absent, so the
    // identity is NVIDIA's. With it the game also probes the `ATOC` format
    // and, when transparency multisampling is on, draws its alpha-tested
    // geometry with alpha-to-coverage instead, which the layer implements.
    //
    // `Fallout3.exe` exits into `FalloutLauncher.exe` whenever the
    // `sD3DDevice` that `FalloutPrefs.ini` holds differs from adapter 0's
    // `Description`, and the launcher writes that key from its own D3D9
    // adapter, so both executables need the same identity or each sends the
    // player back to the other.
    AppProfile {
        name: "fallout3",
        exes: &["Fallout3.exe", "FalloutLauncher.exe"],
        company: Some("Bethesda Softworks"),
        product: Some("Fallout 3"),
        original_filename: None,
        settings: "adapter.spoof=nvidia",
    },
    // Fallout: New Vegas.
    //
    // New Game crashes after the intro video unless the adapter reports an
    // NVIDIA identity. The launcher gets the same identity on the assumption
    // that New Vegas keeps the `sD3DDevice` check the `fallout3` entry
    // describes, so neither executable sends the player back to the other.
    AppProfile {
        name: "falloutnv",
        exes: &["FalloutNV.exe", "FalloutNVLauncher.exe"],
        company: Some("Bethesda Softworks"),
        product: Some("New Vegas"),
        original_filename: None,
        settings: "adapter.spoof=nvidia",
    },
    // Grand Theft Auto IV.
    //
    // Its renderer branches on the reported adapter vendor, and the ATI
    // identity is the one whose paths it completes; the NVIDIA identity stalls
    // in the game's own identifier parsing. It also picks a mixed DF24 plus
    // INTZ depth path when the DF fourccs are advertised, which no hardware of
    // its era offered together, so the depth formats stay hidden. It keeps
    // `query.flushImmediate` off: its occlusion culling reads the counts its
    // FLUSH polls return, and an immediate answer reports every pending query
    // as fully visible. And its late alpha, sky and glow passes z-test one
    // INTZ depth texture against scene depth rendered into a same-size
    // sibling, which only works where equal-size depth surfaces share one
    // allocation.
    AppProfile {
        name: "gta-iv",
        exes: &["GTAIV.exe"],
        company: Some("Rockstar Games"),
        product: Some("Grand Theft Auto IV"),
        original_filename: None,
        settings: "adapter.spoof=amd;caps.dfFormats=false;depth.aliasSameSize=true",
    },
    // Halo 2.
    //
    // It locks a window of its shared world index buffer and writes past the
    // size that lock announced. A `Staged` buffer uploads only the announced
    // range, so the world draws index into zero-filled storage and draw
    // nothing.
    AppProfile {
        name: "halo2",
        exes: &["halo2.exe"],
        company: Some("Microsoft Corporation"),
        product: Some("Halo 2 for Windows Vista"),
        original_filename: Some("Halo2.exe"),
        settings: "buffer.ignoreLockBounds=true",
    },
    // The Elder Scrolls IV: Oblivion.
    //
    // Its renderer takes its shader level from the prefix of the adapter's
    // `Driver` string. Without an `ati` or `nv` prefix it stays on the basic
    // 2.0 level whatever the caps say, and the Video menu greys Shadow
    // Filtering and Shadows on Grass below the level `nv` reaches. The prefix
    // needs the caps as well: `nv` reaches that level only when
    // `PS20Caps.NumInstructionSlots` is above 96, and the game counts the
    // slots as 96 when `PS20Caps.NumTemps` is under 32. The layer reports 512
    // and 32, so a caps change past either threshold greys both options again.
    //
    // The identity is NVIDIA's because `nv` is the only prefix that keeps
    // refraction under multisampling, which this profile therefore turns on.
    // Each identity runs a multi-GPU probe. The `nv` one calls `NvCplGetDataInt`
    // in `NVCPL.dll` and has no effect when the DLL is absent. The `ati` one
    // calls `AtiQueryMgpuCount`, which goes through `ExtEscape`, in the
    // `ATIMGPUD.dll` the game ships, so it always runs.
    AppProfile {
        name: "oblivion",
        exes: &["Oblivion.exe"],
        company: Some("Bethesda Softworks"),
        product: Some("TES4: Oblivion"),
        original_filename: None,
        settings: "adapter.spoof=nvidia",
    },
    // World of Warcraft, the 1.12 and 3.3.5 clients alike.
    //
    // Both poll `GetData(D3DGETDATA_FLUSH)` after every loading-screen upload
    // batch but never read the count, and skipping the spec-correct wait was
    // measured to save seconds per load. This profile assumes the polls do not
    // gate reuse of CPU-writable dynamic storage. Metal cannot track a CPU
    // write through `D3DLOCK_NOOVERWRITE` into pages a queued draw still
    // reads. The sun and moon lens flares read their queries without FLUSH and
    // still get the real counts.
    //
    // `gxFixLag` ("Reduce Input Lag", on by default) polls an EVENT query every
    // frame to keep the CPU from running ahead of the GPU. Answering it from
    // GPU retirement serialises CPU and GPU work and costs at least a third of
    // the frame rate, while the encoder and submit threads already bound how
    // far ahead the CPU can get. As with FLUSH, this profile assumes the polls
    // do not gate reuse of CPU-writable dynamic storage, which an early answer
    // cannot fence.
    //
    // The 1.12 client fills its resolution dropdown from `EnumDisplaySettingsA`,
    // not `EnumAdapterModes`: it keeps every mode of at least 640x480 at 16
    // bits or more, sorts them, and lists each size of at least 800x600 and
    // an aspect of at least 1.248 once, whatever its depths and refresh rates.
    // The dropdown holds 32 buttons (40 on Turtle WoW's interface), and
    // opening Video Options with more sizes than that fails with a Lua error.
    // Wine's emulated mode list runs to forty-odd sizes, so the profile offers
    // at most 32; the refresh rates of each kept size do not add buttons.
    AppProfile {
        name: "wow",
        exes: &["WoW.exe"],
        company: Some("Blizzard Entertainment"),
        product: Some("World of Warcraft"),
        original_filename: None,
        settings: "query.flushImmediate=true;query.eventImmediate=true;display.maxResolutions=32",
    },
];

/// One built-in profile: what it matches, and the options it sets.
///
/// The options are held in `MTLD3D_CONFIG` syntax rather than as typed fields,
/// so a profile flows through the same per-entry decode as a user's file and
/// every present and future key is available to one without new code.
pub struct AppProfile {
    name: &'static str,
    exes: &'static [&'static str],
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
    /// The executable name has to equal one of the profile's names,
    /// case-insensitively. Each version field the profile pins is an additional
    /// case-insensitive substring test, so `Rockstar Games` also matches a
    /// `CompanyName` of `Rockstar Games, Inc.`. The pins hold for every listed
    /// executable alike.
    fn matches(&self, id: &AppIdentity) -> bool {
        self.exes.iter().any(|exe| id.exe.eq_ignore_ascii_case(exe))
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
/// Logs the outcome either way: which profile took effect and the options it
/// sets, or that none did. Both answers are the first thing to establish when a
/// game behaves differently than its options say it should. A miss also logs
/// the image's company and product, so a new profile can be keyed from a
/// user's log.
#[must_use]
pub fn lookup(id: &AppIdentity) -> Option<&'static AppProfile> {
    let profile = PROFILES.iter().find(|p| p.matches(id));
    match profile {
        Some(p) => info!(
            target: LOG_TARGET,
            "app profile: {} matched {} ({}, {}): {}",
            p.name, id.exe, id.company, id.product, p.settings
        ),
        None => info!(
            target: LOG_TARGET,
            "app profile: none for {} ({:?}, {:?})",
            id.exe, id.company, id.product
        ),
    }
    profile
}

/// The built-in profile called `name`, whatever application it matches.
///
/// For a caller that runs a profile's options without being its
/// application, such as a benchmark that must follow what the profile sets.
#[must_use]
pub fn builtin(name: &str) -> Option<&'static AppProfile> {
    PROFILES.iter().find(|p| p.name == name)
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
