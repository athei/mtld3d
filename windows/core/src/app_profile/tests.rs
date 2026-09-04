use super::{AppIdentity, AppProfile, PROFILES, lookup};
use crate::config::{AdapterSpoof, CursorScale, parse};

/// The three fields read out of the shipped `GTAIV.exe`.
const GTA_IV: [(&str, &str); 3] = [
    ("CompanyName", "Rockstar Games"),
    ("ProductName", "Grand Theft Auto IV"),
    ("OriginalFilename", "GTAIV.exe"),
];

/// Encode string entries the way a real `VS_VERSIONINFO` holds them.
///
/// Each entry is preceded by its three header words (`wLength`,
/// `wValueLength`, `wType`), which sit directly against the key text with no
/// separator, so the key decodes as part of that run. Reproducing that here is
/// the point of the fixture: a decoder that looks for an exact `CompanyName`
/// run finds nothing in a real image.
fn blob(entries: &[(&str, &str)]) -> Vec<u16> {
    let mut out = Vec::new();
    for (key, value) in entries {
        out.extend_from_slice(&[0x0040, 0x000f, 0x0001]);
        out.extend(key.encode_utf16());
        out.push(0);
        out.extend(value.encode_utf16());
        out.push(0);
    }
    out
}

#[test]
fn the_version_fields_decode_past_the_entry_headers() {
    let id = AppIdentity::new("GTAIV.exe".to_owned(), Some(&blob(&GTA_IV)));
    assert_eq!(id.company, "Rockstar Games");
    assert_eq!(id.product, "Grand Theft Auto IV");
    assert_eq!(id.original_filename, "GTAIV.exe");
}

#[test]
fn an_image_without_a_version_resource_has_empty_fields() {
    let id = AppIdentity::new("GTAIV.exe".to_owned(), None);
    assert!(id.company.is_empty());
    assert!(id.product.is_empty());
    assert!(id.original_filename.is_empty());
}

#[test]
fn the_gta_iv_profile_matches_the_real_binarys_resource() {
    let id = AppIdentity::new("GTAIV.exe".to_owned(), Some(&blob(&GTA_IV)));
    assert_eq!(lookup(&id).map(AppProfile::name), Some("gta-iv"));
    // The name is matched case-insensitively: a launcher may spell it either way.
    let shouting = AppIdentity::new("gtaiv.exe".to_owned(), Some(&blob(&GTA_IV)));
    assert_eq!(lookup(&shouting).map(AppProfile::name), Some("gta-iv"));
}

#[test]
fn the_wow_profile_matches_both_clients_and_keeps_the_immediate_answer() {
    for version in ["Version 1.12", "Version 3.3"] {
        let blob = blob(&[
            ("CompanyName", "Blizzard Entertainment"),
            ("ProductName", "World of Warcraft"),
            ("ProductVersion", version),
        ]);
        let id = AppIdentity::new("WoW.exe".to_owned(), Some(&blob));
        let profile = lookup(&id).expect("the wow profile matches");
        assert_eq!(profile.name(), "wow");
        assert!(!parse(None, "", None).query_flush_immediate);
        assert!(parse(Some(profile), "", None).query_flush_immediate);
    }
    let nameless = AppIdentity::new("WoW.exe".to_owned(), None);
    assert!(lookup(&nameless).is_none());
}

#[test]
fn a_pinned_field_is_a_substring_test() {
    let suffixed = blob(&[
        ("CompanyName", "Rockstar Games, Inc."),
        ("ProductName", "Grand Theft Auto IV Complete Edition"),
    ]);
    let id = AppIdentity::new("GTAIV.exe".to_owned(), Some(&suffixed));
    assert_eq!(lookup(&id).map(AppProfile::name), Some("gta-iv"));
}

#[test]
fn an_unrelated_program_of_the_same_name_is_left_alone() {
    let nameless = AppIdentity::new("GTAIV.exe".to_owned(), None);
    assert!(lookup(&nameless).is_none());
    let other = blob(&[
        ("CompanyName", "Somebody Else"),
        ("ProductName", "Grand Theft Auto IV"),
    ]);
    let id = AppIdentity::new("GTAIV.exe".to_owned(), Some(&other));
    assert!(lookup(&id).is_none());
}

#[test]
fn an_unprofiled_game_resolves_to_nothing() {
    let id = AppIdentity::new(
        "Morrowind.exe".to_owned(),
        Some(&blob(&[("CompanyName", "Bethesda Softworks")])),
    );
    assert!(lookup(&id).is_none());
}

#[test]
fn every_profile_is_uniquely_named_and_pins_a_version_field() {
    for profile in PROFILES {
        assert!(!profile.name.is_empty(), "a profile has no name");
        assert!(!profile.exe.is_empty(), "{} has no exe", profile.name);
        assert!(
            profile.company.is_some()
                || profile.product.is_some()
                || profile.original_filename.is_some(),
            "{} matches on its executable alone",
            profile.name
        );
    }
    let mut names: Vec<&str> = PROFILES.iter().map(|p| p.name).collect();
    names.sort_unstable();
    let count = names.len();
    names.dedup();
    assert_eq!(names.len(), count, "duplicate profile name");
}

#[test]
fn no_two_profiles_can_match_the_same_application() {
    for (i, profile) in PROFILES.iter().enumerate() {
        // The widest identity this profile accepts: its own pins, nothing else.
        let mut entries: Vec<(&str, &str)> = Vec::new();
        if let Some(company) = profile.company {
            entries.push(("CompanyName", company));
        }
        if let Some(product) = profile.product {
            entries.push(("ProductName", product));
        }
        if let Some(original) = profile.original_filename {
            entries.push(("OriginalFilename", original));
        }
        let id = AppIdentity::new(profile.exe.to_owned(), Some(&blob(&entries)));
        for (j, other) in PROFILES.iter().enumerate() {
            assert!(
                i == j || !other.matches(&id),
                "{} and {} both match one application",
                profile.name,
                other.name
            );
        }
    }
}

#[test]
fn the_gta_iv_profile_resolves_to_the_options_it_names() {
    let id = AppIdentity::new("GTAIV.exe".to_owned(), Some(&blob(&GTA_IV)));
    let profile = lookup(&id).expect("the profile matches its own fixture");
    let cfg = parse(Some(profile), "", None);
    // A typo in a profile's settings string would leave every default in place,
    // so these four assertions are what pins the key spellings.
    assert_eq!(cfg.adapter_spoof, AdapterSpoof::Amd);
    assert!(!cfg.df_formats);
    assert!(!cfg.query_flush_immediate);
    assert!(cfg.depth_alias_same_size);
}

/// A profile that sets two unrelated options, for the precedence cases.
const PROBE: AppProfile = AppProfile {
    name: "probe",
    exe: "probe.exe",
    company: Some("Probe"),
    product: None,
    original_filename: None,
    settings: "cursor.scale=4;present.maxFps=30",
};

#[test]
fn a_profile_beats_the_defaults() {
    let cfg = parse(Some(&PROBE), "", None);
    assert_eq!(cfg.cursor_scale, CursorScale::Fixed(4));
    assert_eq!(cfg.present_max_fps, 30);
}

#[test]
fn the_file_beats_a_profile_key_by_key() {
    let cfg = parse(Some(&PROBE), "cursor.scale = 2\n", None);
    assert_eq!(cfg.cursor_scale, CursorScale::Fixed(2));
    // The key the file said nothing about keeps the profile's value.
    assert_eq!(cfg.present_max_fps, 30);
}

#[test]
fn the_env_beats_both() {
    let cfg = parse(
        Some(&PROBE),
        "cursor.scale = 2\n",
        Some("cursor.scale=8;present.maxFps=0"),
    );
    assert_eq!(cfg.cursor_scale, CursorScale::Fixed(8));
    assert_eq!(cfg.present_max_fps, 0);
}

#[test]
fn a_malformed_profile_entry_does_not_derail_the_rest() {
    let broken = AppProfile {
        settings: "bogusKey=whatever;no equals sign;present.maxFps=45",
        ..PROBE
    };
    let cfg = parse(Some(&broken), "", None);
    assert_eq!(cfg.present_max_fps, 45);
}
