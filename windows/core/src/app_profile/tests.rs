use super::{AppIdentity, AppProfile, PROFILES, builtin, lookup, steam_app_id};
use crate::config::{AdapterSpoof, CursorScale, parse};

/// The three fields read out of the shipped `GTAIV.exe`.
const GTA_IV: [(&str, &str); 3] = [
    ("CompanyName", "Rockstar Games"),
    ("ProductName", "Grand Theft Auto IV"),
    ("OriginalFilename", "GTAIV.exe"),
];

/// The three fields read out of the shipped `Oblivion.exe`.
const OBLIVION: [(&str, &str); 3] = [
    ("CompanyName", "Bethesda Softworks"),
    ("ProductName", "TES4: Oblivion"),
    ("OriginalFilename", "Oblivion.exe"),
];

/// The string table of the shipped `Fallout3.exe`.
const FALLOUT3: [(&str, &str); 7] = [
    ("CompanyName", "Bethesda Softworks"),
    ("FileDescription", "Fallout3"),
    ("FileVersion", "1.7.0.3"),
    ("InternalName", "Fallout3"),
    ("OriginalFilename", "Fallout3.exe"),
    ("ProductName", "Fallout 3"),
    ("ProductVersion", "1.7.0.3"),
];

/// The string table of the shipped `FalloutLauncher.exe`.
const FALLOUT3_LAUNCHER: [(&str, &str); 7] = [
    ("CompanyName", "Bethesda Softworks"),
    ("FileDescription", "Fallout 3 Launcher"),
    ("FileVersion", "1.1.12.000"),
    ("InternalName", "Fallout3Launcher"),
    ("OriginalFilename", "FalloutLauncher.exe"),
    ("ProductName", "Fallout 3 Launcher"),
    ("ProductVersion", "1.1.12.000"),
];

/// The three fields read out of the shipped `FalloutNV.exe`.
const FALLOUTNV: [(&str, &str); 3] = [
    ("CompanyName", "Bethesda Softworks"),
    ("ProductName", "Fallout: New Vegas"),
    ("OriginalFilename", "FalloutNV.exe"),
];

/// The three fields read out of the shipped `FalloutNVLauncher.exe`.
const FALLOUTNV_LAUNCHER: [(&str, &str); 3] = [
    ("CompanyName", "Bethesda Softworks, Obsidian Entertainment"),
    ("ProductName", "Fallout New Vegas Launcher"),
    ("OriginalFilename", "FalloutNVLauncher.exe"),
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

/// An identity from a name and a version resource, started outside Steam.
fn identity(exe: &str, version_blob: Option<&[u16]>) -> AppIdentity {
    AppIdentity::new(exe.to_owned(), version_blob, None)
}

#[test]
fn the_version_fields_decode_past_the_entry_headers() {
    let id = identity("GTAIV.exe", Some(&blob(&GTA_IV)));
    assert_eq!(id.company, "Rockstar Games");
    assert_eq!(id.product, "Grand Theft Auto IV");
    assert_eq!(id.original_filename, "GTAIV.exe");
}

#[test]
fn an_image_without_a_version_resource_has_empty_fields() {
    let id = identity("GTAIV.exe", None);
    assert_eq!(id.company, "");
    assert_eq!(id.product, "");
    assert_eq!(id.original_filename, "");
}

#[test]
fn a_steam_app_id_is_decimal_digits_that_name_an_app() {
    for value in [
        "",
        "0",
        "00",
        " 730",
        "730 ",
        "+730",
        "-1",
        "abc",
        "7a30",
        "4294967296",
    ] {
        assert_eq!(steam_app_id(value), None, "{value:?}");
    }
    assert_eq!(steam_app_id("730"), Some(730));
    assert_eq!(steam_app_id("4465480"), Some(4_465_480));
    assert_eq!(steam_app_id("4294967295"), Some(u32::MAX));
    let id = AppIdentity::new("csgo.exe".to_owned(), None, Some("4465480"));
    assert_eq!(id.steam_app, Some(4_465_480));
    let unset = AppIdentity::new("csgo.exe".to_owned(), None, None);
    assert_eq!(unset.steam_app, None);
}

#[test]
fn the_csgo_profile_matches_the_legacy_build_under_either_app_id() {
    for exe in ["csgo.exe", "CSGO.EXE"] {
        for app in ["4465480", "730"] {
            let id = AppIdentity::new(exe.to_owned(), None, Some(app));
            assert_eq!(
                lookup(&id).map(AppProfile::name),
                Some("csgo"),
                "{exe} under {app}"
            );
        }
    }
}

#[test]
fn the_csgo_profile_needs_its_app_id_and_its_executable() {
    for app in [None, Some(""), Some("0"), Some("440"), Some("abc")] {
        let id = AppIdentity::new("csgo.exe".to_owned(), None, app);
        assert!(lookup(&id).is_none(), "{app:?}");
    }
    // Counter-Strike 2 shares the 730 app id.
    let cs2 = AppIdentity::new("cs2.exe".to_owned(), None, Some("730"));
    assert!(lookup(&cs2).is_none());
    // Another executable started under the legacy build's id keeps the stock
    // identity too.
    let steam = AppIdentity::new("steam.exe".to_owned(), None, Some("4465480"));
    assert!(lookup(&steam).is_none());
}

#[test]
fn the_csgo_profile_resolves_to_the_options_it_names() {
    let id = AppIdentity::new("csgo.exe".to_owned(), None, Some("4465480"));
    let profile = lookup(&id).expect("the csgo profile matches");
    let stock = parse(None, "", None);
    assert_eq!(stock.adapter_spoof, AdapterSpoof::None);
    assert!(!stock.buffer_draw_time_upload);
    // A typo in the settings string would keep Apple's vendor id, which the
    // game refuses at start-up, or free the buffer copies it writes after
    // `Unlock`, which ends the process at the first map.
    let cfg = parse(Some(profile), "", None);
    assert_eq!(cfg.adapter_spoof, AdapterSpoof::Amd);
    assert!(cfg.buffer_draw_time_upload);
}

#[test]
fn a_version_pinned_profile_ignores_the_steam_app_id() {
    for app in [None, Some("4465480"), Some("730"), Some("12210")] {
        let id = AppIdentity::new("GTAIV.exe".to_owned(), Some(&blob(&GTA_IV)), app);
        assert_eq!(lookup(&id).map(AppProfile::name), Some("gta-iv"), "{app:?}");
    }
}

#[test]
fn the_gta_iv_profile_matches_the_real_binarys_resource() {
    let id = identity("GTAIV.exe", Some(&blob(&GTA_IV)));
    assert_eq!(lookup(&id).map(AppProfile::name), Some("gta-iv"));
    // The name is matched case-insensitively: a launcher may spell it either way.
    let shouting = identity("gtaiv.exe", Some(&blob(&GTA_IV)));
    assert_eq!(lookup(&shouting).map(AppProfile::name), Some("gta-iv"));
}

#[test]
fn the_oblivion_profile_matches_the_real_binarys_resource() {
    let id = identity("Oblivion.exe", Some(&blob(&OBLIVION)));
    assert_eq!(lookup(&id).map(AppProfile::name), Some("oblivion"));
    // The name is matched case-insensitively: a launcher may spell it either way.
    let shouting = identity("OBLIVION.EXE", Some(&blob(&OBLIVION)));
    assert_eq!(lookup(&shouting).map(AppProfile::name), Some("oblivion"));
    // A same-named binary from anyone else must not pick it up.
    let impostor = identity("Oblivion.exe", None);
    assert!(lookup(&impostor).is_none());
}

#[test]
fn the_oblivion_profile_needs_both_its_company_and_its_product() {
    for (company, product) in [
        ("Somebody Else", "TES4: Oblivion"),
        ("Bethesda Softworks", "Somebody Else's Game"),
    ] {
        let other = blob(&[("CompanyName", company), ("ProductName", product)]);
        let id = identity("Oblivion.exe", Some(&other));
        assert!(lookup(&id).is_none(), "{company}, {product}");
    }
}

#[test]
fn the_fallout3_profile_matches_the_game_and_its_launcher() {
    assert_eq!(parse(None, "", None).adapter_spoof, AdapterSpoof::None);
    for (exe, fields) in [
        ("Fallout3.exe", &FALLOUT3),
        ("FalloutLauncher.exe", &FALLOUT3_LAUNCHER),
        // The name is matched case-insensitively: a launcher may spell it either way.
        ("FALLOUT3.EXE", &FALLOUT3),
        ("FALLOUTLAUNCHER.EXE", &FALLOUT3_LAUNCHER),
    ] {
        let id = identity(exe, Some(&blob(fields)));
        let profile = lookup(&id).expect("the fallout3 profile matches");
        assert_eq!(profile.name(), "fallout3", "{exe}");
        // A typo in the settings string would keep the stock `Driver` string,
        // and New Game would crash.
        assert_eq!(
            parse(Some(profile), "", None).adapter_spoof,
            AdapterSpoof::Nvidia,
            "{exe}"
        );
    }
    // A same-named binary from anyone else must not pick it up.
    let impostor = identity("Fallout3.exe", None);
    assert!(lookup(&impostor).is_none());
}

#[test]
fn the_fallout3_profile_needs_both_its_company_and_its_product() {
    for exe in ["Fallout3.exe", "FalloutLauncher.exe"] {
        for (company, product) in [
            ("Somebody Else", "Fallout 3"),
            ("Bethesda Softworks", "Somebody Else's Game"),
        ] {
            let other = blob(&[("CompanyName", company), ("ProductName", product)]);
            let id = identity(exe, Some(&other));
            assert!(lookup(&id).is_none(), "{exe}: {company}, {product}");
        }
    }
}

#[test]
fn the_fallout3_profile_leaves_other_bethesda_tools_alone() {
    // Only the two listed names match: an editor carrying the game's own
    // strings keeps the stock identity.
    let id = identity("GECK.exe", Some(&blob(&FALLOUT3)));
    assert!(lookup(&id).is_none());
}

#[test]
fn the_falloutnv_profile_matches_the_game_and_its_launcher() {
    assert_eq!(parse(None, "", None).adapter_spoof, AdapterSpoof::None);
    for (exe, fields) in [
        ("FalloutNV.exe", &FALLOUTNV),
        ("FalloutNVLauncher.exe", &FALLOUTNV_LAUNCHER),
        // The name is matched case-insensitively: a launcher may spell it either way.
        ("FALLOUTNV.EXE", &FALLOUTNV),
        ("FALLOUTNVLAUNCHER.EXE", &FALLOUTNV_LAUNCHER),
    ] {
        let id = identity(exe, Some(&blob(fields)));
        let profile = lookup(&id).expect("the falloutnv profile matches");
        assert_eq!(profile.name(), "falloutnv", "{exe}");
        // A typo in the settings string would keep the stock `Driver` string,
        // and New Game would crash.
        assert_eq!(
            parse(Some(profile), "", None).adapter_spoof,
            AdapterSpoof::Nvidia,
            "{exe}"
        );
    }
    // A same-named binary from anyone else must not pick it up.
    for exe in ["FalloutNV.exe", "FalloutNVLauncher.exe"] {
        let impostor = identity(exe, None);
        assert!(lookup(&impostor).is_none(), "{exe}");
    }
}

#[test]
fn the_falloutnv_profile_needs_both_its_company_and_its_product() {
    for exe in ["FalloutNV.exe", "FalloutNVLauncher.exe"] {
        for (company, product) in [
            ("Somebody Else", "Fallout: New Vegas"),
            ("Bethesda Softworks", "Somebody Else's Game"),
        ] {
            let other = blob(&[("CompanyName", company), ("ProductName", product)]);
            let id = identity(exe, Some(&other));
            assert!(lookup(&id).is_none(), "{exe}: {company}, {product}");
        }
    }
}

#[test]
fn the_falloutnv_profile_leaves_other_bethesda_tools_alone() {
    // Only the two listed names match: an editor carrying the game's own
    // strings keeps the stock identity.
    let id = identity("GECK.exe", Some(&blob(&FALLOUTNV)));
    assert!(lookup(&id).is_none());
}

#[test]
fn the_wow_profile_matches_both_clients_and_keeps_the_immediate_answers() {
    for version in ["Version 1.12", "Version 3.3"] {
        let blob = blob(&[
            ("CompanyName", "Blizzard Entertainment"),
            ("ProductName", "World of Warcraft"),
            ("ProductVersion", version),
        ]);
        let id = identity("WoW.exe", Some(&blob));
        let profile = lookup(&id).expect("the wow profile matches");
        assert_eq!(profile.name(), "wow");
        let stock = parse(None, "", None);
        assert!(!stock.query_flush_immediate);
        assert!(!stock.query_event_immediate);
        assert_eq!(stock.max_resolutions, 0);
        let profiled = parse(Some(profile), "", None);
        assert!(profiled.query_flush_immediate);
        assert!(profiled.query_event_immediate);
        // The resolution dropdown holds 32 buttons, one per size.
        assert_eq!(profiled.max_resolutions, 32);
    }
    let nameless = identity("WoW.exe", None);
    assert!(lookup(&nameless).is_none());
}

#[test]
fn a_pinned_field_is_a_substring_test() {
    let suffixed = blob(&[
        ("CompanyName", "Rockstar Games, Inc."),
        ("ProductName", "Grand Theft Auto IV Complete Edition"),
    ]);
    let id = identity("GTAIV.exe", Some(&suffixed));
    assert_eq!(lookup(&id).map(AppProfile::name), Some("gta-iv"));
}

#[test]
fn an_unrelated_program_of_the_same_name_is_left_alone() {
    let nameless = identity("GTAIV.exe", None);
    assert!(lookup(&nameless).is_none());
    let other = blob(&[
        ("CompanyName", "Somebody Else"),
        ("ProductName", "Grand Theft Auto IV"),
    ]);
    let id = identity("GTAIV.exe", Some(&other));
    assert!(lookup(&id).is_none());
}

#[test]
fn an_unprofiled_game_resolves_to_nothing() {
    let id = identity(
        "Morrowind.exe",
        Some(&blob(&[("CompanyName", "Bethesda Softworks")])),
    );
    assert!(lookup(&id).is_none());
}

#[test]
fn every_profile_is_uniquely_named_and_pins_more_than_its_executable() {
    for profile in PROFILES {
        assert!(!profile.name.is_empty(), "a profile has no name");
        assert!(!profile.exes.is_empty(), "{} has no exe", profile.name);
        assert!(
            profile.exes.iter().all(|exe| !exe.is_empty()),
            "{} lists an empty exe",
            profile.name
        );
        assert!(
            profile.company.is_some()
                || profile.product.is_some()
                || profile.original_filename.is_some()
                || !profile.steam_apps.is_empty(),
            "{} matches on its executable alone",
            profile.name
        );
        // No process carries app id 0, so such an entry could never match.
        assert!(
            !profile.steam_apps.contains(&0),
            "{} lists Steam app id 0",
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
    // Each identity is tried with no app id and with every id any profile
    // lists, so a version-pinned profile and an app-id-pinned one for the same
    // executable cannot both match a process that carries both pins.
    let apps: Vec<Option<String>> = core::iter::once(None)
        .chain(
            PROFILES
                .iter()
                .flat_map(|p| p.steam_apps)
                .map(|app| Some(app.to_string())),
        )
        .collect();
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
        let mut own = 0;
        for exe in profile.exes {
            for app in &apps {
                let id = AppIdentity::new((*exe).to_owned(), Some(&blob(&entries)), app.as_deref());
                if !profile.matches(&id) {
                    // An app id this profile does not accept: not its identity.
                    continue;
                }
                own += 1;
                for (j, other) in PROFILES.iter().enumerate() {
                    assert!(
                        i == j || !other.matches(&id),
                        "{} and {} both match {exe} under {app:?}",
                        profile.name,
                        other.name
                    );
                }
            }
        }
        assert!(
            own > 0,
            "{} matches none of its own identities",
            profile.name
        );
    }
}

#[test]
fn the_halo_2_profile_matches_and_stops_trusting_the_lock_bounds() {
    let blob = blob(&[
        ("CompanyName", "Microsoft Corporation"),
        ("ProductName", "Halo 2 for Windows Vista"),
        ("OriginalFilename", "Halo2.exe"),
    ]);
    let id = identity("halo2.exe", Some(&blob));
    let profile = lookup(&id).expect("the halo2 profile matches");
    assert_eq!(profile.name(), "halo2");
    // The whole point of the entry: the default trusts the announced range and
    // this title writes past it, so a typo here would render a black world.
    assert!(!parse(None, "", None).buffer_ignore_lock_bounds);
    assert!(parse(Some(profile), "", None).buffer_ignore_lock_bounds);
    // A same-named binary from anyone else must not pick it up.
    let impostor = identity("halo2.exe", None);
    assert!(lookup(&impostor).is_none());
}

#[test]
fn the_gta_iv_profile_resolves_to_the_options_it_names() {
    let id = identity("GTAIV.exe", Some(&blob(&GTA_IV)));
    let profile = lookup(&id).expect("the profile matches its own fixture");
    let cfg = parse(Some(profile), "", None);
    // A typo in a profile's settings string would leave every default in place,
    // so these four assertions are what pins the key spellings.
    assert_eq!(cfg.adapter_spoof, AdapterSpoof::Amd);
    assert!(!cfg.df_formats);
    assert!(!cfg.query_flush_immediate);
    assert!(cfg.depth_alias_same_size);
}

#[test]
fn the_oblivion_profile_resolves_to_the_nvidia_identity() {
    let id = identity("Oblivion.exe", Some(&blob(&OBLIVION)));
    let profile = lookup(&id).expect("the profile matches its own fixture");
    // A typo in the settings string would keep the stock `Driver` string, and
    // the game would stay at the level that greys its two shadow options.
    assert_eq!(parse(None, "", None).adapter_spoof, AdapterSpoof::None);
    assert_eq!(
        parse(Some(profile), "", None).adapter_spoof,
        AdapterSpoof::Nvidia
    );
}

/// A profile that sets two unrelated options, for the precedence cases.
const PROBE: AppProfile = AppProfile {
    name: "probe",
    exes: &["probe.exe"],
    company: Some("Probe"),
    product: None,
    original_filename: None,
    steam_apps: &[],
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

#[test]
fn a_builtin_profile_is_found_by_name_alone() {
    let wow = builtin("wow").expect("the wow profile ships");
    assert_eq!(wow.name(), "wow");
    assert!(wow.settings().contains("query.eventImmediate=true"));
    assert!(
        builtin("WoW.exe").is_none(),
        "an executable name is not a profile name"
    );
}
