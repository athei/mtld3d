use super::{
    ModeRequest, ModeSetOutcome, RegistryRestore, drop_unscalable_sizes, effective_refresh, gcd,
    listed_modes, mode_set_attempts, monitor_ratio_fits, physical_extent, select_mode_sizes,
    served_mode_indices, served_modes, served_sizes,
};

const MBP: (u32, u32) = (3456, 2234);

#[test]
fn a_request_without_a_rate_is_one_attempt() {
    let request = ModeRequest {
        width: 1280,
        height: 720,
        refresh_hz: 0,
    };
    assert_eq!(
        mode_set_attempts(request).collect::<Vec<_>>(),
        vec![request]
    );
}

#[test]
fn a_request_with_a_rate_retries_without_it() {
    let request = ModeRequest {
        width: 1280,
        height: 720,
        refresh_hz: 60,
    };
    assert_eq!(
        mode_set_attempts(request).collect::<Vec<_>>(),
        vec![
            request,
            ModeRequest {
                refresh_hz: 0,
                ..request
            }
        ]
    );
}

#[test]
fn the_desktop_mode_comes_first_and_order_is_kept() {
    let sizes = select_mode_sizes(MBP, [(640, 480), (1920, 1200), (1280, 720)]);
    assert_eq!(sizes, vec![MBP, (640, 480), (1920, 1200), (1280, 720)]);
}

#[test]
fn duplicates_and_the_desktop_itself_appear_once() {
    let sizes = select_mode_sizes(MBP, [(640, 480), MBP, (640, 480), (640, 480)]);
    assert_eq!(sizes, vec![MBP, (640, 480)]);
}

#[test]
fn sizes_larger_than_the_desktop_on_either_axis_are_dropped() {
    let sizes = select_mode_sizes((1728, 1117), [(1920, 1080), (1600, 1200), (1680, 1050)]);
    assert_eq!(sizes, vec![(1728, 1117), (1680, 1050)]);
}

#[test]
fn sizes_of_any_aspect_stay_settable() {
    // 5:4 and 21:9 on a 3:2-ish panel; win32u letterboxes them.
    let sizes = select_mode_sizes(MBP, [(1280, 1024), (2560, 1080), (1024, 768)]);
    assert_eq!(sizes, vec![MBP, (1280, 1024), (2560, 1080), (1024, 768)]);
}

#[test]
fn degenerate_sizes_are_dropped() {
    let sizes = select_mode_sizes(MBP, [(0, 480), (640, 0)]);
    assert_eq!(sizes, vec![MBP]);
}

#[test]
fn an_empty_enumeration_serves_the_desktop_mode_alone() {
    assert_eq!(select_mode_sizes(MBP, []), vec![MBP]);
}

/// The sizes Wine lists for a 3456x2234 display under `EmulateModeset`.
const MBP_WINE_SIZES: [(u32, u32); 43] = [
    (1024, 768),
    (1152, 864),
    (1168, 730),
    (1168, 755),
    (1280, 1024),
    (1280, 720),
    (1280, 768),
    (1280, 800),
    (1280, 960),
    (1312, 820),
    (1312, 848),
    (1440, 900),
    (1440, 960),
    (1496, 935),
    (1496, 967),
    (1600, 1200),
    (1600, 900),
    (1680, 1050),
    (1728, 1080),
    (1728, 1117),
    (1920, 1080),
    (1920, 1200),
    (1920, 1280),
    (1920, 800),
    (2056, 1285),
    (2056, 1329),
    (2336, 1460),
    (2336, 1510),
    (2560, 1080),
    (2560, 1440),
    (2560, 1600),
    (2624, 1640),
    (2624, 1696),
    (2880, 1620),
    (2992, 1870),
    (2992, 1934),
    (3200, 1800),
    (3456, 2160),
    (3456, 2234),
    (640, 480),
    (800, 600),
    (960, 540),
    (960, 600),
];

/// The sizes Wine lists for a 1728x1117 display (Retina mode off).
const MBP_NORETINA_WINE_SIZES: [(u32, u32); 23] = [
    (640, 480),
    (800, 600),
    (960, 540),
    (960, 600),
    (1024, 768),
    (1152, 864),
    (1168, 730),
    (1168, 755),
    (1280, 720),
    (1280, 768),
    (1280, 800),
    (1280, 960),
    (1280, 1024),
    (1312, 820),
    (1312, 848),
    (1440, 900),
    (1440, 960),
    (1496, 935),
    (1496, 967),
    (1600, 900),
    (1680, 1050),
    (1728, 1080),
    (1728, 1117),
];

/// The sizes the adapter serves from a Win32 list at `dpi`, bounded to `max`.
fn served_at(desktop: (u32, u32), sizes: &[(u32, u32)], dpi: u32, max: usize) -> Vec<(u32, u32)> {
    let physical = physical_extent(desktop, sizes.iter().copied());
    let mut settable = select_mode_sizes(desktop, sizes.iter().copied());
    drop_unscalable_sizes(&mut settable, physical, dpi);
    let served = served_sizes(&settable, max);
    // Only sizes win32u can scale to are served, and none twice.
    assert!(
        served
            .iter()
            .all(|&size| monitor_ratio_fits(size, physical, dpi))
    );
    let mut distinct = served.clone();
    distinct.dedup();
    assert_eq!(distinct.len(), served.len(), "{served:?}");
    served
}

/// The sizes a 3456x2234 display serves at 192 dpi with no bound, smallest first.
const MBP_192_DPI_SERVED: [(u32, u32); 31] = [
    (640, 480),
    (800, 600),
    (960, 540),
    (960, 600),
    (1024, 768),
    (1152, 864),
    (1280, 720),
    (1280, 768),
    (1280, 800),
    (1280, 960),
    (1280, 1024),
    (1312, 848),
    (1440, 900),
    (1440, 960),
    (1600, 900),
    (1600, 1200),
    (1728, 1080),
    (1728, 1117),
    (1920, 800),
    (1920, 1080),
    (1920, 1200),
    (1920, 1280),
    (2560, 1080),
    (2560, 1440),
    (2560, 1600),
    (2624, 1640),
    (2624, 1696),
    (2880, 1620),
    (3200, 1800),
    (3456, 2160),
    (3456, 2234),
];

#[test]
fn every_settable_size_is_served_smallest_first_without_a_bound() {
    // At 192 dpi win32u cannot scale the monitor to twelve of the 43 sizes
    // Wine lists; the other 31 are all served, ordered by width then height.
    assert_eq!(
        served_at(MBP, &MBP_WINE_SIZES, 192, 0),
        MBP_192_DPI_SERVED.to_vec()
    );
}

#[test]
fn at_96_dpi_the_same_display_serves_the_sizes_only_192_dpi_cannot_scale_to() {
    // 1168x730 and the other heights sharing a factor of 3 or less with
    // 96 * 2234 stay out at 96 dpi too; 1312x820 and 1680x1050 come back.
    let served = served_at(MBP, &MBP_WINE_SIZES, 96, 0);
    assert_eq!(served.len(), 34);
    for size in [(1312, 820), (1680, 1050), (2336, 1460)] {
        assert!(served.contains(&size), "{size:?} missing from {served:?}");
    }
    for height in UNSCALABLE_MBP_HEIGHTS {
        assert!(
            served.iter().all(|&(_, h)| h != height),
            "a {height}-high size served: {served:?}"
        );
    }
    assert!(served.is_sorted(), "{served:?}");
}

#[test]
fn the_same_panel_with_retina_mode_off_serves_its_list_less_three_unscalable_sizes() {
    // On a 1117-pixel-high display at 96 dpi the heights 755, 935 and 967
    // share no factor with 96 * 1117 and stay out; the other 20 are served.
    let served = served_at((1728, 1117), &MBP_NORETINA_WINE_SIZES, 96, 0);
    assert_eq!(served.len(), 20, "{served:?}");
    for size in [(1168, 755), (1496, 935), (1496, 967)] {
        assert!(!served.contains(&size), "{size:?} served");
    }
    assert_eq!(served.first(), Some(&(640, 480)));
    assert_eq!(served.last(), Some(&(1728, 1117)));
}

/// The sizes the mac driver lists for a 2560x1664 panel on a 1470x956 desktop.
///
/// A driver that reports several modes gets no virtual sizes from win32u, so
/// the list is the panel's own: its native size and the desktop sizes macOS
/// offers it, of which seven fit on the desktop.
const PANEL_HOST_SIZES: [(u32, u32); 8] = [
    (2560, 1664),
    (1470, 956),
    (1470, 918),
    (1280, 832),
    (1280, 800),
    (1024, 666),
    (1024, 640),
    (960, 600),
];

#[test]
fn a_host_mode_list_serves_every_size_that_fits_on_the_desktop() {
    // Every one of the seven sizes no larger than the desktop is served,
    // 1280x800 among them, with no bound and under the 32 of the WoW
    // profile alike; the native size is larger than the desktop and is not.
    let expected = vec![
        (960, 600),
        (1024, 640),
        (1024, 666),
        (1280, 800),
        (1280, 832),
        (1470, 918),
        (1470, 956),
    ];
    assert_eq!(served_at((1470, 956), &PANEL_HOST_SIZES, 96, 0), expected);
    assert_eq!(served_at((1470, 956), &PANEL_HOST_SIZES, 96, 32), expected);
    // The main module's `EnumDisplaySettings` walk reaches every listed
    // position of a served size, 1280x800's included.
    assert_eq!(
        served_mode_indices(PANEL_HOST_SIZES, &expected),
        vec![1, 2, 3, 4, 5, 6, 7]
    );
}

#[test]
fn a_bound_keeps_the_largest_sizes_and_the_desktop() {
    // 32 is the size count a 32-button menu holds; at 96 dpi the display has
    // 34, so the two smallest go.
    let served = served_at(MBP, &MBP_WINE_SIZES, 96, 32);
    assert_eq!(served.len(), 32);
    assert!(!served.contains(&(640, 480)));
    assert!(!served.contains(&(800, 600)));
    assert!(served.contains(&MBP));
    assert!(served.is_sorted(), "{served:?}");
}

#[test]
fn a_bound_at_or_above_the_size_count_changes_nothing() {
    assert_eq!(
        served_at(MBP, &MBP_WINE_SIZES, 192, 32),
        MBP_192_DPI_SERVED.to_vec()
    );
    assert_eq!(
        served_at(MBP, &MBP_WINE_SIZES, 192, 31),
        MBP_192_DPI_SERVED.to_vec()
    );
}

#[test]
fn a_tight_bound_keeps_the_largest_by_pixel_count() {
    // 1920x800 is wider than 1600x1200 but has fewer pixels (1536000 to
    // 1920000), so area decides, not width.
    let settable = [(2560, 1600), (1920, 800), (1600, 1200), (640, 480)];
    assert_eq!(served_sizes(&settable, 2), vec![(1600, 1200), (2560, 1600)]);
    // Equal areas go to the wider size.
    assert_eq!(
        served_sizes(&[(2000, 2000), (1000, 4000), (4000, 1000)], 2),
        vec![(2000, 2000), (4000, 1000)]
    );
}

#[test]
fn the_desktop_survives_a_bound_even_when_it_is_not_among_the_largest() {
    // A virtual mode current when the table is built is the first entry,
    // smaller than sizes the display still lists.
    let settable = [(1024, 768), (3456, 2234), (2624, 1696), (640, 480)];
    assert_eq!(served_sizes(&settable, 2), vec![(1024, 768), (3456, 2234)]);
}

#[test]
fn a_bound_of_one_serves_the_desktop_alone() {
    assert_eq!(served_sizes(&[MBP, (640, 480), (2624, 1696)], 1), vec![MBP]);
}

#[test]
fn an_empty_settable_list_serves_nothing() {
    assert_eq!(served_sizes(&[], 5), [] as [(u32, u32); 0]);
    assert_eq!(served_sizes(&[], 0), [] as [(u32, u32); 0]);
}

#[test]
fn every_refresh_rate_of_a_served_size_is_served_lowest_first() {
    // Win32 lists each size at the desktop's rate and at 60 Hz, in its own
    // order and with repeats across depths; the served list holds each
    // (size, rate) once, by size and then rate.
    let modes = [
        ((1440, 900), 120),
        ((640, 480), 120),
        ((1440, 900), 60),
        ((3456, 2234), 120),
        ((640, 480), 60),
        ((1440, 900), 120),
        ((3456, 2234), 60),
        ((1168, 730), 120),
    ];
    assert_eq!(
        served_modes(modes, &[(640, 480), (1440, 900), (3456, 2234)]),
        vec![
            ((640, 480), 60),
            ((640, 480), 120),
            ((1440, 900), 60),
            ((1440, 900), 120),
            ((3456, 2234), 60),
            ((3456, 2234), 120),
        ]
    );
}

#[test]
fn a_default_rate_reads_as_the_desktops_and_the_desktop_mode_is_always_listed() {
    // A virtual display lists 0 Hz for its sizes, and not the desktop's own
    // rate at all: every mode reads at 60 Hz, and the desktop's 60 Hz mode
    // is there once ordered.
    let listed = [((640, 480), 0), ((1024, 768), 1), ((2048, 1536), 0)];
    let modes = listed_modes(listed, ((2048, 1536), 60));
    assert_eq!(
        served_modes(modes, &[(640, 480), (1024, 768), (2048, 1536)]),
        vec![((640, 480), 60), ((1024, 768), 60), ((2048, 1536), 60)]
    );
    // A list that has the desktop's size at another rate only gains the
    // desktop's rate beside it.
    let modes = listed_modes([(MBP, 60)], (MBP, 120));
    assert_eq!(served_modes(modes, &[MBP]), vec![(MBP, 60), (MBP, 120)]);
}

#[test]
fn a_known_rate_passes_through_and_an_unknown_one_takes_the_fallback() {
    assert_eq!(effective_refresh(120, 60), 120);
    assert_eq!(effective_refresh(2, 60), 2);
    assert_eq!(effective_refresh(1, 60), 60);
    assert_eq!(effective_refresh(0, 75), 75);
}

#[test]
fn a_size_left_out_of_the_served_sizes_has_no_modes() {
    assert_eq!(
        served_modes([((1168, 730), 120), ((1168, 730), 60)], &[MBP]),
        [] as [((u32, u32), u32); 0]
    );
}

#[test]
fn served_indices_are_the_positions_of_served_sizes_in_list_order() {
    // user32 lists every depth and rate of a size; each occurrence keeps
    // its position, sizes not served leave gaps.
    let list = [
        (640, 480),
        (1920, 1200),
        (640, 480),
        (1280, 1024),
        (1920, 1200),
        MBP,
    ];
    assert_eq!(
        served_mode_indices(list, &[MBP, (1920, 1200)]),
        vec![1, 4, 5]
    );
}

#[test]
fn no_served_size_in_the_list_yields_no_indices() {
    assert_eq!(served_mode_indices([(640, 480)], &[MBP]), [] as [u32; 0]);
    assert_eq!(served_mode_indices([], &[MBP]), [] as [u32; 0]);
}

/// The heights a 2234-pixel-high display at 96 dpi cannot be scaled to.
///
/// Each shares a factor of 3 or less with 96 * 2234 = 214464, so the reduced
/// numerator stays at 71488 or more. All of them are sizes Wine lists for
/// that display.
const UNSCALABLE_MBP_HEIGHTS: [u32; 9] = [1934, 1870, 1510, 1329, 1285, 967, 935, 755, 730];

#[test]
fn heights_sharing_too_small_a_factor_with_the_physical_height_do_not_fit() {
    for height in UNSCALABLE_MBP_HEIGHTS {
        assert!(
            !monitor_ratio_fits((MBP.0, height), MBP, 96),
            "{}x{height} fits",
            MBP.0
        );
    }
}

#[test]
fn common_and_panel_sizes_fit() {
    for size in [(640, 480), (1728, 1117), (3456, 2160), (2624, 1696), MBP] {
        assert!(monitor_ratio_fits(size, MBP, 96), "{size:?} does not fit");
    }
}

#[test]
fn the_width_is_checked_as_well_as_the_height() {
    // 96 * 3456 = 331776 needs a common factor of 6 or more; an odd width
    // shares at most a 3 with it (3455 shares 1, 3453 shares 3).
    assert!(!monitor_ratio_fits((3455, MBP.1), MBP, 96));
    assert!(!monitor_ratio_fits((3453, MBP.1), MBP, 96));
    assert!(monitor_ratio_fits((3456, MBP.1), MBP, 96));
    // Either axis alone fails the size.
    assert!(!monitor_ratio_fits((3455, 2160), MBP, 96));
    assert!(!monitor_ratio_fits((1728, 1934), MBP, 96));
}

#[test]
fn the_dpi_scales_the_numerator() {
    // 1028 shares 4 with 96 * 2234 = 214464, leaving 53616, which fits; at
    // 192 dpi the same factor leaves 107232, which does not.
    assert!(monitor_ratio_fits((MBP.0, 1028), MBP, 96));
    assert!(!monitor_ratio_fits((MBP.0, 1028), MBP, 192));
    // 1117 divides the numerator at any dpi, and 1728 divides 3456.
    assert!(monitor_ratio_fits((1728, 1117), MBP, 1000));
}

#[test]
fn zero_terms_never_fail() {
    assert!(monitor_ratio_fits((0, 0), MBP, 96));
    assert!(monitor_ratio_fits((2992, 1934), MBP, 0));
    assert!(monitor_ratio_fits((2992, 1934), (0, 0), 96));
}

#[test]
fn gcd_reduces_like_euclid() {
    assert_eq!(gcd(214_464, 1934), 2);
    assert_eq!(gcd(214_464, 1117), 1117);
    assert_eq!(gcd(0, 7), 7);
    assert_eq!(gcd(7, 0), 7);
}

#[test]
fn the_physical_extent_is_the_largest_size_on_each_axis() {
    // A virtual mode is current; the list still carries the physical mode.
    let list = [(640, 480), MBP, (1728, 1117), (2992, 1934)];
    assert_eq!(physical_extent((1728, 1117), list), MBP);
    assert_eq!(physical_extent(MBP, []), MBP);
    // A list with no dominating entry yields the extent of both axes.
    assert_eq!(
        physical_extent((1920, 1080), [(1920, 1200), (2048, 1152)]),
        (2048, 1200)
    );
}

#[test]
fn unscalable_sizes_are_left_out_in_list_order() {
    let mut settable = vec![
        MBP,
        (2992, 1934),
        (1728, 1117),
        (1496, 967),
        (640, 480),
        (2336, 1510),
    ];
    let dropped = drop_unscalable_sizes(&mut settable, MBP, 96);
    assert_eq!(settable, vec![MBP, (1728, 1117), (640, 480)]);
    assert_eq!(dropped, vec![(2992, 1934), (1496, 967), (2336, 1510)]);
}

#[test]
fn the_desktop_stays_even_when_it_does_not_fit() {
    // A virtual mode current when the table is built is its first entry.
    let mut settable = vec![(2992, 1934), (1496, 967), (640, 480)];
    let dropped = drop_unscalable_sizes(&mut settable, MBP, 96);
    assert_eq!(settable, vec![(2992, 1934), (640, 480)]);
    assert_eq!(dropped, vec![(1496, 967)]);
}

#[test]
fn every_known_unscalable_size_is_dropped_and_the_desktop_never_is() {
    let mut settable = vec![MBP];
    settable.extend(UNSCALABLE_MBP_HEIGHTS.map(|height| (1000, height)));
    let dropped = drop_unscalable_sizes(&mut settable, MBP, 96);
    assert_eq!(settable, vec![MBP]);
    assert_eq!(dropped.len(), UNSCALABLE_MBP_HEIGHTS.len());
    assert!(!dropped.contains(&MBP));
}

#[test]
fn an_empty_settable_list_drops_nothing() {
    let mut settable = Vec::new();
    assert_eq!(
        drop_unscalable_sizes(&mut settable, MBP, 96),
        [] as [(u32, u32); 0]
    );
    assert_eq!(settable, [] as [(u32, u32); 0]);
}

const HD_60: ModeRequest = ModeRequest {
    width: 1280,
    height: 720,
    refresh_hz: 60,
};

#[test]
fn a_mode_request_renders_its_rate_only_when_it_names_one() {
    assert_eq!(HD_60.to_string(), "1280x720@60Hz");
    let any_rate = ModeRequest {
        refresh_hz: 0,
        ..HD_60
    };
    assert_eq!(any_rate.to_string(), "1280x720");
}

#[test]
fn a_mode_set_outcome_names_what_was_set_and_what_it_replaced() {
    let changed = ModeSetOutcome::Changed {
        requested: HD_60,
        set: HD_60,
        was: (1728, 1117),
    };
    assert_eq!(
        changed.to_string(),
        "display mode 1280x720@60Hz set (was 1728x1117)"
    );
    let rate_dropped = ModeSetOutcome::Changed {
        requested: HD_60,
        set: ModeRequest {
            refresh_hz: 0,
            ..HD_60
        },
        was: (1728, 1117),
    };
    assert_eq!(
        rate_dropped.to_string(),
        "display mode 1280x720 set for a 1280x720@60Hz request (was 1728x1117)"
    );
    assert_eq!(
        ModeSetOutcome::AlreadyCurrent(HD_60).to_string(),
        "display mode 1280x720@60Hz already current"
    );
    assert_eq!(
        ModeSetOutcome::Refused(HD_60).to_string(),
        "display mode 1280x720@60Hz refused"
    );
    assert_eq!(ModeSetOutcome::NoMode.to_string(), "no display mode set");
}

#[test]
fn only_a_mode_that_took_stays_in_place_and_it_is_the_request() {
    let rate_dropped = ModeSetOutcome::Changed {
        requested: HD_60,
        set: ModeRequest {
            refresh_hz: 0,
            ..HD_60
        },
        was: (1728, 1117),
    };
    assert_eq!(rate_dropped.mode_in_place(), Some(HD_60));
    assert_eq!(
        ModeSetOutcome::AlreadyCurrent(HD_60).mode_in_place(),
        Some(HD_60)
    );
    assert_eq!(ModeSetOutcome::Refused(HD_60).mode_in_place(), None);
    assert_eq!(ModeSetOutcome::NoMode.mode_in_place(), None);
}

#[test]
fn a_registry_restore_names_both_modes_and_a_failure_code() {
    let restored = RegistryRestore::Restored {
        registry: (1728, 1117),
        was: (1280, 720),
    };
    assert_eq!(
        restored.to_string(),
        "registry display mode 1728x1117 restored (was 1280x720)"
    );
    assert!(!restored.failed());
    let current = RegistryRestore::AlreadyCurrent((1728, 1117));
    assert_eq!(
        current.to_string(),
        "registry display mode 1728x1117 already current"
    );
    assert!(!current.failed());
    let failed = RegistryRestore::Failed {
        registry: (1728, 1117),
        was: (1280, 720),
        ret: -1,
    };
    assert_eq!(
        failed.to_string(),
        "restoring the registry display mode 1728x1117 over 1280x720 failed (ret=-1)"
    );
    assert!(failed.failed());
    assert_eq!(
        RegistryRestore::Unreadable.to_string(),
        "display mode unreadable, left as it is"
    );
}
