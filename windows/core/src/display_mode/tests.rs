use super::{
    MAX_SERVED_SIZES, ModeRequest, STANDARD_SIZES, drop_unscalable_sizes, fills_display, gcd,
    mode_set_attempts, monitor_ratio_fits, physical_extent, pixels, select_mode_sizes,
    served_mode_indices, served_mode_sizes,
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

#[test]
fn the_filling_sizes_are_served_largest_first_then_the_standard_sizes() {
    // 2336x1510 and 2992x1934 share the display's shape to within a bar of
    // under one physical pixel and come first, largest first.
    // 1920x1080, 1024x768 and 640x480 are standard sizes and follow them,
    // largest first. 2560x1600, 1920x1280 and 1440x900 are neither and are
    // left to a game's own config, however large.
    let settable = [
        MBP,
        (640, 480),
        (1024, 768),
        (1920, 1080),
        (2336, 1510),
        (2560, 1600),
        (2992, 1934),
        (1920, 1280),
        (1440, 900),
    ];
    assert_eq!(
        served_mode_sizes(&settable, MBP, 15),
        vec![
            MBP,
            (2992, 1934),
            (2336, 1510),
            (1920, 1080),
            (1024, 768),
            (640, 480)
        ]
    );
}

#[test]
fn a_standard_size_that_fills_the_display_is_served_once_among_the_filling_sizes() {
    let desktop = (3840, 2160);
    let settable = [desktop, (800, 600), (1280, 720), (2560, 1440)];
    assert_eq!(
        served_mode_sizes(&settable, desktop, 15),
        vec![desktop, (2560, 1440), (1280, 720), (800, 600)]
    );
}

#[test]
fn the_bound_cuts_the_standard_tier_first() {
    let settable = [MBP, (1920, 1080), (2624, 1696), (640, 480), (1728, 1117)];
    assert_eq!(
        served_mode_sizes(&settable, MBP, 4),
        vec![MBP, (2624, 1696), (1728, 1117), (1920, 1080)]
    );
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

/// The table a desktop serves from `sizes`, the way the adapter builds it.
fn served_for(desktop: (u32, u32), sizes: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let physical = physical_extent(desktop, sizes.iter().copied());
    let mut settable = select_mode_sizes(desktop, sizes.iter().copied());
    drop_unscalable_sizes(&mut settable, physical, 96);
    let served = served_mode_sizes(&settable, physical, MAX_SERVED_SIZES);
    // Only sizes win32u can scale to are served, and none twice.
    assert!(
        served
            .iter()
            .all(|&size| monitor_ratio_fits(size, physical, 96))
    );
    let mut distinct = served.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), served.len(), "{served:?}");
    served
}

#[test]
fn a_3456x2234_display_serves_its_panel_then_the_standard_sizes() {
    assert_eq!(
        served_for(MBP, &MBP_WINE_SIZES),
        vec![
            (3456, 2234),
            (2624, 1696),
            (1728, 1117),
            (1312, 848),
            (2560, 1440),
            (1920, 1080),
            (1600, 900),
            (1280, 720),
            (1024, 768),
            (800, 600),
            (640, 480),
        ]
    );
}

#[test]
fn the_same_panel_with_retina_mode_off_serves_its_panel_then_the_standard_sizes() {
    assert_eq!(
        served_for((1728, 1117), &MBP_NORETINA_WINE_SIZES),
        vec![
            (1728, 1117),
            (1312, 848),
            (1600, 900),
            (1280, 720),
            (1024, 768),
            (800, 600),
            (640, 480),
        ]
    );
}

/// Wine's own table of sizes, which it lists for any display at or below the physical size.
const WINE_TABLE: [(u32, u32); 24] = [
    (640, 480),
    (800, 600),
    (1024, 768),
    (1152, 864),
    (1280, 960),
    (1600, 1200),
    (960, 540),
    (1280, 720),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (2880, 1620),
    (3200, 1800),
    (1440, 900),
    (1680, 1050),
    (1920, 1200),
    (2560, 1600),
    (1440, 960),
    (1920, 1280),
    (2560, 1080),
    (1920, 800),
    (3840, 1600),
    (1280, 1024),
    (1280, 768),
];

/// `WINE_TABLE` as a display of `physical` lists it, with the physical mode last.
fn wine_list(physical: (u32, u32)) -> Vec<(u32, u32)> {
    WINE_TABLE
        .iter()
        .copied()
        .filter(|&(w, h)| w <= physical.0 && h <= physical.1)
        .chain(core::iter::once(physical))
        .collect()
}

#[test]
fn a_16_9_external_display_serves_its_own_aspect_then_the_4_3_standard_sizes() {
    let display = (3840, 2160);
    assert_eq!(
        served_for(display, &wine_list(display)),
        vec![
            display,
            (3200, 1800),
            (2880, 1620),
            (2560, 1440),
            (1920, 1080),
            (1600, 900),
            (1280, 720),
            (960, 540),
            (1024, 768),
            (800, 600),
            (640, 480),
        ]
    );
}

#[test]
fn a_16_10_display_serves_its_own_aspect_then_the_standard_sizes() {
    let display = (2560, 1600);
    assert_eq!(
        served_for(display, &wine_list(display)),
        vec![
            display,
            (1920, 1200),
            (1680, 1050),
            (1440, 900),
            (2560, 1440),
            (1920, 1080),
            (1600, 900),
            (1280, 720),
            (1024, 768),
            (800, 600),
            (640, 480),
        ]
    );
}

#[test]
fn a_4_3_2048x1536_display_serves_its_own_aspect_then_the_16_9_standard_sizes() {
    let display = (2048, 1536);
    assert_eq!(
        served_for(display, &wine_list(display)),
        vec![
            display,
            (1600, 1200),
            (1280, 960),
            (1152, 864),
            (1024, 768),
            (800, 600),
            (640, 480),
            (1920, 1080),
            (1600, 900),
            (1280, 720),
        ]
    );
}

#[test]
fn the_bound_cuts_the_smallest_filling_sizes() {
    let settable = [MBP, (2336, 1510), (2624, 1696), (2992, 1934)];
    assert_eq!(
        served_mode_sizes(&settable, MBP, 3),
        vec![MBP, (2992, 1934), (2624, 1696)]
    );
}

#[test]
fn filling_sizes_are_served_largest_first_whatever_their_enumeration_order() {
    let desktop = (3840, 2160);
    let settable = [desktop, (1280, 720), (1600, 900), (960, 540)];
    assert_eq!(
        served_mode_sizes(&settable, desktop, 15),
        vec![desktop, (1600, 900), (1280, 720), (960, 540)]
    );
}

#[test]
fn the_desktop_is_served_first_even_when_it_does_not_fill_the_display() {
    // A 4:3 mode is current when the table is built; the sizes that fill
    // the physical 3456x2234 display still come before the standard ones.
    let settable = [(1024, 768), (640, 480), (1728, 1117), (800, 600)];
    assert_eq!(
        served_mode_sizes(&settable, MBP, 15),
        vec![(1024, 768), (1728, 1117), (800, 600), (640, 480)]
    );
}

#[test]
fn an_exact_multiple_of_the_display_fills_it() {
    assert!(fills_display(MBP, MBP));
    assert!(fills_display((1728, 1117), MBP));
    assert!(fills_display((1920, 1080), (3840, 2160)));
}

#[test]
fn a_bar_under_one_physical_pixel_fills_the_display() {
    // 2624x1696 scaled by 3456/2624 is 2233.76 rows high, 0.24 short of
    // 2234; 1312x848 is the same shape.
    assert!(fills_display((2624, 1696), MBP));
    assert!(fills_display((1312, 848), MBP));
    // The sizes win32u cannot scale to share the panel's shape too.
    assert!(fills_display((2992, 1934), MBP));
    assert!(fills_display((2336, 1510), MBP));
}

#[test]
fn a_bar_of_one_physical_pixel_or_more_does_not_fill_the_display() {
    // One row short at the display's own width is a bar of exactly one row.
    assert!(!fills_display((3456, 2233), MBP));
    // 1728x1116 scales to 2232 rows, two short.
    assert!(!fills_display((1728, 1116), MBP));
    // The area below the notch and the 16:10 sizes leave bars.
    assert!(!fills_display((3456, 2160), MBP));
    assert!(!fills_display((2560, 1600), MBP));
    assert!(!fills_display((1440, 900), MBP));
    assert!(!fills_display((640, 480), MBP));
}

#[test]
fn a_size_is_fitted_to_whichever_axis_is_tighter() {
    // Wider than the display: fitted to the width, bars above and below.
    // 1001x500 on 2000x1000 scales to 999.001 rows, a bar under one row.
    assert!(fills_display((1001, 500), (2000, 1000)));
    assert!(!fills_display((1003, 500), (2000, 1000)));
    // Taller than the display: fitted to the height, bars left and right.
    // 1000x501 scales to 1996.008 columns, a bar of almost four.
    assert!(!fills_display((1000, 501), (2000, 1000)));
    // 500x1001 on 1000x2000 scales to 999.001 columns.
    assert!(fills_display((500, 1001), (1000, 2000)));
    assert!(!fills_display((500, 1003), (1000, 2000)));
}

#[test]
fn the_largest_extents_do_not_overflow() {
    let max = (u32::MAX, u32::MAX);
    assert!(fills_display(max, max));
    assert!(fills_display((1, 1), max));
    assert!(!fills_display((u32::MAX, 1), max));
    assert!(!fills_display((1, u32::MAX), max));
    assert!(!fills_display((u32::MAX, u32::MAX - 1), max));
}

#[test]
fn a_degenerate_size_fills_nothing() {
    assert!(!fills_display((0, 480), MBP));
    assert!(!fills_display((640, 0), MBP));
    assert!(!fills_display((640, 480), (0, 2234)));
    assert!(!fills_display((640, 480), (3456, 0)));
    assert!(!fills_display((0, 0), (0, 0)));
}

#[test]
fn the_standard_sizes_are_declared_largest_first() {
    // The declared order is the served order, with no sort behind it.
    assert!(
        STANDARD_SIZES
            .windows(2)
            .all(|pair| pixels(pair[0]) > pixels(pair[1])),
        "{STANDARD_SIZES:?}"
    );
}

#[test]
fn a_bound_of_zero_still_serves_the_desktop() {
    assert_eq!(served_mode_sizes(&[MBP, (640, 480)], MBP, 0), vec![MBP]);
}

#[test]
fn an_empty_settable_list_serves_nothing() {
    assert!(served_mode_sizes(&[], MBP, 5).is_empty());
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
    assert!(served_mode_indices([(640, 480)], &[MBP]).is_empty());
    assert!(served_mode_indices([], &[MBP]).is_empty());
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
    assert!(drop_unscalable_sizes(&mut settable, MBP, 96).is_empty());
    assert!(settable.is_empty());
}
