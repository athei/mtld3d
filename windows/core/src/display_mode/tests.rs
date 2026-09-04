use super::{
    ModeRequest, mode_set_attempts, select_mode_sizes, served_mode_indices, served_mode_sizes,
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
fn aspects_outside_the_tolerance_are_dropped() {
    // 5:4 is ~19 % off a 3:2-ish panel, 21:9 ~51 %; 4:3, 16:10 and 16:9 stay.
    let sizes = select_mode_sizes(
        MBP,
        [
            (1280, 1024),
            (2560, 1080),
            (1024, 768),
            (1920, 1200),
            (1920, 1080),
        ],
    );
    assert_eq!(sizes, vec![MBP, (1024, 768), (1920, 1200), (1920, 1080)]);
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
fn only_the_panels_aspect_is_served_largest_first() {
    // The desktop's aspect is 1.547; 2336x1510 and 2992x1934 share it within
    // the panel tolerance (integer rounding), the 16:10, 16:9 and 4:3 sizes
    // do not and are left to a game's own config, however large.
    let settable = [
        MBP,
        (640, 480),
        (1024, 768),
        (1920, 1080),
        (2336, 1510),
        (2560, 1600),
        (2992, 1934),
        (1920, 1280),
    ];
    assert_eq!(
        served_mode_sizes(&settable, 15),
        vec![MBP, (2992, 1934), (2336, 1510)]
    );
}

#[test]
fn the_bound_cuts_the_smallest_panel_sizes() {
    let settable = [MBP, (2336, 1510), (2624, 1696), (2992, 1934)];
    assert_eq!(
        served_mode_sizes(&settable, 3),
        vec![MBP, (2992, 1934), (2624, 1696)]
    );
}

#[test]
fn panel_sizes_of_equal_pixel_count_keep_their_enumeration_order() {
    // Two 16:9 sizes with the same pixel count on a 16:9 desktop.
    let desktop = (3840, 2160);
    let settable = [desktop, (1920, 1080), (1920, 1080)];
    assert_eq!(
        served_mode_sizes(&settable, 15),
        vec![desktop, (1920, 1080), (1920, 1080)]
    );
    let settable = [desktop, (1280, 720), (1600, 900), (960, 540)];
    assert_eq!(
        served_mode_sizes(&settable, 15),
        vec![desktop, (1600, 900), (1280, 720), (960, 540)]
    );
}

#[test]
fn a_bound_of_zero_still_serves_the_desktop() {
    assert_eq!(served_mode_sizes(&[MBP, (640, 480)], 0), vec![MBP]);
}

#[test]
fn an_empty_settable_list_serves_nothing() {
    assert!(served_mode_sizes(&[], 5).is_empty());
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
