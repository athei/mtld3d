use super::{ModeRequest, mode_set_attempts, select_mode_sizes};

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
