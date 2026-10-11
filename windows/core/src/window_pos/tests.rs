//! Unit tests for which window position changes can move the client area.

use super::*;

#[test]
fn a_resize_moves_the_client_area() {
    assert!(moves_client_area(0));
    assert!(moves_client_area(
        SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE
    ));
}

#[test]
fn a_recomputed_frame_or_a_show_moves_it_at_one_size() {
    assert!(moves_client_area(
        SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED
    ));
    assert!(moves_client_area(SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW));
}

#[test]
fn a_move_restack_activation_or_hide_alone_does_not() {
    assert!(!moves_client_area(SWP_NOSIZE));
    assert!(!moves_client_area(SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE));
    assert!(!moves_client_area(SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER));
    assert!(!moves_client_area(SWP_NOSIZE | SWP_NOMOVE | SWP_HIDEWINDOW));
}
