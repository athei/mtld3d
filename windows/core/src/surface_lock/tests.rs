use super::*;

#[test]
fn staging_wins_over_every_other_shape() {
    // A lockable render target reads and writes its own bytes, so it never
    // takes the read-back route even though it also owns a colour texture.
    assert_eq!(
        classify_color_surface_lock(true, false),
        ColorSurfaceLock::Staging
    );
    assert_eq!(
        classify_color_surface_lock(true, true),
        ColorSurfaceLock::Staging
    );
}

#[test]
fn the_implicit_back_buffer_is_read_back() {
    assert_eq!(
        classify_color_surface_lock(false, true),
        ColorSurfaceLock::BackBufferReadback
    );
}

#[test]
fn a_non_lockable_render_target_is_rejected() {
    assert_eq!(
        classify_color_surface_lock(false, false),
        ColorSurfaceLock::Reject
    );
}
