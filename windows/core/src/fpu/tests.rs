use mtld3d_types::{D3DCREATE_HARDWARE_VERTEXPROCESSING, D3DCREATE_MULTITHREADED};

use super::*;

#[test]
fn the_default_control_word_becomes_single_precision() {
    assert_eq!(device_control_word(0x027f), 0x007f);
}

#[test]
fn precision_rounding_and_unmasked_exceptions_are_all_reset() {
    // Extended precision, truncation, every exception but precision unmasked.
    assert_eq!(device_control_word(0x0f60), 0x007f);
    // Double precision, round up, every exception masked.
    assert_eq!(device_control_word(0x0a7f), 0x007f);
}

#[test]
fn bits_outside_the_device_fields_keep_their_value() {
    assert_eq!(device_control_word(0x1000), 0x103f);
    assert_eq!(device_control_word(0x0000), 0x003f);
    assert_eq!(device_control_word(0xffff), 0xf0ff);
}

#[test]
fn only_fpu_preserve_keeps_the_control_word() {
    assert!(sets_control_word(0));
    assert!(sets_control_word(
        D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED
    ));
    assert!(!sets_control_word(D3DCREATE_FPU_PRESERVE));
    assert!(!sets_control_word(
        D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE
    ));
}
