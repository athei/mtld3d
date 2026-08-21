use super::{DisplaySync, display_sync_for, present_interval as pi};

#[test]
fn default_and_one_enable_vsync() {
    assert_eq!(display_sync_for(pi::DEFAULT), DisplaySync::On);
    assert_eq!(display_sync_for(pi::ONE), DisplaySync::On);
}

#[test]
fn immediate_disables_vsync() {
    assert_eq!(display_sync_for(pi::IMMEDIATE), DisplaySync::Off);
}

#[test]
fn non_unit_ratios_fall_through_to_display_rate() {
    assert_eq!(display_sync_for(pi::TWO), DisplaySync::Fallthrough);
    assert_eq!(display_sync_for(pi::THREE), DisplaySync::Fallthrough);
    assert_eq!(display_sync_for(pi::FOUR), DisplaySync::Fallthrough);
    assert!(display_sync_for(pi::TWO).enabled());
    assert!(display_sync_for(pi::THREE).enabled());
    assert!(display_sync_for(pi::FOUR).enabled());
}

#[test]
fn unknown_bits_fall_through() {
    assert_eq!(display_sync_for(0x1234_5678), DisplaySync::Fallthrough);
    assert!(display_sync_for(0x1234_5678).enabled());
}

#[test]
fn enabled_polarity() {
    assert!(DisplaySync::On.enabled());
    assert!(!DisplaySync::Off.enabled());
    assert!(DisplaySync::Fallthrough.enabled());
}
