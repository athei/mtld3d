use super::*;

#[test]
fn uma_picks_shared() {
    assert_eq!(buffer_storage_mode(true), StorageMode::Shared);
}

#[test]
fn non_uma_picks_managed() {
    assert_eq!(buffer_storage_mode(false), StorageMode::Managed);
}
