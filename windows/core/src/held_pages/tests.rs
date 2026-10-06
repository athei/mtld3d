//! Each holder's gauge follows the pages it holds.
//!
//! The gauges are process-wide and `cargo test` runs tests as threads of one
//! process. `PageHolder::Surface` is created by these tests alone, so they
//! hold `GAUGE` while they assert deltas on it. `PageHolder::EncoderLease`
//! is also created by every test that builds an encoder packet with a
//! retired backing, so its deltas are not asserted here.

use std::sync::{Mutex, PoisonError};

use super::*;
use crate::page_box::PAGE_SIZE;

/// The lock the tests that assert on the surface gauge hold.
///
/// A `static` because the resource is process-wide: the gauge is, and the
/// tests share the process.
static GAUGE: Mutex<()> = Mutex::new(());

#[test]
fn the_gauge_charges_padded_bytes_while_held_and_returns_them_on_drop() {
    let _gauge = GAUGE.lock().unwrap_or_else(PoisonError::into_inner);
    let before = live_surface_bytes();
    let one = HeldPages::new(PageBox::new_uninit(100), PageHolder::Surface);
    assert_eq!(live_surface_bytes() - before, PAGE_SIZE as u64);
    let mut two = HeldPages::new(PageBox::new_zeroed(PAGE_SIZE + 1), PageHolder::Surface);
    assert_eq!(live_surface_bytes() - before, 3 * PAGE_SIZE as u64);
    assert_eq!(two.len(), 2 * PAGE_SIZE);
    assert_eq!(two.logical_len(), PAGE_SIZE + 1);
    two.as_mut_slice()[3] = 7;
    assert_eq!(two.as_slice()[3], 7);
    drop(one);
    assert_eq!(live_surface_bytes() - before, 2 * PAGE_SIZE as u64);
    drop(two);
    assert_eq!(live_surface_bytes(), before);
}

#[test]
fn handing_the_box_back_returns_its_charge() {
    let _gauge = GAUGE.lock().unwrap_or_else(PoisonError::into_inner);
    let before = live_surface_bytes();
    let held = HeldPages::new(PageBox::new_uninit(3 * PAGE_SIZE), PageHolder::Surface);
    let page = held.into_page();
    assert_eq!(live_surface_bytes(), before);
    assert_eq!(page.len(), 3 * PAGE_SIZE);
}

#[test]
fn replacing_the_box_through_deref_mut_returns_the_charged_length() {
    let _gauge = GAUGE.lock().unwrap_or_else(PoisonError::into_inner);
    let before = live_surface_bytes();
    let mut held = HeldPages::new(PageBox::new_uninit(1), PageHolder::Surface);
    *held = PageBox::new_uninit(4 * PAGE_SIZE);
    drop(held);
    assert_eq!(live_surface_bytes(), before);
}
