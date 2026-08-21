//! Unit test for the `GpuCaps` snapshot constructor.
//!
//! The one test pins `GpuCaps::apple_silicon_default` to what a real Apple Silicon device
//! reports: unified memory and a 16-byte linear-texture alignment floor. The encoder derives the
//! buffer storage mode and the blit-staging `bytes_per_row` floor from those two fields, so a
//! constructor that drifted from the hardware would move both with nothing else complaining.

use super::*;

#[test]
fn default_matches_apple_silicon() {
    let caps = GpuCaps::apple_silicon_default();
    assert!(caps.unified_memory);
    assert_eq!(caps.min_linear_texture_align, 16);
}
