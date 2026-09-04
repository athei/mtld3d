//! Unit tests for the `GpuCaps` snapshot constructor and the `intel.*` overrides.
//!
//! `default_matches_apple_silicon` pins `GpuCaps::apple_silicon_default` to what a real Apple
//! Silicon device reports: unified memory and a 16-byte linear-texture alignment floor. The
//! encoder derives the buffer storage mode and the blit-staging `bytes_per_row` floor from those
//! two fields, so a constructor that drifted from the hardware would move both with nothing else
//! complaining.
//!
//! The override tests pin `with_intel_overrides`: each key moves its field to the Mac2 answer and
//! nothing else, a device already at that answer is left alone, a larger device alignment is
//! kept, and no overrides mean no change. Nothing D3D9-visible proves a forced override took, so
//! these are the only place the fold itself is checked.

use super::*;

#[test]
fn default_matches_apple_silicon() {
    let caps = GpuCaps::apple_silicon_default();
    assert!(caps.unified_memory);
    assert_eq!(caps.min_linear_texture_align, 16);
}

#[test]
fn no_overrides_leave_the_device_answer_unchanged() {
    let caps = GpuCaps::apple_silicon_default().with_intel_overrides(false, false);
    assert!(caps.unified_memory);
    assert_eq!(caps.min_linear_texture_align, 16);
}

#[test]
fn managed_memory_clears_unified_memory_only() {
    let caps = GpuCaps::apple_silicon_default().with_intel_overrides(true, false);
    assert!(!caps.unified_memory);
    assert_eq!(caps.min_linear_texture_align, 16);
}

#[test]
fn linear_align256_raises_the_floor_only() {
    let caps = GpuCaps::apple_silicon_default().with_intel_overrides(false, true);
    assert!(caps.unified_memory);
    assert_eq!(caps.min_linear_texture_align, MAC2_LINEAR_TEXTURE_ALIGN);
}

#[test]
fn both_overrides_describe_a_mac2_device() {
    let caps = GpuCaps::apple_silicon_default().with_intel_overrides(true, true);
    assert!(!caps.unified_memory);
    assert_eq!(caps.min_linear_texture_align, 256);
}

#[test]
fn overrides_are_idempotent_on_a_mac2_device() {
    let mac2 = GpuCaps {
        unified_memory: false,
        min_linear_texture_align: 256,
    };
    let caps = mac2.with_intel_overrides(true, true);
    assert!(!caps.unified_memory);
    assert_eq!(caps.min_linear_texture_align, 256);
}

#[test]
fn linear_align256_keeps_a_larger_device_alignment() {
    let wide = GpuCaps {
        unified_memory: true,
        min_linear_texture_align: 512,
    };
    let caps = wide.with_intel_overrides(false, true);
    assert_eq!(caps.min_linear_texture_align, 512);
}
