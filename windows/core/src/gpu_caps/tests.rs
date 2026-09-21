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
//!
//! The resolve-retire tests pin `resolve_needs_retire` to the `RESOLVE_NEEDS_RETIRE` bit of the
//! device's answer and pin the overrides to leaving it alone: the forced Intel answers describe
//! an Intel/AMD Mac, whose GPU orders its queue, so they must not turn the wait on.
//!
//! The submission-split tests pin `multisample_read_splits_submission`: it takes both the device
//! bit and a source of more than one sample, so a device without the bit never submits early
//! whatever it reads, a single-sampled read never does on any device, and the forced Intel
//! answers change neither.

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
        device_caps: DeviceCapsFlags::empty(),
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
        device_caps: DeviceCapsFlags::empty(),
    };
    let caps = wide.with_intel_overrides(false, true);
    assert_eq!(caps.min_linear_texture_align, 512);
}

#[test]
fn resolve_retire_follows_the_device_bit() {
    assert!(!GpuCaps::apple_silicon_default().resolve_needs_retire());
    let paravirtual = GpuCaps {
        device_caps: DeviceCapsFlags::RESOLVE_NEEDS_RETIRE | DeviceCapsFlags::SAMPLE_COUNT_4,
        ..GpuCaps::apple_silicon_default()
    };
    assert!(paravirtual.resolve_needs_retire());
    let other_bits = GpuCaps {
        device_caps: DeviceCapsFlags::SAMPLER_BORDER | DeviceCapsFlags::SAMPLE_COUNT_4,
        ..GpuCaps::apple_silicon_default()
    };
    assert!(!other_bits.resolve_needs_retire());
}

#[test]
fn intel_overrides_leave_resolve_retire_alone() {
    let caps = GpuCaps::apple_silicon_default().with_intel_overrides(true, true);
    assert!(!caps.resolve_needs_retire());
    let paravirtual = GpuCaps {
        device_caps: DeviceCapsFlags::RESOLVE_NEEDS_RETIRE,
        ..GpuCaps::apple_silicon_default()
    }
    .with_intel_overrides(true, true);
    assert!(paravirtual.resolve_needs_retire());
}

#[test]
fn a_multisample_read_splits_the_submission_only_on_the_device_bit() {
    let ordered = GpuCaps::apple_silicon_default();
    let paravirtual = GpuCaps {
        device_caps: DeviceCapsFlags::RESOLVE_NEEDS_RETIRE | DeviceCapsFlags::SAMPLE_COUNT_4,
        ..GpuCaps::apple_silicon_default()
    };
    for samples in [0, 1, 2, 4, 8, u8::MAX] {
        assert!(
            !ordered.multisample_read_splits_submission(samples),
            "a device that orders its encoders submits nothing extra at {samples} samples"
        );
        assert_eq!(
            paravirtual.multisample_read_splits_submission(samples),
            samples > 1,
            "{samples} samples on the device that answered the bit"
        );
    }
}

#[test]
fn intel_overrides_leave_the_submission_split_alone() {
    let forced = GpuCaps::apple_silicon_default().with_intel_overrides(true, true);
    assert!(!forced.multisample_read_splits_submission(4));
    let other_bits = GpuCaps {
        device_caps: DeviceCapsFlags::SAMPLER_BORDER | DeviceCapsFlags::SAMPLE_COUNT_4,
        ..GpuCaps::apple_silicon_default()
    }
    .with_intel_overrides(true, true);
    assert!(!other_bits.multisample_read_splits_submission(4));
    let paravirtual = GpuCaps {
        device_caps: DeviceCapsFlags::RESOLVE_NEEDS_RETIRE,
        ..GpuCaps::apple_silicon_default()
    }
    .with_intel_overrides(true, true);
    assert!(paravirtual.multisample_read_splits_submission(4));
    assert!(!paravirtual.multisample_read_splits_submission(1));
}
