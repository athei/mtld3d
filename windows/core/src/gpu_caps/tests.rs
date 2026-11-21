//! Unit tests for the `GpuCaps` snapshot constructor.
//!
//! `default_matches_apple_silicon` pins `GpuCaps::apple_silicon_default` to the 16-byte
//! linear-texture alignment floor a real Apple Silicon device reports. The encoder derives the
//! blit-staging `bytes_per_row` floor from that field, so a constructor that drifted from the
//! hardware would move it with nothing else complaining.
//!
//! The resolve-retire test pins `resolve_needs_retire` to the `RESOLVE_NEEDS_RETIRE` bit of the
//! device's answer.
//!
//! The GPU-gate tests pin `unsupported_gpu`, the decision `CreateDevice` makes before it creates
//! anything: an Apple GPU passes whatever else it reports, any other device is refused with a
//! line that names it, and a process with no Metal device is refused too.

use super::*;

#[test]
fn default_matches_apple_silicon() {
    let caps = GpuCaps::apple_silicon_default();
    assert_eq!(caps.min_linear_texture_align, 16);
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
fn an_apple_gpu_passes_the_gate() {
    assert_eq!(
        unsupported_gpu(DeviceCapsFlags::APPLE_GPU, "Apple M2"),
        None
    );
    assert_eq!(
        unsupported_gpu(
            DeviceCapsFlags::APPLE_GPU | DeviceCapsFlags::RESOLVE_NEEDS_RETIRE,
            "Apple Paravirtual device"
        ),
        None,
        "the virtual GPU of a VM on Apple Silicon is an Apple GPU"
    );
}

#[test]
fn any_other_gpu_is_refused_by_name() {
    let line = unsupported_gpu(
        DeviceCapsFlags::SAMPLER_BORDER | DeviceCapsFlags::SAMPLE_COUNT_4,
        "AMD Radeon Pro 5500M",
    )
    .expect("a Mac2-only device is refused");
    assert_eq!(
        line,
        "mtld3d supports only Apple Silicon GPUs; AMD Radeon Pro 5500M is not one"
    );
}

#[test]
fn a_process_without_a_metal_device_is_refused() {
    assert_eq!(
        unsupported_gpu(DeviceCapsFlags::empty(), "").as_deref(),
        Some("mtld3d supports only Apple Silicon GPUs; this process has no Metal device")
    );
}
