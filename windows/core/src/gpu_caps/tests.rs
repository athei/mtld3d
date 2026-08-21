use super::*;

#[test]
fn default_matches_apple_silicon() {
    let caps = GpuCaps::apple_silicon_default();
    assert!(caps.unified_memory);
    assert_eq!(caps.min_linear_texture_align, 16);
}
