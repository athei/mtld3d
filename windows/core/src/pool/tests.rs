use mtld3d_types::{D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM};

use super::*;

#[test]
fn the_two_system_memory_pools_are_cpu_only() {
    assert!(is_cpu_only(D3DPOOL_SYSTEMMEM));
    assert!(is_cpu_only(D3DPOOL_SCRATCH));
}

#[test]
fn the_gpu_resident_pools_are_not_cpu_only() {
    assert!(!is_cpu_only(D3DPOOL_DEFAULT));
    assert!(!is_cpu_only(D3DPOOL_MANAGED));
}

#[test]
fn an_out_of_range_pool_value_is_not_cpu_only() {
    // Create* rejects these before they reach a resource; classify them with
    // the GPU pools so a stray value never silently skips a Metal allocation.
    for pool in [4u32, 0xFFFF_FFFF] {
        assert!(!is_cpu_only(pool), "pool {pool}");
    }
}
