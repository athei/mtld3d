use mtld3d_types::{
    D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM, D3DUSAGE_AUTOGENMIPMAP,
    D3DUSAGE_DYNAMIC,
};

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

#[test]
fn only_the_managed_pool_is_runtime_managed() {
    assert!(is_runtime_managed(D3DPOOL_MANAGED));
    // Create* rejects the out-of-range values before they reach a resource;
    // classifying them as unmanaged keeps an eviction off a resource whose
    // pixels nothing can replay.
    for pool in [
        D3DPOOL_DEFAULT,
        D3DPOOL_SYSTEMMEM,
        D3DPOOL_SCRATCH,
        4,
        0xFFFF_FFFF,
    ] {
        assert!(!is_runtime_managed(pool), "pool {pool}");
    }
}

#[test]
fn the_managed_and_scratch_pools_conflict_with_dynamic_usage() {
    for pool in [D3DPOOL_MANAGED, D3DPOOL_SCRATCH] {
        assert!(
            usage_conflicts_with_pool(D3DUSAGE_DYNAMIC, pool),
            "pool {pool}"
        );
        // The flag stays the one that conflicts when it arrives beside another.
        assert!(
            usage_conflicts_with_pool(D3DUSAGE_DYNAMIC | D3DUSAGE_AUTOGENMIPMAP, pool),
            "pool {pool}"
        );
    }
}

#[test]
fn the_other_pools_take_dynamic_usage() {
    for pool in [D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM] {
        assert!(
            !usage_conflicts_with_pool(D3DUSAGE_DYNAMIC, pool),
            "pool {pool}"
        );
    }
}

#[test]
fn a_pool_without_dynamic_usage_conflicts_with_nothing() {
    for pool in [
        D3DPOOL_DEFAULT,
        D3DPOOL_MANAGED,
        D3DPOOL_SYSTEMMEM,
        D3DPOOL_SCRATCH,
    ] {
        for usage in [0, D3DUSAGE_AUTOGENMIPMAP] {
            assert!(
                !usage_conflicts_with_pool(usage, pool),
                "pool {pool} usage {usage:#x}"
            );
        }
    }
}
