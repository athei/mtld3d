//! `IUnknown` / `IDirect3DResource9` plumbing.
//!
//! Refcounts, `QueryInterface`, `GetDevice`, `GetType`, no-op methods, caps
//! queries, and stub contracts.

use mtld3d_tests::Harness;
use mtld3d_types::{
    D3D_OK, D3DERR_INVALIDCALL, D3DFMT_A8R8G8B8, D3DFMT_INDEX16, D3DFVF_XYZ, D3DPOOL_DEFAULT,
    D3DPOOL_MANAGED, D3DQUERYTYPE_EVENT, D3DRTYPE_TEXTURE, D3DSBT_ALL, D3DUSAGE_WRITEONLY,
    E_NOINTERFACE,
};

/// Every child resource forwards exactly one reference to the owning device.
///
/// The reference is held for the child's public lifetime (the D3D9
/// child-refcount model): creating one raises the device refcount by one,
/// releasing it lowers it back. Guards the central `ComChild` forwarding engine
/// against per-type imbalance.
#[test]
fn child_resources_balance_device_refcount() {
    let h = Harness::new();
    let base = h.device_refcount();

    {
        let _vb = h.create_vertex_buffer(64, D3DUSAGE_WRITEONLY, D3DFVF_XYZ, D3DPOOL_DEFAULT);
        assert_eq!(h.device_refcount(), base + 1, "vertex buffer forwards +1");
    }
    assert_eq!(h.device_refcount(), base, "vertex buffer release balances");

    {
        let _ib = h.create_index_buffer(64, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
        assert_eq!(h.device_refcount(), base + 1, "index buffer forwards +1");
    }
    assert_eq!(h.device_refcount(), base, "index buffer release balances");

    {
        let _tex = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
        assert_eq!(h.device_refcount(), base + 1, "texture forwards +1");
    }
    assert_eq!(h.device_refcount(), base, "texture release balances");

    {
        let _rt = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
        assert_eq!(h.device_refcount(), base + 1, "render target forwards +1");
    }
    assert_eq!(h.device_refcount(), base, "render target release balances");

    {
        let _sb = h.create_state_block(D3DSBT_ALL);
        assert_eq!(h.device_refcount(), base + 1, "state block forwards +1");
    }
    assert_eq!(h.device_refcount(), base, "state block release balances");

    if let Some(q) = h.create_query(D3DQUERYTYPE_EVENT) {
        assert_eq!(h.device_refcount(), base + 1, "query forwards +1");
        drop(q);
        assert_eq!(h.device_refcount(), base, "query release balances");
    }
}

/// A `D3DSBT_ALL` state block captures the bound state.
///
/// That includes the implicit FVF vertex declaration (which sits at public
/// refcount 0 in the cache, so the capture's `AddRef` forwards a device
/// reference). Creating then releasing the block must leave the device refcount
/// unchanged — i.e. the captured objects' forwarded references are released
/// with the block. Otherwise the device is left holding references it can never
/// shed, and teardown never reaches a zero refcount.
#[test]
fn state_block_capture_balances_device_refcount() {
    let h = Harness::new();
    // Bind an FVF so the device has a (cached, implicit) vertex declaration for
    // the block to capture.
    assert_eq!(h.set_fvf(D3DFVF_XYZ), 0, "SetFVF");
    let base = h.device_refcount();
    {
        let _sb = h.create_state_block(D3DSBT_ALL);
    }
    assert_eq!(
        h.device_refcount(),
        base,
        "D3DSBT_ALL capture + release leaves the device refcount balanced",
    );
}

#[test]
fn factory_refcount_increments_and_decrements() {
    // factory_only avoids the extra reference a device holds on its factory.
    let h = Harness::factory_only();
    // The factory starts at 1 (Direct3DCreate9); AddRef → 2, Release → 1.
    assert_eq!(h.add_ref_factory(), 2, "AddRef returns the new count");
    assert_eq!(
        h.release_factory(),
        1,
        "Release returns the post-decrement count"
    );
}

#[test]
fn query_interface_unknown_is_rejected() {
    let h = Harness::new();
    assert_eq!(
        h.device_query_interface_unknown(),
        E_NOINTERFACE,
        "QueryInterface for an unknown GUID returns E_NOINTERFACE",
    );
}

#[test]
fn resource_get_device_is_a_stub() {
    let h = Harness::new();
    let tex = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, 0);
    assert_eq!(
        tex.get_device_hr(),
        D3DERR_INVALIDCALL,
        "resource GetDevice is a documented stub"
    );
}

#[test]
fn resource_reports_its_type() {
    let h = Harness::new();
    let tex = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, 0);
    assert_eq!(tex.resource_type(), D3DRTYPE_TEXTURE, "texture GetType");
}

#[test]
fn resource_no_op_methods_are_callable() {
    let h = Harness::new();
    let tex = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, 0);
    // PreLoad / SetPriority are managed-pool hints; on a DEFAULT-pool texture
    // they are no-ops that must not crash. Priority stays 0 (managed-only).
    tex.pre_load();
    assert_eq!(
        tex.set_priority(5),
        0,
        "SetPriority returns the previous priority"
    );
    assert_eq!(
        tex.priority(),
        0,
        "GetPriority stays 0 — priority is managed-only"
    );
}

/// `Get`/`SetPriority` round-trip for `D3DPOOL_MANAGED` resources.
///
/// They stay pinned at `0` for every other pool. D3D9 honours priority only for
/// managed resources — it orders the resource manager's eviction — so
/// `SetPriority` returns the previously stored value and `GetPriority` reads it
/// back; non-managed pools report `0` and discard the write. Covers the two
/// resource types the contract round-trips (texture and vertex buffer); surfaces
/// and render targets are always `0`.
#[test]
fn priority_round_trips_for_managed_resources() {
    let h = Harness::new();

    // Managed texture: stored priority round-trips, SetPriority returns the
    // previous value.
    let managed_tex = h.create_texture(16, 16, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(managed_tex.priority(), 0, "managed texture starts at 0");
    assert_eq!(
        managed_tex.set_priority(1),
        0,
        "SetPriority returns the previous priority (0)"
    );
    assert_eq!(managed_tex.priority(), 1, "GetPriority reads the new value");
    assert_eq!(
        managed_tex.set_priority(2),
        1,
        "SetPriority returns the previous priority (1)"
    );

    // Managed vertex buffer: same round-trip.
    let managed_vb = h.create_vertex_buffer(256, 0, D3DFVF_XYZ, D3DPOOL_MANAGED);
    assert_eq!(
        managed_vb.priority(),
        0,
        "managed vertex buffer starts at 0"
    );
    assert_eq!(
        managed_vb.set_priority(1),
        0,
        "SetPriority returns the previous priority (0)"
    );
    assert_eq!(managed_vb.priority(), 1, "GetPriority reads the new value");

    // Non-managed resources never store a priority: GetPriority is 0 and
    // SetPriority returns 0 (the discarded previous value).
    let default_tex = h.create_texture(16, 16, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(default_tex.priority(), 0, "non-managed texture stays at 0");
    assert_eq!(
        default_tex.set_priority(1),
        0,
        "non-managed SetPriority returns 0 and discards the write"
    );
    assert_eq!(
        default_tex.priority(),
        0,
        "non-managed GetPriority remains 0 after a write"
    );
}

#[test]
fn available_texture_mem_is_nonzero() {
    let h = Harness::new();
    assert!(
        h.available_texture_mem() > 0,
        "GetAvailableTextureMem reports memory"
    );
}

#[test]
fn evict_managed_resources_succeeds() {
    let h = Harness::new();
    assert_eq!(
        h.evict_managed_resources(),
        0,
        "EvictManagedResources is a successful no-op"
    );
}

#[test]
fn validate_device_succeeds_and_clip_plane_round_trips() {
    let h = Harness::new();
    // ValidateDevice reports the current state as single-pass valid: Metal
    // validates pipeline state at PSO-creation time, so every state we accept
    // renders in one pass.
    assert_eq!(h.validate_device_hr(), 0, "ValidateDevice → S_OK");
    // SetClipPlane/GetClipPlane are a CPU state round-trip (no GPU application).
    // An unset plane reads back zero; a set plane reads back exactly.
    assert_eq!(
        h.get_clip_plane(0),
        (D3D_OK, [0.0; 4]),
        "GetClipPlane(0) before any set → S_OK + zero"
    );
    let plane = [2.0f32, 8.0, 5.0, 3.0];
    assert_eq!(h.set_clip_plane(3, plane), D3D_OK, "SetClipPlane(3) → S_OK");
    assert_eq!(
        h.get_clip_plane(3),
        (D3D_OK, plane),
        "GetClipPlane(3) returns the set coefficients"
    );
}

#[test]
fn set_gamma_ramp_is_a_safe_no_op() {
    let h = Harness::new();
    // SetGammaRamp is a no-op (Wine/Metal handle gamma); it must not crash.
    h.set_gamma_ramp_noop();
}

#[test]
fn legacy_feature_stub_contracts() {
    // Raster/clip status and dialog-box mode remain unimplemented legacy
    // features; pin their INVALIDCALL contracts. Texture private data is now
    // implemented (a GUID-keyed store, like surfaces). SetPaletteEntries
    // succeeds-and-ignores the palette per D3D9, EXCEPT that without
    // D3DPTEXTURECAPS_ALPHAPALETTE (the default caps set) every entry's peFlags
    // must be 0xFF — the harness passes all-zero entries, so it is INVALIDCALL.
    let h = Harness::new();
    let tex = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, 0);
    assert_eq!(
        tex.set_private_data_hr(),
        D3D_OK,
        "SetPrivateData stores a blob"
    );
    assert_eq!(
        h.set_palette_entries_hr(),
        D3DERR_INVALIDCALL,
        "SetPaletteEntries rejects alpha entries without ALPHAPALETTE"
    );
    assert_eq!(
        h.get_raster_status_hr(),
        D3DERR_INVALIDCALL,
        "GetRasterStatus stub"
    );
    assert_eq!(
        h.get_clip_status_hr(),
        D3DERR_INVALIDCALL,
        "GetClipStatus stub"
    );
    assert_eq!(
        h.set_dialog_box_mode_hr(),
        D3DERR_INVALIDCALL,
        "SetDialogBoxMode stub"
    );
}
