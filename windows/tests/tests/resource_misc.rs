//! `IUnknown` / `IDirect3DResource9` plumbing.
//!
//! Refcounts, `QueryInterface`, `GetDevice`, `GetType`, no-op methods, caps
//! queries, and stub contracts.

use core::ffi::c_void;

use mtld3d_tests::Harness;
use mtld3d_types::{
    D3D_OK, D3DDECL_END_STREAM, D3DDECLTYPE_FLOAT3, D3DDECLTYPE_UNUSED, D3DDECLUSAGE_POSITION,
    D3DERR_INVALIDCALL, D3DERR_MOREDATA, D3DERR_NOTFOUND, D3DFMT_A8R8G8B8, D3DFMT_D24S8,
    D3DFMT_INDEX16, D3DFVF_XYZ, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SCRATCH,
    D3DQUERYTYPE_EVENT, D3DRTYPE_TEXTURE, D3DSBT_ALL, D3DUSAGE_WRITEONLY, D3DVERTEXELEMENT9,
    E_NOINTERFACE, Guid, IID_IDIRECT3D9, IID_IDIRECT3DDEVICE9, IID_IUNKNOWN,
};

/// `GetPrivateData` as a test reads it: the hr and the size it reported.
type GetPrivateData<'a> = &'a dyn Fn(Option<&mut [u8]>) -> (i32, u32);

/// `vs_2_0 { dcl_position v0; mov oPos, v0 }` and `ps_2_0 { mov oC0, c0 }`.
///
/// The programs are here only to have a shader object to ask; what they
/// compute never reaches a draw.
const VS_PASSTHROUGH: [u32; 8] = [
    0xFFFE_0200,
    0x0200_001F,
    0x8000_0000,
    0x900F_0000,
    0x0200_0001,
    0xC00F_0000,
    0x90E4_0000,
    0x0000_FFFF,
];

const PS_WHITE: [u32; 5] = [
    0xFFFF_0200,
    0x0200_0001,
    0x800F_0800,
    0xA0E4_0000,
    0x0000_FFFF,
];

const POSITION_DECL: [D3DVERTEXELEMENT9; 2] = [
    D3DVERTEXELEMENT9 {
        stream: 0,
        offset: 0,
        type_: D3DDECLTYPE_FLOAT3,
        method: 0,
        usage: D3DDECLUSAGE_POSITION,
        usage_index: 0,
    },
    D3DVERTEXELEMENT9 {
        stream: D3DDECL_END_STREAM,
        offset: 0,
        type_: D3DDECLTYPE_UNUSED,
        method: 0,
        usage: 0,
        usage_index: 0,
    },
];

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

/// The device answers `QueryInterface` for `IUnknown` and `IDirect3DDevice9` with itself.
///
/// One reference stronger. SDKs that are handed the game's device take their own typed reference
/// through `QueryInterface(IID_IDirect3DDevice9)` and treat a failure as an
/// unusable device.
#[test]
fn query_interface_identity_on_device() {
    let h = Harness::new();
    let base = h.device_refcount();
    for iid in [IID_IUNKNOWN, IID_IDIRECT3DDEVICE9] {
        let (hr, same, held) = h.device_query_interface(&iid);
        assert_eq!(hr, D3D_OK, "QueryInterface({:#010x})", iid.data1);
        assert!(same, "the interface is the device object itself");
        assert_eq!(
            held,
            base + 1,
            "QueryInterface hands out one counted reference"
        );
    }
    assert_eq!(
        h.device_refcount(),
        base,
        "releasing the QI references balances"
    );
}

/// The factory answers for `IUnknown` and `IDirect3D9`, and for nothing else.
#[test]
fn query_interface_identity_on_factory() {
    let h = Harness::factory_only();
    for iid in [IID_IUNKNOWN, IID_IDIRECT3D9] {
        let (hr, same, held) = h.factory_query_interface(&iid);
        assert_eq!(hr, D3D_OK, "QueryInterface({:#010x})", iid.data1);
        assert!(same, "the interface is the factory object itself");
        assert_eq!(
            held, 2,
            "the factory's own reference plus the one QI handed out"
        );
    }
    let (hr, same, held) = h.factory_query_interface(&IID_IDIRECT3DDEVICE9);
    assert_eq!(hr, E_NOINTERFACE, "the factory is not a device");
    assert!(!same, "nothing is handed out on a miss");
    assert_eq!(held, 1, "a miss leaves the refcount alone");
}

/// `GetDevice` names the device that created the resource, in every pool.
///
/// MANAGED is covered because a managed resource deliberately does not pin its
/// device: it still came from one, and reporting otherwise would be a wrong
/// answer rather than a missing feature. The returned reference is the
/// caller's, so it is released here and the count is checked back to its
/// pre-call value.
#[test]
fn resource_get_device_returns_the_creating_device() {
    let h = Harness::new();
    // The reference `GetDevice` writes belongs to the caller, so every case
    // releases it and asserts the count is back where it started: a thunk
    // that handed out a pointer without taking a reference fails here rather
    // than under the application's own Release.
    let check = |label: &str, get: &dyn Fn() -> (i32, *mut c_void)| {
        let before = h.device_refcount();
        let (hr, dev) = get();
        assert_eq!(hr, D3D_OK, "{label}: GetDevice");
        assert_eq!(dev, h.device(), "{label}: the device that created it");
        // SAFETY: `dev` is the reference `GetDevice` just handed out.
        let back = unsafe { h.release_device_ref(dev) };
        assert_eq!(
            back, before,
            "{label}: the reference handed out is the one given back"
        );
    };

    // MANAGED and SCRATCH are the pools that deliberately do not pin the
    // device. A resource in one still came from a device, and answering
    // otherwise would be a wrong answer rather than a missing feature.
    for (pool, label) in [
        (D3DPOOL_DEFAULT, "texture (DEFAULT)"),
        (D3DPOOL_MANAGED, "texture (MANAGED)"),
    ] {
        let tex = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, pool);
        check(label, &|| tex.get_device());
    }
    // The cube and volume vtables share the 2D texture's thunk, which reads
    // the wrapper through one type and only holds because the three share a
    // layout. A SCRATCH cube is the CPU-only shell: no Metal texture behind
    // it, and no forwarded reference either.
    for (pool, label) in [
        (D3DPOOL_DEFAULT, "cube (DEFAULT)"),
        (D3DPOOL_SCRATCH, "cube (SCRATCH)"),
    ] {
        let cube = h.create_cube_texture_owned(4, 1, 0, D3DFMT_A8R8G8B8, pool);
        check(label, &|| cube.get_device());
    }

    let (hr, vol) = h.try_create_volume_texture([4, 4, 4], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(hr, D3D_OK, "CreateVolumeTexture");
    let vol = vol.expect("CreateVolumeTexture returned null");
    check("volume texture", &|| vol.get_device());
    // A volume level holds no device of its own; it resolves through the
    // texture that owns it, which is the one path here that answers
    // indirectly.
    let (hr, volume) = vol.get_volume_level(0);
    assert_eq!(hr, D3D_OK, "GetVolumeLevel");
    let volume = volume.expect("GetVolumeLevel returned null");
    check("volume", &|| volume.get_device());

    let vb = h.create_vertex_buffer(64, D3DUSAGE_WRITEONLY, D3DFVF_XYZ, D3DPOOL_DEFAULT);
    check("vertex buffer", &|| vb.get_device());
    let ib = h.create_index_buffer(64, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
    check("index buffer", &|| ib.get_device());

    // A surface belonging to a texture holds no device of its own: it answers
    // through the texture, which is the object the device detaches at
    // teardown. MANAGED is the case that matters, since its texture is not
    // pinned by the surfaces handed out of it.
    for (pool, label) in [
        (D3DPOOL_DEFAULT, "texture surface (DEFAULT)"),
        (D3DPOOL_MANAGED, "texture surface (MANAGED)"),
    ] {
        let tex = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, pool);
        let surf = tex.surface_level(0);
        check(label, &|| surf.get_device());
    }
    // A standalone surface pins the device itself.
    let rt = h.create_render_target(4, 4, D3DFMT_A8R8G8B8);
    check("render target surface", &|| rt.get_device());

    let vs = h.create_vertex_shader(&VS_PASSTHROUGH);
    check("vertex shader", &|| vs.get_device());
    let ps = h.create_pixel_shader(&PS_WHITE);
    check("pixel shader", &|| ps.get_device());

    let decl = h.create_vertex_declaration(&POSITION_DECL);
    check("vertex declaration", &|| decl.get_device());
    let sb = h.create_state_block(D3DSBT_ALL);
    check("state block", &|| sb.get_device());
    let query = h
        .create_query(D3DQUERYTYPE_EVENT)
        .expect("an event query is always available");
    check("query", &|| query.get_device());
}

/// The two ways a `GetDevice` call can be malformed, neither of them fatal.
///
/// An out-param the thunk cannot write is the one early return that must
/// touch nothing, and a null `this` reaches the same thunk through a vtable
/// the caller kept after the object died. Both are answered with
/// `INVALIDCALL`: a fault here would take the application down inside a call
/// that cannot fail on real hardware.
#[test]
fn get_device_answers_a_malformed_call() {
    let h = Harness::new();
    let vb = h.create_vertex_buffer(64, D3DUSAGE_WRITEONLY, D3DFVF_XYZ, D3DPOOL_DEFAULT);

    // SAFETY: a live buffer with no out-param to write.
    let hr = unsafe { vb.get_device_raw(vb.as_ptr(), core::ptr::null_mut()) };
    assert_eq!(hr, D3DERR_INVALIDCALL, "null out-param");

    let mut out: *mut c_void = vb.as_ptr();
    // SAFETY: a writable out-param, and a `this` the thunk must reject.
    let hr = unsafe { vb.get_device_raw(core::ptr::null_mut(), &raw mut out) };
    assert_eq!(hr, D3DERR_INVALIDCALL, "null this");
    assert!(
        out.is_null(),
        "a rejected call clears the out-param rather than leaving the caller's value"
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

/// The two standalone-surface entry points move the reported figure.
///
/// A `CreateRenderTarget` / `CreateDepthStencilSurface` surface owns a real
/// `D3DPOOL_DEFAULT` Metal texture without going through the texture path, so
/// each has to be charged at creation and refunded at release. Both are
/// 2048x2048 at four bytes per pixel, so each is exactly 16 MiB, and the
/// reported figure is in bytes (no rounding to whole MiB).
#[test]
fn available_texture_mem_tracks_standalone_surfaces() {
    const SURFACE_BYTES: u32 = 2048 * 2048 * 4;

    let h = Harness::new();
    let base = h.available_texture_mem();
    assert!(
        base > 2 * SURFACE_BYTES,
        "budget {base} leaves room for both surfaces"
    );
    {
        let _rt = h.create_render_target(2048, 2048, D3DFMT_A8R8G8B8);
        assert_eq!(
            h.available_texture_mem(),
            base - SURFACE_BYTES,
            "a standalone render target costs its own bytes"
        );
        {
            let _ds = h.create_depth_stencil_surface(2048, 2048, D3DFMT_D24S8);
            assert_eq!(
                h.available_texture_mem(),
                base - 2 * SURFACE_BYTES,
                "a standalone depth-stencil surface costs its own bytes"
            );
        }
        assert_eq!(
            h.available_texture_mem(),
            base - SURFACE_BYTES,
            "releasing the depth-stencil surface gives its bytes back"
        );
    }
    assert_eq!(
        h.available_texture_mem(),
        base,
        "releasing the render target gives its bytes back"
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
    // `SetClipPlane`/`GetClipPlane` round-trip here; that the planes reach
    // the GPU is `clip_planes.rs`'s job.
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

/// Assert the blob round trip on one resource.
fn check_private_data(
    label: &str,
    blob: &[u8],
    set: &dyn Fn(&[u8]) -> i32,
    get: GetPrivateData<'_>,
    free: &dyn Fn() -> i32,
) {
    assert_eq!(set(blob), D3D_OK, "{label}: SetPrivateData");

    let (hr, size) = get(None);
    assert_eq!(hr, D3D_OK, "{label}: size query");
    assert_eq!(
        size as usize,
        blob.len(),
        "{label}: the size query reports the stored length"
    );

    let mut out = vec![0u8; blob.len()];
    let (hr, _) = get(Some(&mut out));
    assert_eq!(hr, D3D_OK, "{label}: GetPrivateData");
    assert_eq!(out, blob, "{label}: the blob comes back unchanged");

    assert_eq!(free(), D3D_OK, "{label}: FreePrivateData");
    assert_eq!(
        get(None).0,
        D3DERR_NOTFOUND,
        "{label}: the key is unknown once freed, not merely unreadable"
    );
}

/// Vertex and index buffers hold private data like every other resource.
///
/// The conformance corpus exercises private data on textures and surfaces
/// only, so the buffers' own store is covered here: the blob survives a
/// round-trip, the size query reports its length, and freeing it makes the
/// key unknown again rather than leaving a stale copy behind.
#[test]
fn buffer_private_data_round_trips() {
    let h = Harness::new();
    let guid = Guid {
        data1: 0x1234_5678,
        data2: 0x9abc,
        data3: 0xdef0,
        data4: [7; 8],
    };
    let blob = [0xAAu8, 0xBB, 0xCC, 0xDD, 0xEE];

    let vb = h.create_vertex_buffer(64, D3DUSAGE_WRITEONLY, D3DFVF_XYZ, D3DPOOL_DEFAULT);
    check_private_data(
        "vertex buffer",
        &blob,
        &|b| vb.set_private_data_hr(&guid, b),
        &|o| vb.get_private_data(&guid, o),
        &|| vb.free_private_data_hr(&guid),
    );

    let ib = h.create_index_buffer(64, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
    check_private_data(
        "index buffer",
        &blob,
        &|b| ib.set_private_data_hr(&guid, b),
        &|o| ib.get_private_data(&guid, o),
        &|| ib.free_private_data_hr(&guid),
    );
}

/// The `IUnknown` form of private data holds a real COM reference.
///
/// This is the store's one dangerous path: it `AddRef`s on store and must
/// `Release` on overwrite, on free, and when the resource dies. A leak here
/// keeps the pointed-at object alive forever, and a double release frees an
/// object the application still holds. The device stands in for the
/// application's object because its refcount is observable from a test.
///
/// One buffer kind is enough: both forward to the same store.
#[test]
fn buffer_private_data_holds_a_reference_to_a_stored_iunknown() {
    let h = Harness::new();
    let guid = Guid {
        data1: 0x2222_3333,
        data2: 0x4444,
        data3: 0x5555,
        data4: [1; 8],
    };
    let vb = h.create_vertex_buffer(64, D3DUSAGE_WRITEONLY, D3DFVF_XYZ, D3DPOOL_DEFAULT);

    // An unknown key is NOTFOUND, not an empty success, even for a pure size
    // query: an application that reads before it writes must be able to tell
    // the two apart.
    assert_eq!(
        vb.get_private_data(&guid, None).0,
        D3DERR_NOTFOUND,
        "GetPrivateData before any Set"
    );
    assert_eq!(
        vb.free_private_data_hr(&guid),
        D3DERR_NOTFOUND,
        "FreePrivateData before any Set"
    );

    let before = h.device_refcount();
    assert_eq!(
        vb.set_private_data_unknown(&guid, h.device()),
        D3D_OK,
        "SetPrivateData(D3DSPD_IUNKNOWN)"
    );
    assert_eq!(
        h.device_refcount(),
        before + 1,
        "the store took a reference of its own"
    );

    // Storing over the same key releases what was there.
    assert_eq!(
        vb.set_private_data_unknown(&guid, h.device()),
        D3D_OK,
        "SetPrivateData over an existing key"
    );
    assert_eq!(
        h.device_refcount(),
        before + 1,
        "the overwrite released the previous reference"
    );

    // Reading one out hands the caller a reference of its own.
    let (hr, punk, size) = vb.get_private_data_unknown(&guid);
    assert_eq!(hr, D3D_OK, "GetPrivateData(IUnknown)");
    assert_eq!(punk, h.device(), "the pointer that was stored");
    assert_eq!(
        size as usize,
        size_of::<*mut c_void>(),
        "an IUnknown entry reports pointer width"
    );
    assert_eq!(
        h.device_refcount(),
        before + 2,
        "Get takes a reference for the caller"
    );
    // SAFETY: `punk` is the reference `GetPrivateData` just handed out.
    unsafe { h.release_device_ref(punk) };

    assert_eq!(vb.free_private_data_hr(&guid), D3D_OK, "FreePrivateData");
    assert_eq!(
        h.device_refcount(),
        before,
        "freeing the key released the store's reference"
    );
}

/// A buffer too small to hold the blob reports the size it needs.
///
/// Applications size their buffer from this call, so returning success with a
/// truncated copy would corrupt whatever they store.
#[test]
fn buffer_private_data_reports_the_size_it_needs() {
    let h = Harness::new();
    let guid = Guid {
        data1: 0x6666_7777,
        data2: 0x8888,
        data3: 0x9999,
        data4: [2; 8],
    };
    let blob = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    let vb = h.create_vertex_buffer(64, D3DUSAGE_WRITEONLY, D3DFVF_XYZ, D3DPOOL_DEFAULT);
    assert_eq!(
        vb.set_private_data_hr(&guid, &blob),
        D3D_OK,
        "SetPrivateData"
    );

    let mut small = [0u8; 4];
    let (hr, size) = vb.get_private_data(&guid, Some(&mut small));
    assert_eq!(hr, D3DERR_MOREDATA, "an undersized buffer is MOREDATA");
    assert_eq!(
        size as usize,
        blob.len(),
        "the call reports the size the blob needs"
    );
    assert_eq!(small, [0u8; 4], "a rejected read writes nothing");
}
