//! D3D9 sub-resource identity: one wrapper per texture level, face and volume level.
//!
//! `GetSurfaceLevel`, `GetCubeMapSurface` and `GetVolumeLevel` each hand back
//! the *same* interface pointer every call, one reference stronger, and that
//! object outlives the application's last `Release` of it (the container owns
//! it). Applications rely on it two ways: comparing a cached pointer against
//! what a getter returns, and hanging per-sub-resource private data off it.
//! Both need the wrapper to be one object with one identity.
//!
//! The bound-sub-resource test at the end covers the lifetime half: a level
//! surface handed to `SetRenderTarget` reads its extent, format and device
//! *through* its container, so the container has to outlive the binding even
//! once the application has released both.

use mtld3d_tests::{Harness, Surface};
use mtld3d_types::{
    D3D_OK, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DUSAGE_RENDERTARGET, Guid,
};

/// A distinct private-data key per test, so one test never reads another's blob.
const fn key(data1: u32) -> Guid {
    Guid {
        data1,
        data2: 0x5ab0,
        data3: 0x1efe,
        data4: [9; 8],
    }
}

#[test]
fn texture_level_surface_has_one_identity() {
    let h = Harness::new();
    let tex = h.create_texture(64, 64, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);

    let s1 = tex.surface_level(0);
    let s2 = tex.surface_level(0);
    assert_eq!(
        s1.as_ptr(),
        s2.as_ptr(),
        "GetSurfaceLevel(0) must return the one cached sub-resource every call"
    );

    // A different level is a different sub-resource, so a different object.
    let other = tex.surface_level(1);
    assert_ne!(
        s1.as_ptr(),
        other.as_ptr(),
        "two levels are two sub-resources"
    );
    let (hr, desc) = other.desc();
    assert_eq!(hr, D3D_OK, "GetDesc on level 1");
    assert_eq!((desc.width, desc.height), (32, 32), "level 1 extent");
}

#[test]
fn texture_level_surface_survives_its_own_refcount_zero() {
    let h = Harness::new();
    let tex = h.create_texture(64, 64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);

    // Take the pointer, then release every reference the application holds to it.
    let cached = {
        let surf = tex.surface_level(0);
        surf.as_ptr()
    };

    // Container-owned: the wrapper is not freed at refcount 0, so re-acquiring
    // returns the very same object and it is still usable.
    let again = tex.surface_level(0);
    assert_eq!(
        again.as_ptr(),
        cached,
        "a level surface must persist past refcount 0"
    );
    let (hr, desc) = again.desc();
    assert_eq!(hr, D3D_OK, "GetDesc on the re-acquired level surface");
    assert_eq!((desc.width, desc.height), (64, 64));
}

#[test]
fn cube_face_surface_has_one_identity() {
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(32, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);

    let f2a = cube.surface(2, 0);
    let f2b = cube.surface(2, 0);
    assert_eq!(
        f2a.as_ptr(),
        f2b.as_ptr(),
        "GetCubeMapSurface(2, 0) must return one cached sub-resource"
    );

    let f3 = cube.surface(3, 0);
    assert_ne!(f2a.as_ptr(), f3.as_ptr(), "two faces are two sub-resources");

    // Released to zero and re-acquired: same object, still describable.
    let cached = f2a.as_ptr();
    drop(f2a);
    drop(f2b);
    let again = cube.surface(2, 0);
    assert_eq!(again.as_ptr(), cached, "a face surface persists past zero");
    let (hr, desc) = again.desc();
    assert_eq!(hr, D3D_OK, "GetDesc on the re-acquired face");
    assert_eq!((desc.width, desc.height), (32, 32));
}

#[test]
fn volume_level_has_one_identity() {
    let h = Harness::new();
    let (hr, tex) = h.try_create_volume_texture([4, 4, 4], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(hr, D3D_OK, "CreateVolumeTexture");
    let tex = tex.expect("volume texture");

    let (hr, v1) = tex.get_volume_level(0);
    assert_eq!(hr, D3D_OK, "GetVolumeLevel");
    let v1 = v1.expect("volume level");
    let (hr, v2) = tex.get_volume_level(0);
    assert_eq!(hr, D3D_OK, "GetVolumeLevel again");
    let v2 = v2.expect("volume level");
    assert_eq!(
        v1.as_ptr(),
        v2.as_ptr(),
        "GetVolumeLevel(0) must return one cached sub-resource"
    );

    let cached = v1.as_ptr();
    drop(v1);
    drop(v2);
    let (hr, again) = tex.get_volume_level(0);
    assert_eq!(hr, D3D_OK, "GetVolumeLevel after release to zero");
    let again = again.expect("volume level");
    assert_eq!(
        again.as_ptr(),
        cached,
        "a volume level persists past refcount 0"
    );
    let (hr, desc) = again.desc();
    assert_eq!(hr, D3D_OK, "IDirect3DVolume9::GetDesc");
    assert_eq!((desc.width, desc.height, desc.depth), (4, 4, 4));
}

#[test]
fn level_surface_private_data_round_trips_across_handles() {
    let h = Harness::new();
    let guid = key(0x0001_0000);
    let blob = [0xAAu8, 0xBB, 0xCC, 0xDD];
    let tex = h.create_texture(16, 16, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);

    let writer = tex.surface_level(0);
    assert_eq!(
        writer.set_private_data_hr(&guid, &blob),
        D3D_OK,
        "SetPrivateData on a level surface"
    );

    // A second handle for the same level is the same object, so it reads the
    // blob the first one stored.
    let reader = tex.surface_level(0);
    let mut out = [0u8; 4];
    let (hr, size) = reader.get_private_data(&guid, Some(&mut out));
    assert_eq!(hr, D3D_OK, "GetPrivateData through a second handle");
    assert_eq!(size, 4, "the stored size");
    assert_eq!(out, blob, "the stored bytes");

    assert_eq!(
        reader.free_private_data_hr(&guid),
        D3D_OK,
        "FreePrivateData"
    );
    assert_ne!(
        writer.get_private_data(&guid, None).0,
        D3D_OK,
        "the freed key is gone for both handles"
    );
}

#[test]
fn volume_level_private_data_round_trips_across_handles() {
    let h = Harness::new();
    let guid = key(0x0002_0000);
    let blob = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66];
    let (hr, tex) = h.try_create_volume_texture([4, 4, 4], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(hr, D3D_OK, "CreateVolumeTexture");
    let tex = tex.expect("volume texture");

    let (hr, writer) = tex.get_volume_level(0);
    assert_eq!(hr, D3D_OK, "GetVolumeLevel");
    let writer = writer.expect("volume level");
    assert_eq!(
        writer.set_private_data_hr(&guid, &blob),
        D3D_OK,
        "SetPrivateData on a volume level"
    );

    let (hr, reader) = tex.get_volume_level(0);
    assert_eq!(hr, D3D_OK, "GetVolumeLevel again");
    let reader = reader.expect("volume level");
    let mut out = [0u8; 6];
    let (hr, size) = reader.get_private_data(&guid, Some(&mut out));
    assert_eq!(hr, D3D_OK, "GetPrivateData through a second handle");
    assert_eq!(size, 6, "the stored size");
    assert_eq!(out, blob, "the stored bytes");

    assert_eq!(
        reader.free_private_data_hr(&guid),
        D3D_OK,
        "FreePrivateData"
    );
}

#[test]
fn level_surface_handles_balance_the_container_refcount() {
    let h = Harness::new();
    let tex = h.create_texture(16, 16, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    let base = tex.refcount();

    // A sub-resource's count IS its container's, so each handle the getter hands
    // out is one reference on the texture, cache hit or not.
    let first = tex.surface_level(0);
    assert_eq!(
        tex.refcount(),
        base + 1,
        "the creating GetSurfaceLevel takes one container reference"
    );
    let second = tex.surface_level(0);
    assert_eq!(
        tex.refcount(),
        base + 2,
        "a cache hit takes one container reference too"
    );

    drop(second);
    assert_eq!(
        tex.refcount(),
        base + 1,
        "releasing one handle gives one back"
    );
    drop(first);
    assert_eq!(
        tex.refcount(),
        base,
        "both handles released leaves the container where it started"
    );
}

#[test]
fn a_bound_level_surface_outlives_its_container() {
    let h = Harness::new();

    // Hand the device the only surviving reference to a render-target level,
    // then release the surface AND its container. The device holds it in a bound
    // slot, and every accessor on it reads through the container, so releasing
    // the container must not free it out from under the binding.
    let bound = {
        let tex = h.create_texture(
            64,
            64,
            1,
            D3DUSAGE_RENDERTARGET,
            D3DFMT_A8R8G8B8,
            D3DPOOL_DEFAULT,
        );
        let surf = tex.surface_level(0);
        let raw = surf.as_ptr();
        assert_eq!(
            h.set_render_target(0, &surf),
            D3D_OK,
            "SetRenderTarget(0, level surface)"
        );
        drop(surf); // the application's last public reference to the level
        drop(tex); // and to its container
        raw
    };

    // A non-owning view: the device's bound-slot reference is the only one left.
    let view = Surface::from_raw(bound);
    let (hr, desc) = view.desc();
    assert_eq!(
        hr, D3D_OK,
        "GetDesc on a bound level surface past its container's release"
    );
    assert_eq!(
        (desc.width, desc.height),
        (64, 64),
        "the extent still resolves through the container"
    );
    let (hr, dev) = view.get_device();
    assert_eq!(hr, D3D_OK, "GetDevice answers through the container");
    assert_eq!(dev, h.device(), "the creating device");
    // SAFETY: `dev` is the reference `GetDevice` just handed out.
    unsafe { h.release_device_ref(dev) };
    let _ = view.into_raw();

    // Rebinding the back buffer drops the last reference: surface and container
    // are freed here, and a Present afterwards proves the device is intact.
    assert_eq!(
        h.set_render_target(0, &h.back_buffer(0)),
        D3D_OK,
        "SetRenderTarget(0, backbuffer)"
    );
    assert_eq!(h.present(), D3D_OK, "Present after the rebind");
}
