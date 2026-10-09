//! The resource rules an extended device answers differently.
//!
//! An extended device refuses `D3DPOOL_MANAGED` for every kind of resource a
//! plain device creates there, and both kinds of device take
//! `D3DPOOL_MANAGED_EX` as the managed pool: a texture and a vertex buffer
//! created there report `D3DPOOL_MANAGED`, lock, and give back what was
//! written. A `pSharedHandle` on a single-level
//! system-memory texture or offscreen plain surface is user memory, copied
//! into the level once at its row pitch; any other shape is an invalid call,
//! a system-memory buffer takes none, and a shared default-pool resource is
//! not available. A plain device refuses every shared handle with
//! `E_NOTIMPL`. The extended surface creates take only the restriction
//! usages, the texture memory figure does not shrink, the frame latency is
//! stored and reported, and Reset takes FLIPEX and 30 back buffers. Source's
//! upload path, `UpdateSurface` from system memory into a default-pool DXT
//! texture and a cube face, works on it, and so do the extended extras:
//! `StretchRect` between whole default-pool textures, `SetPriority` on the
//! default pool, and no ATI2 plain surface.

use core::ffi::c_void;

use mtld3d_tests::{
    CubeTexture, Factory, Harness, HarnessConfig, IndexBuffer, SharedHandle, Surface, Texture,
    TexturedVertex, VertexBuffer, assert_pixel_approx, assert_pixel_eq,
};
use mtld3d_types::{
    D3D_OK, D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DFMT_A8R8G8B8, D3DFMT_ATI2, D3DFMT_D24S8,
    D3DFMT_D32_LOCKABLE, D3DFMT_DXT1, D3DFMT_INDEX16, D3DFMT_L8, D3DFMT_S8_LOCKABLE,
    D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DLOCK_READONLY, D3DPOOL_DEFAULT,
    D3DPOOL_MANAGED, D3DPOOL_MANAGED_EX, D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST,
    D3DRS_LIGHTING, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DSWAPEFFECT_FLIPEX, D3DTEXF_NONE,
    D3DTEXF_POINT, D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_RENDERTARGET, D3DUSAGE_RESTRICT_SHARED_RESOURCE,
    D3DUSAGE_RESTRICTED_CONTENT, E_NOTIMPL,
};

const RED: u32 = 0xFFFF_0000;
const BLUE: u32 = 0xFF00_00FF;

fn extended() -> Harness {
    Harness::create(&HarnessConfig {
        factory: Factory::Extended,
        ..HarnessConfig::default()
    })
}

/// Draw `texture` over the whole back buffer and read its centre.
fn sample_center(h: &Harness, texture: &Texture<'_>) -> u32 {
    const W: u32 = 0xFFFF_FFFF;
    let quad = [
        (-1.0, 1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 0.0),
        (-1.0, -1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0, 0.0),
        (1.0, -1.0, 1.0, 1.0),
        (-1.0, -1.0, 0.0, 1.0),
    ]
    .map(|(x, y, u, v)| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: W,
        u,
        v,
    });
    assert_eq!(h.set_texture(0, texture), D3D_OK, "SetTexture");
    h.select_texture_stage(0);
    assert_eq!(
        h.set_sampler_state(0, D3DSAMP_MINFILTER, D3DTEXF_POINT),
        D3D_OK
    );
    assert_eq!(
        h.set_sampler_state(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        D3D_OK
    );
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), D3D_OK);
    h.render_once(0xFF00_0000, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            D3D_OK,
            "draw"
        );
    });
    h.read_pixel(320, 240)
}

/// The rows a lock maps, `row_bytes` of each, `rows` of them, at the lock's pitch.
fn locked_rows(surface: &Surface<'_>, row_bytes: usize, rows: usize) -> (i32, Vec<Vec<u8>>) {
    let lock = surface.lock_rect(D3DLOCK_READONLY);
    let pitch = lock.pitch();
    let stride = usize::try_from(pitch).expect("positive pitch");
    let bytes = lock.as_u8((rows - 1) * stride + row_bytes).to_vec();
    drop(lock);
    let rows = (0..rows)
        .map(|row| bytes[row * stride..row * stride + row_bytes].to_vec())
        .collect();
    (pitch, rows)
}

#[test]
fn an_extended_device_refuses_the_managed_pool_for_every_kind() {
    let ex = extended();
    let plain = Harness::new();
    for (h, expected) in [(&ex, D3DERR_INVALIDCALL), (&plain, D3D_OK)] {
        let (hr, texture) = h.try_create_texture(16, 16, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
        assert_eq!(hr, expected, "texture");
        if !texture.is_null() {
            drop(Texture::from_raw(texture));
        }
        let (hr, cube) = h.try_create_cube_texture(16, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
        assert_eq!(hr, expected, "cube texture");
        if !cube.is_null() {
            drop(CubeTexture::from_raw(cube));
        }
        let (hr, volume) =
            h.try_create_volume_texture([8, 8, 4], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
        assert_eq!(hr, expected, "volume texture");
        drop(volume);
        let (hr, vb) = h.try_create_vertex_buffer(64, 0, 0, D3DPOOL_MANAGED);
        assert_eq!(hr, expected, "vertex buffer");
        if !vb.is_null() {
            drop(VertexBuffer::from_raw(vb));
        }
        let (hr, ib) = h.try_create_index_buffer(64, 0, D3DFMT_INDEX16, D3DPOOL_MANAGED);
        assert_eq!(hr, expected, "index buffer");
        if !ib.is_null() {
            drop(IndexBuffer::from_raw(ib));
        }
    }
}

#[test]
fn the_managed_ex_pool_is_the_managed_pool_on_either_kind_of_device() {
    for h in [extended(), Harness::new()] {
        let (hr, texture) = h.try_create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED_EX);
        assert_eq!(hr, D3D_OK, "a texture in D3DPOOL_MANAGED_EX");
        let texture = Texture::from_raw(texture);
        let (hr, desc) = texture.level_desc(0);
        assert_eq!(hr, D3D_OK, "GetLevelDesc");
        assert_eq!(desc.pool, D3DPOOL_MANAGED, "the texture is a managed one");
        let mut lock = texture.lock_rect(0, 0);
        lock.write_u32_rect(4, 4, &[RED; 16]);
        drop(lock);
        assert_pixel_eq(
            sample_center(&h, &texture),
            RED,
            "the locked write is what the texture samples",
        );

        let (hr, vb) = h.try_create_vertex_buffer(64, 0, 0, D3DPOOL_MANAGED_EX);
        assert_eq!(hr, D3D_OK, "a vertex buffer in D3DPOOL_MANAGED_EX");
        let vb = VertexBuffer::from_raw(vb);
        let (hr, desc) = vb.desc();
        assert_eq!(hr, D3D_OK, "GetDesc");
        assert_eq!(desc.pool, D3DPOOL_MANAGED, "the buffer is a managed one");
        let words: [u32; 16] =
            core::array::from_fn(|i| 0xC0DE_0000 | u32::try_from(i).expect("small"));
        let mut lock = vb.lock(0, 64, 0);
        lock.write(&words);
        drop(lock);
        let lock = vb.lock(0, 64, D3DLOCK_READONLY);
        assert_eq!(
            lock.read::<u32>(16),
            words,
            "the buffer gives back what was written"
        );
    }
}

#[test]
fn user_memory_seeds_a_system_memory_texture_once_at_its_pitch() {
    let h = extended();
    // L8 at 33 texels: 33 bytes a row in the application's memory, a ramp
    // from 0 to 255 across each row.
    let mut ramp: Vec<u8> = (0..33u32 * 33)
        .map(|i| u8::try_from(i % 33 * 255 / 32).expect("a ramp byte"))
        .collect();
    let mut data = ramp.as_mut_ptr();
    // SAFETY: `data` points at `ramp`, a whole 33x33 L8 level, alive across the create.
    let handle = unsafe { SharedHandle::to(&mut data) };
    let (hr, texture) =
        h.try_create_texture_shared((33, 33), 1, D3DFMT_L8, D3DPOOL_SYSTEMMEM, &handle);
    assert_eq!(hr, D3D_OK, "an L8 texture over user memory");
    let texture = texture.expect("a texture");
    ramp.fill(0);
    let (pitch, rows) = locked_rows(&texture.surface_level(0), 33, 33);
    assert!(pitch >= 33, "the lock pitch holds a row");
    let expected: Vec<u8> = (0..33u32)
        .map(|x| u8::try_from(x * 255 / 32).expect("a ramp byte"))
        .collect();
    for (y, row) in rows.iter().enumerate() {
        assert_eq!(
            row, &expected,
            "row {y} is the ramp, copied when the texture was made"
        );
    }

    let target = h.create_texture(33, 33, 1, 0, D3DFMT_L8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_texture_hr(&texture, &target),
        D3D_OK,
        "UpdateTexture"
    );
    // The paravirtual device samples a swizzle view through the base
    // texture's lanes, so there only red and alpha carry the luminance; the
    // copied bytes are what this checks, not the replication.
    let (pixel, expected) = if h.device_is_paravirtual() {
        (sample_center(&h, &target) & 0xFFFF_0000, 0xFF7F_0000)
    } else {
        (sample_center(&h, &target), 0xFF7F_7F7F)
    };
    assert_pixel_approx(pixel, expected, 2, "the ramp's middle");
}

#[test]
fn user_memory_seeds_a_system_memory_offscreen_plain_surface() {
    let h = extended();
    let mut texels: Vec<u32> = (0..16u32 * 4).map(|i| 0xFF00_0000 | i).collect();
    let mut data = texels.as_mut_ptr();
    // SAFETY: `data` points at `texels`, a whole 16x4 A8R8G8B8 surface, alive
    // across both creates.
    let handle = unsafe { SharedHandle::to(&mut data) };
    let (hr, plain) = h.try_create_offscreen_plain_surface_shared(
        (16, 4),
        D3DFMT_A8R8G8B8,
        D3DPOOL_SYSTEMMEM,
        &handle,
    );
    assert_eq!(hr, D3D_OK, "CreateOffscreenPlainSurface over user memory");
    let plain = plain.expect("a surface");
    let (_, rows) = locked_rows(&plain, 64, 4);
    let bytes: Vec<u8> = texels.iter().flat_map(|t| t.to_le_bytes()).collect();
    for (y, row) in rows.iter().enumerate() {
        assert_eq!(row.as_slice(), &bytes[y * 64..(y + 1) * 64], "row {y}");
    }

    let (hr, plain_ex) = h.create_offscreen_plain_surface_ex(
        (16, 4),
        D3DFMT_A8R8G8B8,
        D3DPOOL_SYSTEMMEM,
        &handle,
        0,
    );
    assert_eq!(hr, D3D_OK, "CreateOffscreenPlainSurfaceEx over user memory");
    let (_, rows) = locked_rows(plain_ex.as_ref().expect("a surface"), 64, 1);
    assert_eq!(rows[0].as_slice(), &bytes[..64], "its first row");
}

#[test]
fn user_memory_outside_its_one_shape_is_refused() {
    let h = extended();
    let mut backing = vec![0u32; 128 * 128];
    let mut data = backing.as_mut_ptr();
    // SAFETY: `data` points at `backing`, a 128x128 A8R8G8B8 level, the
    // largest any create below asks for, alive across all of them.
    let handle = &unsafe { SharedHandle::to(&mut data) };
    for (size, levels, pool, what) in [
        ((128, 128), 0, D3DPOOL_SYSTEMMEM, "a full chain"),
        ((1, 1), 0, D3DPOOL_SYSTEMMEM, "a full chain of one level"),
        ((128, 128), 2, D3DPOOL_SYSTEMMEM, "two levels"),
        ((128, 128), 1, D3DPOOL_SCRATCH, "the scratch pool"),
    ] {
        let (hr, texture) =
            h.try_create_texture_shared(size, levels, D3DFMT_A8R8G8B8, pool, handle);
        assert_eq!(
            (hr, texture.is_none()),
            (D3DERR_INVALIDCALL, true),
            "{what}"
        );
    }
    let (hr, _) = h.try_create_cube_texture_shared(2, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, handle);
    assert_eq!(hr, D3DERR_INVALIDCALL, "a cube texture");
    let (hr, _) =
        h.try_create_volume_texture_shared([2, 2, 2], D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, handle);
    assert_eq!(hr, D3DERR_INVALIDCALL, "a volume texture");
    let (hr, _) = h.try_create_vertex_buffer_shared(16, D3DPOOL_SYSTEMMEM, handle);
    assert_eq!(hr, D3DERR_NOTAVAILABLE, "a vertex buffer");
    let (hr, _) = h.try_create_index_buffer_shared(16, D3DPOOL_SYSTEMMEM, handle);
    assert_eq!(hr, D3DERR_NOTAVAILABLE, "an index buffer");
    let (hr, _) = h.try_create_offscreen_plain_surface_shared(
        (128, 128),
        D3DFMT_A8R8G8B8,
        D3DPOOL_SCRATCH,
        handle,
    );
    assert_eq!(hr, D3DERR_INVALIDCALL, "a scratch offscreen plain");
    let (hr, _) = h.create_offscreen_plain_surface_ex(
        (128, 128),
        D3DFMT_A8R8G8B8,
        D3DPOOL_SCRATCH,
        handle,
        0,
    );
    assert_eq!(
        hr, D3DERR_INVALIDCALL,
        "an extended scratch offscreen plain"
    );
}

/// A create's answer reduced to its hr and whether it handed out no object.
fn refusal<T>((hr, object): (i32, Option<T>)) -> (i32, bool) {
    (hr, object.is_none())
}

/// Every shared-handle create on `h` in `pool`, each reduced by [`refusal`].
fn shared_creates(h: &Harness, pool: u32, handle: &SharedHandle<'_>) -> [(i32, bool); 6] {
    [
        refusal(h.try_create_texture_shared((16, 16), 1, D3DFMT_A8R8G8B8, pool, handle)),
        refusal(h.try_create_cube_texture_shared(16, D3DFMT_A8R8G8B8, pool, handle)),
        refusal(h.try_create_volume_texture_shared([4, 4, 4], D3DFMT_A8R8G8B8, pool, handle)),
        refusal(h.try_create_vertex_buffer_shared(16, pool, handle)),
        refusal(h.try_create_index_buffer_shared(16, pool, handle)),
        refusal(h.try_create_offscreen_plain_surface_shared(
            (16, 16),
            D3DFMT_A8R8G8B8,
            pool,
            handle,
        )),
    ]
}

/// The two shared-handle creates that take no pool, each reduced by [`refusal`].
fn shared_target_creates(h: &Harness, handle: &SharedHandle<'_>) -> [(i32, bool); 2] {
    [
        refusal(h.try_create_render_target_shared((16, 16), D3DFMT_A8R8G8B8, handle)),
        refusal(h.try_create_depth_stencil_surface_shared((16, 16), D3DFMT_D24S8, handle)),
    ]
}

#[test]
fn a_plain_device_refuses_every_shared_handle_with_e_notimpl() {
    let h = Harness::new();
    let mut slot: *mut c_void = core::ptr::null_mut();
    // SAFETY: the slot holds null, a request to share, so nothing is read through it.
    let handle = &unsafe { SharedHandle::to(&mut slot) };
    for pool in [D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM] {
        for (index, answer) in shared_creates(&h, pool, handle).into_iter().enumerate() {
            assert_eq!(answer, (E_NOTIMPL, true), "create {index}, pool {pool}");
        }
    }
    for (index, answer) in shared_target_creates(&h, handle).into_iter().enumerate() {
        assert_eq!(answer, (E_NOTIMPL, true), "target create {index}");
    }
}

#[test]
fn a_shared_default_pool_resource_is_not_available_on_an_extended_device() {
    let h = extended();
    let mut slot: *mut c_void = core::ptr::null_mut();
    // SAFETY: the slot holds null, a request to share, so nothing is read through it.
    let handle = &unsafe { SharedHandle::to(&mut slot) };
    for (index, answer) in shared_creates(&h, D3DPOOL_DEFAULT, handle)
        .into_iter()
        .enumerate()
    {
        assert_eq!(answer, (D3DERR_NOTAVAILABLE, true), "create {index}");
    }
    for (index, answer) in shared_target_creates(&h, handle).into_iter().enumerate() {
        assert_eq!(answer, (D3DERR_NOTAVAILABLE, true), "target create {index}");
    }
}

#[test]
fn the_extended_surface_creates_take_only_the_restriction_usages() {
    let h = extended();
    let none = &SharedHandle::NONE;
    let (hr, rt) = h.create_render_target_ex((16, 16), D3DFMT_A8R8G8B8, none, 0);
    assert_eq!((hr, rt.is_some()), (D3D_OK, true), "CreateRenderTargetEx");
    let (hr, rt) =
        h.create_render_target_ex((16, 16), D3DFMT_A8R8G8B8, none, D3DUSAGE_RENDERTARGET);
    assert_eq!(
        (hr, rt.is_none()),
        (D3DERR_INVALIDCALL, true),
        "even the implied usage"
    );
    let (hr, restricted) =
        h.create_render_target_ex((16, 16), D3DFMT_A8R8G8B8, none, D3DUSAGE_RESTRICTED_CONTENT);
    assert_eq!(hr, D3D_OK, "restricted content");
    let (hr, desc) = restricted.expect("a surface").desc();
    assert_eq!(hr, D3D_OK);
    assert_eq!(
        desc.usage,
        D3DUSAGE_RENDERTARGET | D3DUSAGE_RESTRICTED_CONTENT,
        "GetDesc reports the extended usage"
    );
    assert_eq!(
        h.create_depth_stencil_surface_ex_slot(D3DUSAGE_DEPTHSTENCIL),
        (D3DERR_INVALIDCALL, true),
        "a refused usage leaves the out slot as it was"
    );
    let (hr, _) = h.create_render_target_ex(
        (16, 16),
        D3DFMT_A8R8G8B8,
        none,
        D3DUSAGE_RESTRICT_SHARED_RESOURCE,
    );
    assert_eq!(
        hr, D3DERR_INVALIDCALL,
        "a sharing restriction with nothing to share"
    );
    let (hr, ds) = h.create_depth_stencil_surface_ex((16, 16), D3DFMT_D24S8, none, 0);
    assert_eq!(
        (hr, ds.is_some()),
        (D3D_OK, true),
        "CreateDepthStencilSurfaceEx"
    );
    let (hr, _) =
        h.create_depth_stencil_surface_ex((16, 16), D3DFMT_D24S8, none, D3DUSAGE_DEPTHSTENCIL);
    assert_eq!(hr, D3DERR_INVALIDCALL, "even the implied usage");
    let (hr, plain) =
        h.create_offscreen_plain_surface_ex((16, 16), D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, none, 0);
    assert_eq!(
        (hr, plain.is_some()),
        (D3D_OK, true),
        "CreateOffscreenPlainSurfaceEx"
    );
}

#[test]
fn texture_memory_shrinks_on_a_plain_device_and_not_on_an_extended_one() {
    for (h, shrinks) in [(extended(), false), (Harness::new(), true)] {
        let before = h.available_texture_mem();
        let target = h.create_texture(
            1024,
            1024,
            1,
            D3DUSAGE_RENDERTARGET,
            D3DFMT_X8R8G8B8,
            D3DPOOL_DEFAULT,
        );
        let after = h.available_texture_mem();
        assert_eq!(
            after < before,
            shrinks,
            "{before} before the target, {after} after it"
        );
        drop(target);
    }
}

#[test]
fn the_frame_latency_defaults_to_three_and_stops_at_thirty() {
    let h = extended();
    assert_eq!(h.maximum_frame_latency(), (D3D_OK, 3), "the default");
    assert_eq!(h.set_maximum_frame_latency(1), D3D_OK);
    assert_eq!(h.maximum_frame_latency(), (D3D_OK, 1));
    assert_eq!(h.set_maximum_frame_latency(0), D3D_OK);
    assert_eq!(
        h.maximum_frame_latency(),
        (D3D_OK, 3),
        "zero restores the default"
    );
    assert_eq!(h.set_maximum_frame_latency(30), D3D_OK);
    assert_eq!(
        h.set_maximum_frame_latency(31),
        D3DERR_INVALIDCALL,
        "past the limit"
    );
    assert_eq!(
        h.maximum_frame_latency(),
        (D3D_OK, 30),
        "a refusal stores nothing"
    );
}

#[test]
fn flipex_and_thirty_back_buffers_reset_on_an_extended_device_alone() {
    let ex = extended();
    let plain = Harness::new();
    for (h, expected) in [(&ex, D3D_OK), (&plain, D3DERR_INVALIDCALL)] {
        let (width, height) = h.dims();
        let mut flipex = h.windowed_present_params(width, height);
        flipex.swap_effect = D3DSWAPEFFECT_FLIPEX;
        assert_eq!(h.reset_params(&mut flipex), expected, "FLIPEX");
        let mut thirty = h.windowed_present_params(width, height);
        thirty.back_buffer_count = 30;
        assert_eq!(h.reset_params(&mut thirty), expected, "30 back buffers");
    }
    ex.render_once(BLUE, |_| {});
    assert_pixel_eq(
        ex.read_pixel(320, 240),
        BLUE,
        "the extended device draws after them",
    );
}

#[test]
fn system_memory_uploads_reach_a_default_dxt_texture_and_a_cube_face() {
    let h = extended();
    // Four DXT1 blocks of solid red: both endpoints 0xF800, every index 0.
    let block = [0x00, 0xF8, 0x00, 0xF8, 0, 0, 0, 0];
    let blocks: Vec<u8> = block.iter().copied().cycle().take(32).collect();
    let staging = h.create_texture(8, 8, 1, 0, D3DFMT_DXT1, D3DPOOL_SYSTEMMEM);
    staging.lock_rect(0, 0).write_u8_rect(16, 2, &blocks);
    let target = h.create_texture(8, 8, 1, 0, D3DFMT_DXT1, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_surface_hr(&staging.surface_level(0), &target.surface_level(0)),
        D3D_OK,
        "UpdateSurface into a DXT1 texture"
    );
    assert_pixel_eq(
        sample_center(&h, &target),
        RED,
        "the DXT1 texture samples red",
    );

    let plain = h.create_offscreen_plain_surface(8, 8, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    plain.lock_rect(0).write_u32_rect(8, 8, &[BLUE; 64]);
    let cube = h.create_cube_texture_owned(8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let face = cube.surface(2, 0);
    assert_eq!(
        h.update_surface_hr(&plain, &face),
        D3D_OK,
        "UpdateSurface into a cube face"
    );
    let target = h.create_render_target(8, 8, D3DFMT_A8R8G8B8);
    assert_eq!(
        h.stretch_rect(&face, &target, D3DTEXF_NONE),
        D3D_OK,
        "copy the face out"
    );
    let readback = h.create_offscreen_plain_surface(8, 8, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(h.get_render_target_data_hr(&target, &readback), D3D_OK);
    let lock = readback.lock_rect(D3DLOCK_READONLY);
    let stride = usize::try_from(lock.pitch()).expect("positive pitch") / 4;
    let texels = lock.as_u32(7 * stride + 8).to_vec();
    drop(lock);
    for (x, y) in [(0, 0), (7, 0), (0, 7), (7, 7), (4, 4)] {
        assert_pixel_eq(
            texels[y * stride + x],
            BLUE,
            &format!("the cube face holds the uploaded texel at ({x}, {y})"),
        );
    }
}

#[test]
fn an_extended_device_copies_between_whole_default_textures() {
    let ex = extended();
    let plain = Harness::new();
    for (h, expected) in [(&ex, D3D_OK), (&plain, D3DERR_INVALIDCALL)] {
        let staging = h.create_texture(8, 8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
        staging.lock_rect(0, 0).write_u32_rect(8, 8, &[BLUE; 64]);
        let source = h.create_texture(8, 8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
        assert_eq!(h.update_texture_hr(&staging, &source), D3D_OK);
        let destination = h.create_texture(8, 8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
        let (src, dst) = (source.surface_level(0), destination.surface_level(0));
        assert_eq!(
            h.stretch_rect(&src, &dst, D3DTEXF_NONE),
            expected,
            "a whole-surface copy"
        );
        assert_eq!(
            h.stretch_rect_rects(&src, (0, 0, 8, 8), &dst, (0, 0, 8, 8), D3DTEXF_NONE),
            D3DERR_INVALIDCALL,
            "rects, even whole ones, make it a stretch"
        );
        if expected == D3D_OK {
            assert_pixel_eq(sample_center(h, &destination), BLUE, "the copy landed");
        }
    }
}

#[test]
fn set_priority_takes_the_default_pool_on_an_extended_device() {
    let ex = extended();
    let texture = ex.create_texture(8, 8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(texture.set_priority(7), 0, "the previous priority");
    assert_eq!(
        texture.priority(),
        7,
        "an extended default-pool texture keeps it"
    );
    let vb = ex.create_vertex_buffer(64, 0, 0, D3DPOOL_DEFAULT);
    assert_eq!(vb.set_priority(5), 0);
    assert_eq!(vb.priority(), 5, "so does a default-pool vertex buffer");
    let sysmem = ex.create_texture(8, 8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(sysmem.set_priority(7), 0);
    assert_eq!(sysmem.priority(), 0, "a system-memory texture does not");

    let plain = Harness::new();
    let texture = plain.create_texture(8, 8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(texture.set_priority(7), 0);
    assert_eq!(
        texture.priority(),
        0,
        "a plain default-pool texture does not"
    );
    let managed = plain.create_texture(8, 8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(managed.set_priority(7), 0);
    assert_eq!(managed.priority(), 7, "a plain managed texture does");
}

#[test]
fn ati2_is_no_offscreen_plain_surface_on_either_kind_of_device() {
    for h in [extended(), Harness::new()] {
        assert_eq!(
            h.create_offscreen_plain_surface_hr(16, 16, D3DFMT_ATI2, D3DPOOL_DEFAULT),
            D3DERR_INVALIDCALL
        );
    }
}

#[test]
fn the_extended_lockable_depth_formats_are_refused_on_an_extended_device() {
    let h = extended();
    for format in [D3DFMT_D32_LOCKABLE, D3DFMT_S8_LOCKABLE] {
        let (hr, texture) =
            h.try_create_texture(16, 16, 1, D3DUSAGE_DEPTHSTENCIL, format, D3DPOOL_DEFAULT);
        assert_eq!(
            (hr, texture.is_null()),
            (D3DERR_INVALIDCALL, true),
            "texture {format}"
        );
        let (hr, surface) =
            h.try_create_depth_stencil_surface_shared((16, 16), format, &SharedHandle::NONE);
        assert_eq!(
            (hr, surface.is_none()),
            (D3DERR_INVALIDCALL, true),
            "surface {format}"
        );
        let (width, height) = h.dims();
        let mut pp = h.windowed_present_params(width, height);
        pp.enable_auto_depth_stencil = 1;
        pp.auto_depth_stencil_format = format;
        assert_eq!(
            h.reset_ex(&mut pp, None),
            D3DERR_INVALIDCALL,
            "auto depth {format}"
        );
    }
}
