//! Lock, read-back and lifetime contracts of buffers, surfaces and textures.
//!
//! Nested buffer locks, a writable lock of a lockable back buffer, the read
//! back of a released level whose GPU format is wider than its staging, an
//! `UpdateTexture` between levels that overlap only in part, cube face locks
//! taken through one object and released through another, `ReleaseDC` on a
//! level that holds no device context, and creates whose system-memory copy
//! cannot be allocated.

use mtld3d_tests::{Harness, TexturedVertex, Vertex, assert_pixel_eq};
use mtld3d_types::{
    D3D_OK, D3DERR_DEVICELOST, D3DERR_INVALIDCALL, D3DFMT_A8R8G8B8, D3DFMT_INDEX16, D3DFMT_R8G8B8,
    D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_MANAGED,
    D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DRS_LIGHTING, D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV,
    D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DTADDRESS_CLAMP, D3DTEXF_POINT, D3DUSAGE_WRITEONLY,
};

use super::device::{await_logged_lines, run_in_private_log_child, running_as};

const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE;
const FAILED_DEVICE_CHILD_NAME: &str = "lock_lifetime_failed_device.exe";
#[cfg(target_arch = "x86")]
const OUT_OF_MEMORY_CHILD_NAME: &str = "lock_lifetime_out_of_memory.exe";
const BLACK: u32 = 0xFF00_0000;
const BLUE: u32 = 0xFF00_00FF;
const GREEN: u32 = 0xFF00_FF00;
const RED: u32 = 0xFFFF_0000;

fn stride() -> u32 {
    u32::try_from(size_of::<Vertex>()).expect("vertex stride fits u32")
}

const fn solid_triangle(color: u32) -> [Vertex; 3] {
    [
        Vertex {
            x: 0.0,
            y: 0.5,
            z: 0.5,
            color,
        },
        Vertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            color,
        },
        Vertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color,
        },
    ]
}

/// Drive the fixed-function pipeline so a draw shows the vertex diffuse colour.
fn arm_diffuse(h: &Harness) {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture");
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(FVF), 0, "SetFVF");
}

/// A whole-target quad sampling the unit square, white so MODULATE passes the texel.
fn textured_quad() -> [TexturedVertex; 6] {
    const W: u32 = 0xFFFF_FFFF;
    let corner = |x: f32, y: f32, u: f32, v: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: W,
        u,
        v,
    };
    [
        corner(-1.0, 1.0, 0.0, 0.0),
        corner(1.0, 1.0, 1.0, 0.0),
        corner(-1.0, -1.0, 0.0, 1.0),
        corner(1.0, 1.0, 1.0, 0.0),
        corner(1.0, -1.0, 1.0, 1.0),
        corner(-1.0, -1.0, 0.0, 1.0),
    ]
}

/// A second lock of a buffer already locked keeps its pointer until the last `Unlock`.
///
/// D3D9 counts the locks of a buffer and publishes what was written through
/// them once the count returns to zero. A lock record that is only a flag ends
/// at the first `Unlock`, which uploads and, for a write-only default-pool
/// buffer, releases the very pages the second lock's pointer still maps.
#[test]
fn nested_vertex_buffer_locks_publish_at_the_last_unlock() {
    let h = Harness::new();
    let vb = h.create_vertex_buffer(stride() * 3, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    let outer = vb.lock(0, 0, 0);
    let mut inner = vb.lock(0, 0, 0);
    assert_eq!(outer.unlock(), D3D_OK, "the first Unlock of two locks");
    inner.write(&solid_triangle(GREEN));
    assert_eq!(inner.unlock(), D3D_OK, "the Unlock that ends the last lock");

    arm_diffuse(&h);
    assert_eq!(
        h.set_stream_source(0, &vb, 0, stride()),
        0,
        "SetStreamSource"
    );
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1),
            0,
            "DrawPrimitive"
        );
    });
    assert_pixel_eq(
        h.read_pixel(320, 280),
        GREEN,
        "the vertices written through the second lock reach the draw",
    );
}

/// The index-buffer form of [`nested_vertex_buffer_locks_publish_at_the_last_unlock`].
#[test]
fn nested_index_buffer_locks_publish_at_the_last_unlock() {
    let h = Harness::new();
    let vb = h.create_vertex_buffer(stride() * 3, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    vb.lock(0, 0, 0).write(&solid_triangle(GREEN));
    let ib = h.create_index_buffer(6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
    let outer = ib.lock(0, 0, 0);
    let mut inner = ib.lock(0, 0, 0);
    assert_eq!(outer.unlock(), D3D_OK, "the first Unlock of two locks");
    inner.write(&[0u16, 1, 2]);
    assert_eq!(inner.unlock(), D3D_OK, "the Unlock that ends the last lock");

    arm_diffuse(&h);
    assert_eq!(
        h.set_stream_source(0, &vb, 0, stride()),
        0,
        "SetStreamSource"
    );
    assert_eq!(h.set_indices(&ib), 0, "SetIndices");
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 3, 0, 1),
            0,
            "DrawIndexedPrimitive"
        );
    });
    assert_pixel_eq(
        h.read_pixel(320, 280),
        GREEN,
        "the indices written through the second lock reach the draw",
    );
}

/// A writable lock of a lockable back buffer carries its writes into the back buffer.
///
/// `D3DPRESENTFLAG_LOCKABLE_BACKBUFFER` makes the back buffer an ordinary
/// lockable surface, so what the application writes through the pointer is
/// what the back buffer holds after `UnlockRect`. The block is 64 pixels wide
/// so the scaled leg, which resamples the write-back, still reads an interior
/// pixel unchanged.
#[test]
fn a_writable_lock_of_a_lockable_back_buffer_reaches_the_back_buffer() {
    const SIDE: usize = 64;
    let h = Harness::with_lockable_back_buffer();
    assert_eq!(h.clear_target(GREEN), D3D_OK, "clear the back buffer green");
    let bb = h.back_buffer(0);
    {
        let mut locked = bb.lock_rect_partial(&[64, 64, 128, 128], 0);
        locked.write_u32_rect(SIDE, SIDE, &[RED; SIDE * SIDE]);
    }
    assert_pixel_eq(
        h.read_pixel(96, 96) | BLACK,
        RED,
        "the pixels written through the lock reach the back buffer",
    );
    assert_pixel_eq(
        h.read_pixel(320, 240) | BLACK,
        GREEN,
        "the pixels outside the locked rect keep the clear colour",
    );
}

/// A lock of a released 24-bit level reads the level back at its own three-byte layout.
///
/// No GPU format has three bytes a texel, so the level lives on the GPU as
/// BGRA8 and its staging, released once the upload carried it, holds the D3D9
/// layout. Reading it back has to narrow the four-byte texels into that
/// layout, the way every other read back of a widened level does.
#[test]
fn a_lock_of_a_released_r8g8b8_level_reads_its_texels_back() {
    const SIZE: u32 = 8;
    const ROW: usize = SIZE as usize * 3;
    let h = Harness::new();
    let tex = h.create_texture(SIZE, SIZE, 1, 0, D3DFMT_R8G8B8, D3DPOOL_DEFAULT);
    // One distinct byte per position, so neither a short row stride nor an
    // uninitialised page can match.
    let written: Vec<u8> = (0..ROW * SIZE as usize)
        .map(|i| u8::try_from(i % 251).expect("below 251"))
        .collect();
    tex.lock_rect(0, 0)
        .write_u8_rect(ROW, SIZE as usize, &written);

    // The draw uploads the level, the read back waits for the encoder to
    // emit that upload, and the Present after it releases the staging.
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &textured_quad()),
            0
        );
    });
    let _sampled = h.read_pixel(320, 240);
    assert_eq!(h.clear_texture(0), 0, "unbind");
    assert_eq!(h.present(), 0, "the Present that releases uploaded staging");

    let locked = tex.lock_rect(0, D3DLOCK_READONLY);
    let pitch = usize::try_from(locked.pitch()).expect("a positive row pitch");
    let bytes = locked.as_u8(pitch * (SIZE as usize - 1) + ROW);
    for (y, row) in written.as_chunks::<ROW>().0.iter().enumerate() {
        assert_eq!(
            &bytes[y * pitch..y * pitch + ROW],
            row.as_slice(),
            "row {y} of the read back holds the texels the lock wrote"
        );
    }
}

/// `UpdateTexture` copies what two levels share and skips what they do not.
///
/// D3D9 does not require the two textures to match in size: a level pair
/// that overlaps only in part copies the overlap. A source dirty rectangle
/// that lies wholly outside the destination level copies nothing for that
/// level, and the call still succeeds with every other level copied.
#[test]
fn update_texture_skips_a_dirty_region_outside_the_destination_level() {
    let h = Harness::new();
    let src = h.create_texture(8, 2, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    let dst = h.create_texture(2, 8, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    src.lock_rect(0, 0).write_u32(&[RED; 16]);
    src.lock_rect(1, 0).write_u32(&[RED; 4]);
    assert_eq!(h.update_texture_hr(&src, &dst), D3D_OK, "the first copy");

    // Level 0 dirty whole; level 1 dirty only at x 2..4, past the one texel
    // column the destination's 1x4 level 1 has.
    src.lock_rect(0, 0).write_u32(&[BLUE; 16]);
    src.lock_rect_partial(1, &[2, 0, 4, 1], 0)
        .write_u32(&[GREEN; 2]);
    assert_eq!(
        h.update_texture_hr(&src, &dst),
        D3D_OK,
        "a level whose dirty region lies outside the destination copies nothing and fails nothing"
    );
    let locked = dst.lock_rect(0, D3DLOCK_READONLY);
    let pitch_px = usize::try_from(locked.pitch()).expect("a positive row pitch") / 4;
    let texels = locked.as_u32(pitch_px + 2);
    assert_eq!(
        [texels[0], texels[1], texels[pitch_px], texels[pitch_px + 1]],
        [BLUE; 4],
        "level 0's overlap holds the second copy"
    );
}

/// A cube face locked through the cube texture unlocks through its surface.
///
/// The surface and the cube name one subresource, so they share one lock:
/// either object's `UnlockRect` ends a lock either one took, and the face can
/// be locked again afterwards.
#[test]
fn a_cube_face_locked_through_the_texture_unlocks_through_its_surface() {
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    let face = cube.surface(2, 0);
    core::mem::forget(cube.lock_rect(2, 0, 0));
    assert_eq!(
        face.unlock_rect(),
        D3D_OK,
        "the surface ends the cube's lock"
    );
    assert_eq!(
        face.unlock_rect(),
        D3DERR_INVALIDCALL,
        "a second unlock finds nothing locked"
    );
    assert_eq!(
        cube.lock_rect(2, 0, 0).unlock(),
        D3D_OK,
        "the face locks again"
    );
}

/// A cube face locked through its surface unlocks through the cube texture.
#[test]
fn a_cube_face_locked_through_its_surface_unlocks_through_the_texture() {
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    let face = cube.surface(2, 0);
    core::mem::forget(face.lock_rect(0));
    assert_eq!(
        cube.unlock_rect(2, 0),
        D3D_OK,
        "the cube ends the surface's lock"
    );
    assert_eq!(
        face.lock_rect_probe(0).0,
        D3D_OK,
        "the face locks again through its surface"
    );
    assert_eq!(face.unlock_rect(), D3D_OK, "and unlocks");
}

/// `ReleaseDC` is accepted only on the subresource whose `GetDC` returned the handle.
///
/// A device context belongs to one level of one face. Releasing it through a
/// sibling's surface is refused and leaves the context held, so the owner's
/// own `ReleaseDC` still writes back what GDI drew.
#[test]
fn release_dc_on_a_sibling_subresource_is_refused() {
    let h = Harness::new();
    let tex = h.create_texture(8, 8, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    let (level0, level1) = (tex.surface_level(0), tex.surface_level(1));
    let dc = level0.dc();
    assert_eq!(
        level1.release_dc_of(&dc),
        D3DERR_INVALIDCALL,
        "a sibling level does not hold the DC"
    );
    assert_eq!(dc.release(), D3D_OK, "the level that holds it releases it");

    let cube = h.create_cube_texture_owned(8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    let (face0, face1) = (cube.surface(0, 0), cube.surface(1, 0));
    let dc = face0.dc();
    assert_eq!(
        face1.release_dc_of(&dc),
        D3DERR_INVALIDCALL,
        "a sibling face does not hold the DC"
    );
    assert_eq!(dc.release(), D3D_OK, "the face that holds it releases it");
}

/// A create whose system-memory copy no allocation can hold answers `E_OUTOFMEMORY`.
///
/// Inside a 32-bit process each of these is 2 GiB or more, past what a
/// layout can describe there, whatever the sizes the device reports allow,
/// so nothing is allocated. The create fails cleanly, hands back no object,
/// and the device keeps working.
#[cfg(target_arch = "x86")]
#[test]
fn creates_whose_staging_cannot_be_allocated_answer_out_of_memory() {
    use mtld3d_types::{D3DFMT_A32B32G32R32F, E_OUTOFMEMORY};
    let h = Harness::new();
    let (hr, vb) = h.try_create_vertex_buffer(0xFFFF_0000, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    assert_eq!(
        (hr, vb.is_null()),
        (E_OUTOFMEMORY, true),
        "CreateVertexBuffer"
    );
    let (hr, ib) = h.try_create_index_buffer(
        0xFFFF_0000,
        D3DUSAGE_WRITEONLY,
        D3DFMT_INDEX16,
        D3DPOOL_DEFAULT,
    );
    assert_eq!(
        (hr, ib.is_null()),
        (E_OUTOFMEMORY, true),
        "CreateIndexBuffer"
    );
    // 2 GiB for level 0 alone.
    let (hr, tex) = h.try_create_texture(16384, 8192, 1, 0, D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED);
    assert_eq!((hr, tex.is_null()), (E_OUTOFMEMORY, true), "CreateTexture");
    // 2 GiB for the one level.
    let (hr, volume) = h.try_create_volume_texture(
        [1024, 1024, 128],
        1,
        0,
        D3DFMT_A32B32G32R32F,
        D3DPOOL_MANAGED,
    );
    assert_eq!(
        (hr, volume.is_none()),
        (E_OUTOFMEMORY, true),
        "CreateVolumeTexture"
    );
    assert_eq!(
        h.create_offscreen_plain_surface_hr(16384, 8192, D3DFMT_A32B32G32R32F, D3DPOOL_SYSTEMMEM),
        E_OUTOFMEMORY,
        "CreateOffscreenPlainSurface"
    );

    let vb = h.create_vertex_buffer(stride() * 3, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    vb.lock(0, 0, 0).write(&solid_triangle(GREEN));
    arm_diffuse(&h);
    assert_eq!(
        h.set_stream_source(0, &vb, 0, stride()),
        0,
        "SetStreamSource"
    );
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1),
            0,
            "DrawPrimitive"
        );
    });
    assert_pixel_eq(h.read_pixel(320, 280), GREEN, "the device still draws");
}

/// A render target released after the device failed is destroyed with the device.
///
/// `debug.failNextSubmit` refuses the first frame the way a Metal rejection
/// does, so the device latches `D3DERR_DEVICELOST` and sends no frame again.
/// A retire recorded into a frame then never reaches the encoder that
/// destroys the target's Metal texture; the final `Release` destroys it
/// instead and logs the count. Runs in a process of its own so the log it
/// reads is its device's alone.
#[test]
fn a_target_released_after_the_device_failed_is_destroyed_with_the_device() {
    if running_as(FAILED_DEVICE_CHILD_NAME) {
        failed_device_workload();
        return;
    }
    run_in_private_log_child(
        FAILED_DEVICE_CHILD_NAME,
        "lock_lifetime::a_target_released_after_the_device_failed_is_destroyed_with_the_device",
        "warn,mtld3d::d3d9=info",
    );
}

/// Fail the device, release two targets, then the device, and read the count it destroyed.
fn failed_device_workload() {
    let h = Harness::with_config("debug.failNextSubmit=true");
    let first = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    let second = h.create_render_target(32, 32, D3DFMT_A8R8G8B8);
    assert_eq!(
        h.present(),
        D3DERR_DEVICELOST,
        "the first submission is refused"
    );
    assert_eq!(
        h.test_cooperative_level(),
        D3DERR_DEVICELOST,
        "the failure is latched"
    );
    drop(first);
    drop(second);
    assert_eq!(
        h.present(),
        D3DERR_DEVICELOST,
        "a failed device presents nothing"
    );
    assert_eq!(h.release_device(), 0, "the final Release");
    let lines = await_logged_lines(
        "Metal textures of targets released after the device failed",
        1,
    );
    let destroyed: usize = lines[0]
        .split("destroyed ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|count| count.parse().ok())
        .expect("the line names a count");
    // Each target carries its colour texture and, for a format with an sRGB
    // twin, the twin view taken of it.
    assert!(
        destroyed >= 2,
        "both targets' textures are destroyed with the device: {lines:?}"
    );
}

/// A cube whose six faces the 32-bit address space cannot hold answers `E_OUTOFMEMORY`.
///
/// Each face is 1 GiB, a size a layout can describe, so the allocator itself
/// refuses one of them, and the faces taken before it are given back. The
/// allocations are real, so the create runs in a process of its own, where
/// no other test's allocation competes for the address space.
#[cfg(target_arch = "x86")]
#[test]
fn a_cube_the_allocator_refuses_answers_out_of_memory() {
    if running_as(OUT_OF_MEMORY_CHILD_NAME) {
        out_of_memory_workload();
        return;
    }
    run_in_private_log_child(
        OUT_OF_MEMORY_CHILD_NAME,
        "lock_lifetime::a_cube_the_allocator_refuses_answers_out_of_memory",
        "warn",
    );
}

/// Create the cube the allocator refuses, then draw.
#[cfg(target_arch = "x86")]
fn out_of_memory_workload() {
    use mtld3d_types::{D3DFMT_A32B32G32R32F, E_OUTOFMEMORY};
    let h = Harness::new();
    let (hr, cube) = h.try_create_cube_texture(8192, 1, 0, D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED);
    assert_eq!(
        (hr, cube.is_null()),
        (E_OUTOFMEMORY, true),
        "CreateCubeTexture"
    );
    let vb = h.create_vertex_buffer(stride() * 3, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    vb.lock(0, 0, 0).write(&solid_triangle(GREEN));
    arm_diffuse(&h);
    assert_eq!(
        h.set_stream_source(0, &vb, 0, stride()),
        0,
        "SetStreamSource"
    );
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1),
            0,
            "DrawPrimitive"
        );
    });
    assert_pixel_eq(h.read_pixel(320, 280), GREEN, "the device still draws");
}
