//! Fixed-function texture-stage colour ops and argument sources.

use mtld3d_tests::{Harness, Rgba8, Texture, TexturedVertex};
use mtld3d_types::{
    D3DFMT_A8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DPT_TRIANGLELIST, D3DRS_LIGHTING,
    D3DRS_TEXTUREFACTOR, D3DTA_DIFFUSE, D3DTA_TEXTURE, D3DTA_TFACTOR, D3DTOP_ADD, D3DTOP_MODULATE,
    D3DTOP_SELECTARG1, D3DTOP_SELECTARG2, D3DTSS_ALPHAARG1, D3DTSS_ALPHAOP, D3DTSS_COLORARG1,
    D3DTSS_COLORARG2, D3DTSS_COLOROP,
};

const BLACK: u32 = 0xFF00_0000;

/// A 2x1 texture sampled at u = 0.75 must return its second texel.
///
/// The other stage tests use a 1x1 texture, which samples identically for every
/// texcoord, so they never exercise texcoord *delivery*. This one does: texel 0
/// is RED, texel 1 is GREEN, so a vertex texcoord that is dropped (read as 0)
/// returns RED instead.
#[test]
fn up_fvf_delivers_nonzero_texcoord() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;

    let h = Harness::new();
    let tex = h.create_texture(2, 1, 1, 0, D3DFMT_A8R8G8B8, 0);
    tex.lock_rect(0, 0).write_u32(&[RED, GREEN]);

    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
        0
    );
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);

    let v = |x: f32, y: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0,
        u: 0.75,
        v: 0.5,
    };
    let verts = [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &verts),
            0,
            "draw"
        );
    });
    let px = h.read_pixel(320, 240);
    assert_eq!(
        Rgba8::from_pixel(px),
        Rgba8::from_pixel(GREEN),
        "u=0.75 must sample the GREEN texel (texcoord delivered); got {px:#010x}"
    );
}

/// Same as `up_fvf_delivers_nonzero_texcoord` but the layout comes from a declaration.
///
/// `SetVertexDeclaration` (POSITION+COLOR+TEXCOORD) supplies it instead of
/// `SetFVF`. Isolates whether the declaration path delivers the texcoord.
#[test]
fn up_decl_delivers_nonzero_texcoord() {
    use mtld3d_types::{
        D3DDECL_END_STREAM, D3DDECLTYPE_D3DCOLOR, D3DDECLTYPE_FLOAT2, D3DDECLTYPE_FLOAT3,
        D3DDECLTYPE_UNUSED, D3DDECLUSAGE_COLOR, D3DDECLUSAGE_POSITION, D3DDECLUSAGE_TEXCOORD,
        D3DVERTEXELEMENT9,
    };
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;

    let elem = |offset: u16, type_: u8, usage: u8| D3DVERTEXELEMENT9 {
        stream: 0,
        offset,
        type_,
        method: 0,
        usage,
        usage_index: 0,
    };
    let decl_elems = [
        elem(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
        elem(12, D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_COLOR),
        elem(16, D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_TEXCOORD),
        D3DVERTEXELEMENT9 {
            stream: D3DDECL_END_STREAM,
            offset: 0,
            type_: D3DDECLTYPE_UNUSED,
            method: 0,
            usage: 0,
            usage_index: 0,
        },
    ];

    let h = Harness::new();
    let tex = h.create_texture(2, 1, 1, 0, D3DFMT_A8R8G8B8, 0);
    tex.lock_rect(0, 0).write_u32(&[RED, GREEN]);

    let decl = h.create_vertex_declaration(&decl_elems);
    assert_eq!(h.set_vertex_declaration(&decl), 0, "SetVertexDeclaration");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
        0
    );

    let v = |x: f32, y: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0,
        u: 0.75,
        v: 0.5,
    };
    let verts = [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &verts),
            0,
            "draw"
        );
    });
    let px = h.read_pixel(320, 240);
    assert_eq!(
        Rgba8::from_pixel(px),
        Rgba8::from_pixel(GREEN),
        "decl path: u=0.75 must sample GREEN (texcoord delivered); got {px:#010x}"
    );
}

/// A POSITION+NORMAL+TEXCOORD declaration still delivers the texcoord.
///
/// The layout comes from `SetVertexDeclaration` with lighting OFF, so the
/// NORMAL at attribute index 1 is dead and gets stripped from the compiled FF
/// VS. Verifies the texcoord at attribute index 4 is still delivered.
#[test]
fn up_decl_normal_then_texcoord_delivers() {
    use mtld3d_types::{
        D3DDECL_END_STREAM, D3DDECLTYPE_FLOAT2, D3DDECLTYPE_FLOAT3, D3DDECLTYPE_UNUSED,
        D3DDECLUSAGE_NORMAL, D3DDECLUSAGE_POSITION, D3DDECLUSAGE_TEXCOORD, D3DVERTEXELEMENT9,
    };
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PosNormalTexVertex {
        x: f32,
        y: f32,
        z: f32,
        nx: f32,
        ny: f32,
        nz: f32,
        u: f32,
        v: f32,
    }

    let elem = |offset: u16, type_: u8, usage: u8| D3DVERTEXELEMENT9 {
        stream: 0,
        offset,
        type_,
        method: 0,
        usage,
        usage_index: 0,
    };
    let decl_elems = [
        elem(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
        elem(12, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_NORMAL),
        elem(24, D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_TEXCOORD),
        D3DVERTEXELEMENT9 {
            stream: D3DDECL_END_STREAM,
            offset: 0,
            type_: D3DDECLTYPE_UNUSED,
            method: 0,
            usage: 0,
            usage_index: 0,
        },
    ];

    let h = Harness::new();
    let tex = h.create_texture(2, 1, 1, 0, D3DFMT_A8R8G8B8, 0);
    tex.lock_rect(0, 0).write_u32(&[RED, GREEN]);

    let decl = h.create_vertex_declaration(&decl_elems);
    assert_eq!(h.set_vertex_declaration(&decl), 0, "SetVertexDeclaration");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
        0
    );

    let v = |x: f32, y: f32| PosNormalTexVertex {
        x,
        y,
        z: 0.5,
        nx: 0.0,
        ny: 0.0,
        nz: 1.0,
        u: 0.75,
        v: 0.5,
    };
    let verts = [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &verts),
            0,
            "draw"
        );
    });
    let px = h.read_pixel(320, 240);
    assert_eq!(
        Rgba8::from_pixel(px),
        Rgba8::from_pixel(GREEN),
        "POSITION+NORMAL+TEXCOORD: u=0.75 must sample GREEN; got {px:#010x}"
    );
}

/// A 2x1 A32B32G32R32F (128-bit float) texture sampled at u = 0.75 must return its second texel.
///
/// Covers the float-format sampled-texture path, which the 8-bit stage tests
/// don't exercise. If a float-format texture uploads or samples as zero, this
/// returns black.
#[test]
fn up_fvf_samples_float_texture() {
    use mtld3d_types::D3DFMT_A32B32G32R32F;

    let h = Harness::new();
    let tex = h.create_texture(2, 1, 1, 0, D3DFMT_A32B32G32R32F, 0);
    // texel 0 = red (1,0,0,1), texel 1 = green (0,1,0,1), RGBA float order.
    tex.lock_rect(0, 0)
        .write::<f32>(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
        0
    );
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);

    let v = |x: f32, y: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0,
        u: 0.75,
        v: 0.5,
    };
    let verts = [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &verts),
            0,
            "draw"
        );
    });
    let px = h.read_pixel(320, 240);
    assert_eq!(
        Rgba8::from_pixel(px),
        Rgba8::from_pixel(0xFF00_FF00),
        "u=0.75 must sample the GREEN float texel; got {px:#010x}"
    );
}

/// A 2x1 A16B16G16R16F (half-float) texture sampled at u = 0.75 must return its second texel.
///
/// The half-float twin of the A32B32G32R32F case above: it pins the binary16
/// upload path (bytes go in as the CPU encoded them) and the sampler's decode.
#[test]
fn up_fvf_samples_half_float_texture() {
    use mtld3d_core::convert::f32_to_f16_bits;
    use mtld3d_types::D3DFMT_A16B16G16R16F;

    let h = Harness::new();
    let tex = h.create_texture(2, 1, 1, 0, D3DFMT_A16B16G16R16F, 0);
    // texel 0 = red (1,0,0,1), texel 1 = green (0,1,0,1), RGBA half order.
    let texels: Vec<u16> = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        .into_iter()
        .map(f32_to_f16_bits)
        .collect();
    tex.lock_rect(0, 0).write::<u16>(&texels);

    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
        0
    );
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);

    let v = |x: f32, y: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0,
        u: 0.75,
        v: 0.5,
    };
    let verts = [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &verts),
            0,
            "draw"
        );
    });
    let px = h.read_pixel(320, 240);
    assert_eq!(
        Rgba8::from_pixel(px),
        Rgba8::from_pixel(0xFF00_FF00),
        "u=0.75 must sample the GREEN half-float texel; got {px:#010x}"
    );
}

/// A larger A32B32G32R32F texture filled row-by-row honouring the locked pitch.
///
/// Texel (x,y) = (x/W, y/H, 0, 1), then sampled at a known texcoord. If the
/// multi-row pitch / upload of a large float texture is wrong, the sample
/// lands on the wrong texel (commonly (0,0) -> black).
#[test]
fn up_fvf_samples_large_float_texture_via_pitch() {
    use mtld3d_types::{D3DFMT_A32B32G32R32F, D3DLOCK_READONLY};

    const W: u32 = 200;
    const H: u32 = 200;

    let h = Harness::new();
    // D3DPOOL_MANAGED (1) — the pool an app uploads a static gradient through.
    let tex = h.create_texture(W, H, 1, 0, D3DFMT_A32B32G32R32F, 1);
    {
        let lr = tex.lock_rect(0, 0);
        let pitch = usize::try_from(lr.pitch()).expect("non-negative pitch");
        let base = lr.bits_ptr();
        let dim = f32::from(u16::try_from(W).expect("W < 65536"));
        for y in 0..H {
            let fy = f32::from(u16::try_from(y).expect("y < 65536")) / dim;
            for x in 0..W {
                let fx = f32::from(u16::try_from(x).expect("x < 65536")) / dim;
                let px = [fx, fy, 0.0_f32, 1.0_f32];
                let off = y as usize * pitch + x as usize * 16;
                // SAFETY: `off` is in-bounds of the locked region (y < H, x < W,
                // pitch >= W*16).
                let dst = unsafe { base.add(off) };
                // SAFETY: `dst` is valid for the 16 bytes of one float4 texel.
                unsafe { core::ptr::copy_nonoverlapping(px.as_ptr().cast::<u8>(), dst, 16) };
            }
        }
    }

    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
        0
    );
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);

    // u=0.5 -> texel column 32 -> r = 32/64 = 0.5 (~128). v=0.25 -> row 16 ->
    // g = 16/64 = 0.25 (~64).
    let v = |x: f32, y: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0,
        u: 0.5,
        v: 0.25,
    };
    let verts = [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ];
    let _ = D3DLOCK_READONLY;
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &verts),
            0,
            "draw"
        );
    });
    let got = Rgba8::from_pixel(h.read_pixel(320, 240));
    // Expect ~(128, 64, 0). Allow a wide tolerance for point-sample texel choice.
    assert!(
        got.r > 100 && got.r < 160 && got.g > 40 && got.g < 96 && got.b < 16,
        "large float texture sampled at (0.5,0.25) should be ~(128,64,0); got {got:?}"
    );
}

/// XYZRHW (pre-transformed) texcoord delivery.
///
/// The RHW path, which the other stage tests (all XYZ) never cover. A 2-texel
/// texture sampled at u=0.75 over a screen-space RHW quad must return the
/// second (GREEN) texel.
#[test]
fn up_rhw_delivers_texcoord() {
    use mtld3d_types::D3DFVF_XYZRHW;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RhwTexVertex {
        x: f32,
        y: f32,
        z: f32,
        rhw: f32,
        u: f32,
        v: f32,
    }

    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;

    let h = Harness::new();
    let tex = h.create_texture(2, 1, 1, 0, D3DFMT_A8R8G8B8, 0);
    tex.lock_rect(0, 0).write_u32(&[RED, GREEN]);

    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
        0
    );
    assert_eq!(h.set_fvf(D3DFVF_XYZRHW | D3DFVF_TEX1), 0);

    // Screen-space quad covering the whole 640x480 backbuffer, u=0.75.
    let v = |x: f32, y: f32| RhwTexVertex {
        x,
        y,
        z: 0.5,
        rhw: 1.0,
        u: 0.75,
        v: 0.5,
    };
    let verts = [
        v(0.0, 0.0),
        v(640.0, 0.0),
        v(0.0, 480.0),
        v(640.0, 0.0),
        v(640.0, 480.0),
        v(0.0, 480.0),
    ];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &verts),
            0,
            "draw"
        );
    });
    let px = h.read_pixel(320, 240);
    assert_eq!(
        Rgba8::from_pixel(px),
        Rgba8::from_pixel(GREEN),
        "XYZRHW u=0.75 must sample GREEN texel; got {px:#010x}"
    );
}

/// XYZRHW texcoord delivery via `SetVertexDeclaration` (POSITIONT FLOAT4 + TEXCOORD).
///
/// The declaration RHW layout — distinct from the FVF XYZRHW path. A 2-texel
/// texture sampled at u=0.75 must return GREEN.
#[test]
fn up_rhw_decl_delivers_texcoord() {
    use mtld3d_types::{
        D3DDECL_END_STREAM, D3DDECLTYPE_FLOAT2, D3DDECLTYPE_FLOAT4, D3DDECLTYPE_UNUSED,
        D3DDECLUSAGE_POSITIONT, D3DDECLUSAGE_TEXCOORD, D3DVERTEXELEMENT9,
    };

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RhwTexVertex {
        x: f32,
        y: f32,
        z: f32,
        rhw: f32,
        u: f32,
        v: f32,
    }

    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;

    let elem = |offset: u16, type_: u8, usage: u8| D3DVERTEXELEMENT9 {
        stream: 0,
        offset,
        type_,
        method: 0,
        usage,
        usage_index: 0,
    };
    let decl_elems = [
        elem(0, D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_POSITIONT),
        elem(16, D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_TEXCOORD),
        D3DVERTEXELEMENT9 {
            stream: D3DDECL_END_STREAM,
            offset: 0,
            type_: D3DDECLTYPE_UNUSED,
            method: 0,
            usage: 0,
            usage_index: 0,
        },
    ];

    let h = Harness::new();
    let tex = h.create_texture(2, 1, 1, 0, D3DFMT_A8R8G8B8, 0);
    tex.lock_rect(0, 0).write_u32(&[RED, GREEN]);

    let decl = h.create_vertex_declaration(&decl_elems);
    assert_eq!(h.set_vertex_declaration(&decl), 0, "SetVertexDeclaration");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
        0
    );

    let v = |x: f32, y: f32| RhwTexVertex {
        x,
        y,
        z: 0.5,
        rhw: 1.0,
        u: 0.75,
        v: 0.5,
    };
    let verts = [
        v(0.0, 0.0),
        v(640.0, 0.0),
        v(0.0, 480.0),
        v(640.0, 0.0),
        v(640.0, 480.0),
        v(0.0, 480.0),
    ];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &verts),
            0,
            "draw"
        );
    });
    let px = h.read_pixel(320, 240);
    assert_eq!(
        Rgba8::from_pixel(px),
        Rgba8::from_pixel(GREEN),
        "XYZRHW decl path: u=0.75 must sample GREEN; got {px:#010x}"
    );
}

/// The vertex stride is the application's `VertexStreamZeroStride` (`size_of::<V>()` here).
///
/// NOT the declaration's min-extent. A struct with trailing padding has a
/// stride strictly larger than `max(offset + size)` over its elements; if the
/// draw path derives stride from the declaration it reads every vertex after
/// the first at the wrong offset and the attributes past position decode as
/// garbage (typically 0). Here the texcoord then reads as 0 for verts 1..N and
/// the sampled colour collapses to texel 0 (RED).
#[test]
fn up_padded_vertex_stride_renders() {
    use mtld3d_types::{
        D3DDECL_END_STREAM, D3DDECLTYPE_FLOAT2, D3DDECLTYPE_FLOAT3, D3DDECLTYPE_UNUSED,
        D3DDECLUSAGE_POSITION, D3DDECLUSAGE_TEXCOORD, D3DVERTEXELEMENT9,
    };

    // 36-byte stride; declaration min-extent is only 20 (FLOAT3@0 + FLOAT2@12).
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PaddedVertex {
        x: f32,
        y: f32,
        z: f32,
        u: f32,
        v: f32,
        _pad: [f32; 4],
    }

    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;

    let elem = |offset: u16, type_: u8, usage: u8| D3DVERTEXELEMENT9 {
        stream: 0,
        offset,
        type_,
        method: 0,
        usage,
        usage_index: 0,
    };
    let decl_elems = [
        elem(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
        elem(12, D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_TEXCOORD),
        D3DVERTEXELEMENT9 {
            stream: D3DDECL_END_STREAM,
            offset: 0,
            type_: D3DDECLTYPE_UNUSED,
            method: 0,
            usage: 0,
            usage_index: 0,
        },
    ];

    let h = Harness::new();
    let tex = h.create_texture(2, 1, 1, 0, D3DFMT_A8R8G8B8, 0);
    tex.lock_rect(0, 0).write_u32(&[RED, GREEN]);

    let decl = h.create_vertex_declaration(&decl_elems);
    assert_eq!(h.set_vertex_declaration(&decl), 0, "SetVertexDeclaration");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
        0
    );

    let v = |x: f32, y: f32| PaddedVertex {
        x,
        y,
        z: 0.5,
        u: 0.75,
        v: 0.5,
        _pad: [0.0; 4],
    };
    let verts = [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &verts),
            0,
            "draw"
        );
    });
    let px = h.read_pixel(320, 240);
    assert_eq!(
        Rgba8::from_pixel(px),
        Rgba8::from_pixel(GREEN),
        "padded stride: u=0.75 must sample GREEN; got {px:#010x}"
    );
}

fn solid_texture(h: &Harness, argb: u32) -> Texture<'_> {
    let tex = h.create_texture(1, 1, 1, 0, D3DFMT_A8R8G8B8, 0);
    tex.lock_rect(0, 0).write_u32(&[argb]);
    tex
}

fn quad(diffuse: u32) -> [TexturedVertex; 6] {
    let v = |x: f32, y: f32, u: f32, vv: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: diffuse,
        u,
        v: vv,
    };
    [
        v(-1.0, 1.0, 0.0, 0.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(-1.0, -1.0, 0.0, 1.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(1.0, -1.0, 1.0, 1.0),
        v(-1.0, -1.0, 0.0, 1.0),
    ]
}

/// Bind `tex`, set the stage colour op/args, draw a full quad of `diffuse`.
///
/// Returns the centre pixel. The independent alpha operation selects the texture.
fn render_stage(
    h: &Harness,
    tex: &Texture<'_>,
    op: u32,
    arg1: u32,
    arg2: u32,
    diffuse: u32,
) -> Rgba8 {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_texture(0, tex), 0, "SetTexture");
    assert_eq!(h.set_texture_stage_state(0, D3DTSS_COLOROP, op), 0);
    assert_eq!(h.set_texture_stage_state(0, D3DTSS_COLORARG1, arg1), 0);
    assert_eq!(h.set_texture_stage_state(0, D3DTSS_COLORARG2, arg2), 0);
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE),
        0
    );
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    let verts = quad(diffuse);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &verts),
            0,
            "stage draw"
        );
    });
    Rgba8::from_pixel(h.read_pixel(320, 240))
}

#[test]
fn colorop_round_trips() {
    let h = Harness::new();
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_ADD),
        (D3DTSS_COLORARG1, D3DTA_TEXTURE),
        (D3DTSS_COLORARG2, D3DTA_DIFFUSE),
        (D3DTSS_ALPHAOP, D3DTOP_MODULATE),
    ] {
        assert_eq!(
            h.set_texture_stage_state(0, state, value),
            0,
            "SetTSS {state}"
        );
        assert_eq!(
            h.texture_stage_state(0, state),
            value,
            "GetTSS {state} round-trip"
        );
    }
}

#[test]
fn out_of_range_stage_states_read_as_the_stage_defaults() {
    let h = Harness::new();
    let red = solid_texture(&h, 0xFFFF_0000);
    let expected = render_stage(
        &h,
        &red,
        D3DTOP_ADD,
        D3DTA_TEXTURE,
        D3DTA_DIFFUSE,
        0xFF00_FF00,
    );

    // Stage 1 carries no texture and its spec default op is D3DTOP_DISABLE,
    // which is what a value no D3DTOP_* names reads as, so the cascade still
    // ends at stage 0 and the frame is the one above.
    for state in [
        D3DTSS_COLOROP,
        D3DTSS_COLORARG1,
        D3DTSS_COLORARG2,
        D3DTSS_ALPHAOP,
        D3DTSS_ALPHAARG1,
    ] {
        assert_eq!(
            h.set_texture_stage_state(1, state, 0x0001_0000),
            0,
            "SetTSS {state} out of range"
        );
        assert_eq!(
            h.texture_stage_state(1, state),
            0x0001_0000,
            "GetTSS {state} round-trip"
        );
    }

    let px = render_stage(
        &h,
        &red,
        D3DTOP_ADD,
        D3DTA_TEXTURE,
        D3DTA_DIFFUSE,
        0xFF00_FF00,
    );
    assert_eq!(
        px, expected,
        "out-of-range stage-1 states changed the frame"
    );
}

#[test]
fn modulate_texture_by_diffuse() {
    let h = Harness::new();
    let gray = solid_texture(&h, 0xFF80_8080); // 0.5 per channel
    let px = render_stage(
        &h,
        &gray,
        D3DTOP_MODULATE,
        D3DTA_TEXTURE,
        D3DTA_DIFFUSE,
        0xFF00_FF00,
    );
    // 0.5 * green → ~(0, 128, 0).
    assert!(
        px.r < 25 && (100..=150).contains(&px.g) && px.b < 25,
        "modulate, got {px:?}"
    );
}

#[test]
fn add_texture_and_diffuse() {
    let h = Harness::new();
    let red = solid_texture(&h, 0xFFFF_0000);
    let px = render_stage(
        &h,
        &red,
        D3DTOP_ADD,
        D3DTA_TEXTURE,
        D3DTA_DIFFUSE,
        0xFF00_FF00,
    );
    // red + green → yellow.
    assert!(px.r > 200 && px.g > 200 && px.b < 40, "add, got {px:?}");
}

#[test]
fn selectarg2_ignores_texture() {
    let h = Harness::new();
    let red = solid_texture(&h, 0xFFFF_0000);
    let px = render_stage(
        &h,
        &red,
        D3DTOP_SELECTARG2,
        D3DTA_TEXTURE,
        D3DTA_DIFFUSE,
        0xFF00_FF00,
    );
    // SELECTARG2 → diffuse (green), texture ignored.
    assert!(
        px.r < 40 && px.g > 200 && px.b < 40,
        "selectarg2, got {px:?}"
    );
}

#[test]
fn texturefactor_as_color_source() {
    let h = Harness::new();
    let red = solid_texture(&h, 0xFFFF_0000);
    assert_eq!(
        h.set_render_state(D3DRS_TEXTUREFACTOR, 0xFF00_00FF),
        0,
        "TFACTOR = blue"
    );
    // COLORARG1 = TFACTOR, SELECTARG1 → output the factor, ignoring the texture.
    let px = render_stage(
        &h,
        &red,
        D3DTOP_SELECTARG1,
        D3DTA_TFACTOR,
        D3DTA_DIFFUSE,
        0xFFFF_FFFF,
    );
    assert!(
        px.r < 40 && px.g < 40 && px.b > 200,
        "tfactor blue, got {px:?}"
    );
}

fn dotproduct3_alpha_cascade(
    h: &Harness,
    color_arg: u32,
    alpha_arg: u32,
    diffuse: u32,
    factor: u32,
) -> Rgba8 {
    use mtld3d_types::{D3DTA_ALPHAREPLICATE, D3DTA_CURRENT, D3DTOP_DOTPRODUCT3};

    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, factor), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    for (stage, state, value) in [
        (0, D3DTSS_COLOROP, D3DTOP_DOTPRODUCT3),
        (0, D3DTSS_COLORARG1, color_arg),
        (0, D3DTSS_COLORARG2, D3DTA_TFACTOR),
        (0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (0, D3DTSS_ALPHAARG1, alpha_arg),
        (1, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        (1, D3DTSS_COLORARG1, D3DTA_CURRENT | D3DTA_ALPHAREPLICATE),
        (1, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (1, D3DTSS_ALPHAARG1, D3DTA_CURRENT),
    ] {
        assert_eq!(h.set_texture_stage_state(stage, state, value), 0);
    }
    let verts = quad(diffuse);
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &verts), 0);
    });
    Rgba8::from_pixel(h.read_pixel(320, 240))
}

#[test]
fn dotproduct3_overrides_alpha_and_chains_current() {
    let h = Harness::new();
    let px = dotproduct3_alpha_cascade(&h, D3DTA_DIFFUSE, D3DTA_DIFFUSE, 0x4080_80C0, 0xFFFF_FFFF);
    // The signed RGB dot is 131/255, while the conflicting alpha is 64/255.
    for channel in [px.r, px.g, px.b] {
        assert!(
            channel.abs_diff(131) <= 2,
            "DOTPRODUCT3 alpha in next stage: {px:?}"
        );
    }
}

#[test]
fn dotproduct3_unbound_color_texture_preserves_independent_alpha() {
    let h = Harness::new();
    let px = dotproduct3_alpha_cascade(&h, D3DTA_TEXTURE, D3DTA_TFACTOR, 0x4080_80C0, 0xC0FF_FFFF);
    // An unbound color texture selects CURRENT instead of executing DOTPRODUCT3.
    for channel in [px.r, px.g, px.b] {
        assert_eq!(
            channel, 192,
            "independent factor alpha after unbound color fallback: {px:?}"
        );
    }
}

#[test]
fn dotproduct3_signed_clamp_and_argument_modifiers() {
    use mtld3d_types::{D3DTA_ALPHAREPLICATE, D3DTA_COMPLEMENT, D3DTA_CURRENT, D3DTOP_DOTPRODUCT3};

    let h = Harness::new();
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_CURRENT),
    ] {
        assert_eq!(h.set_texture_stage_state(1, state, value), 0);
    }
    for (name, texel, factor, texture_modifier, factor_modifier, expected) in [
        ("positive", 0x4080_80C0, 0xFFFF_FFFF, 0, 0, 131u8),
        ("negative clamp", 0x40FF_FFFF, 0xFF00_0000, 0, 0, 0),
        ("positive clamp", 0x40FF_FFFF, 0xFFFF_FFFF, 0, 0, 255),
        ("both negative", 0x4000_0000, 0xFF00_0000, 0, 0, 255),
        (
            "complement",
            0x4080_80C0,
            0xFFFF_FFFF,
            D3DTA_COMPLEMENT,
            0,
            0,
        ),
        (
            "both complemented",
            0x4080_80C0,
            0xFFFF_FFFF,
            D3DTA_COMPLEMENT,
            D3DTA_COMPLEMENT,
            131,
        ),
        (
            "alpha replicate",
            0xA020_4080,
            0xFF80_80FF,
            D3DTA_ALPHAREPLICATE,
            0,
            66,
        ),
        (
            "replicate then complement",
            0xA020_4080,
            0xFF80_80FF,
            D3DTA_ALPHAREPLICATE | D3DTA_COMPLEMENT,
            D3DTA_COMPLEMENT,
            66,
        ),
    ] {
        let tex = solid_texture(&h, texel);
        assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, factor), 0);
        // Read color directly, then observe alpha through the next stage's RGB.
        for modifier in [0, D3DTA_ALPHAREPLICATE] {
            assert_eq!(
                h.set_texture_stage_state(1, D3DTSS_COLORARG1, D3DTA_CURRENT | modifier),
                0
            );
            let px = render_stage(
                &h,
                &tex,
                D3DTOP_DOTPRODUCT3,
                D3DTA_TEXTURE | texture_modifier,
                D3DTA_TFACTOR | factor_modifier,
                0x4000_0000,
            );
            for channel in [px.r, px.g, px.b] {
                assert!(
                    channel.abs_diff(expected) <= 2,
                    "{name}, next-stage modifier {modifier:#x}: expected {expected}, got {px:?}"
                );
            }
        }
    }
}

// TEMP is a fragment-local register shared by the fixed-function stage cascade.
fn temp_probe_stage(h: &Harness, stage: u32, color: u32, alpha: u32, result: u32) {
    use mtld3d_types::D3DTSS_RESULTARG;
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        (D3DTSS_COLORARG1, color),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, alpha),
        (D3DTSS_RESULTARG, result),
    ] {
        assert_eq!(h.set_texture_stage_state(stage, state, value), 0);
    }
}

fn temp_probe_pixel(h: &Harness, diffuse: u32) -> u32 {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(diffuse)),
            0
        );
    });
    h.read_pixel(320, 240) & 0x00ff_ffff
}

#[test]
fn texture_stage_temp_zero_initialization() {
    use mtld3d_types::{D3DTA_ALPHAREPLICATE, D3DTA_COMPLEMENT, D3DTA_CURRENT, D3DTA_TEMP};
    let h = Harness::new();
    temp_probe_stage(&h, 0, D3DTA_TEMP, D3DTA_TEMP, D3DTA_CURRENT);
    assert_eq!(
        temp_probe_pixel(&h, 0xff12_3456),
        0,
        "TEMP starts with zero RGB"
    );
    temp_probe_stage(
        &h,
        0,
        D3DTA_TEMP | D3DTA_ALPHAREPLICATE | D3DTA_COMPLEMENT,
        D3DTA_TEMP,
        D3DTA_CURRENT,
    );
    assert_eq!(
        temp_probe_pixel(&h, 0xff12_3456),
        0x00ff_ffff,
        "TEMP starts with zero alpha"
    );
}

#[test]
fn texture_stage_temp_preserves_current() {
    use mtld3d_types::{D3DTA_CURRENT, D3DTA_TEMP};
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0x2010_2030), 0);
    temp_probe_stage(&h, 0, D3DTA_DIFFUSE, D3DTA_DIFFUSE, D3DTA_CURRENT);
    temp_probe_stage(&h, 1, D3DTA_TFACTOR, D3DTA_TFACTOR, D3DTA_TEMP);
    temp_probe_stage(&h, 2, D3DTA_CURRENT, D3DTA_CURRENT, D3DTA_CURRENT);
    assert_eq!(h.set_texture_stage_state(2, D3DTSS_COLOROP, D3DTOP_ADD), 0);
    assert_eq!(
        h.set_texture_stage_state(2, D3DTSS_COLORARG2, D3DTA_TEMP),
        0
    );
    assert_eq!(
        temp_probe_pixel(&h, 0x4020_4060),
        0x0030_6090,
        "TEMP write preserves CURRENT"
    );
}

#[test]
fn texture_stage_temp_preserves_temp_and_resultarg_invalidation() {
    use mtld3d_types::{D3DTA_CURRENT, D3DTA_TEMP, D3DTSS_RESULTARG};
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0x2010_2030), 0);
    temp_probe_stage(&h, 0, D3DTA_TFACTOR, D3DTA_TFACTOR, D3DTA_TEMP);
    temp_probe_stage(&h, 1, D3DTA_DIFFUSE, D3DTA_DIFFUSE, D3DTA_CURRENT);
    temp_probe_stage(&h, 2, D3DTA_TEMP, D3DTA_TEMP, D3DTA_CURRENT);
    assert_eq!(
        temp_probe_pixel(&h, 0x4020_4060),
        0x0010_2030,
        "CURRENT write preserves TEMP"
    );
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_RESULTARG, D3DTA_TEMP),
        0
    );
    assert_eq!(
        temp_probe_pixel(&h, 0x4020_4060),
        0x0020_4060,
        "RESULTARG alone selects a new shader"
    );
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_RESULTARG, D3DTA_CURRENT),
        0
    );
    assert_eq!(
        temp_probe_pixel(&h, 0x4020_4060),
        0x0010_2030,
        "restored destination reuses the right shader"
    );
    assert_eq!(h.set_texture_stage_state(1, D3DTSS_RESULTARG, 0x105), 0);
    assert_eq!(h.texture_stage_state(1, D3DTSS_RESULTARG), 0x105);
    assert_eq!(
        temp_probe_pixel(&h, 0x4020_4060),
        0x0010_2030,
        "invalid RESULTARG keeps its raw getter value but renders the CURRENT default"
    );
}

#[test]
fn texture_stage_temp_same_stage_read_before_write_and_modifiers() {
    use mtld3d_types::{D3DTA_ALPHAREPLICATE, D3DTA_COMPLEMENT, D3DTA_CURRENT, D3DTA_TEMP};
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0x4020_4060), 0);
    temp_probe_stage(&h, 0, D3DTA_TFACTOR, D3DTA_TFACTOR, D3DTA_TEMP);
    temp_probe_stage(
        &h,
        1,
        D3DTA_TEMP | D3DTA_ALPHAREPLICATE,
        D3DTA_TEMP | D3DTA_COMPLEMENT,
        D3DTA_TEMP,
    );
    temp_probe_stage(&h, 2, D3DTA_TEMP, D3DTA_TEMP, D3DTA_CURRENT);
    assert_eq!(
        temp_probe_pixel(&h, 0xff00_0000),
        0x0040_4040,
        "color reads old TEMP alpha before alpha write"
    );
    assert_eq!(
        h.set_texture_stage_state(2, D3DTSS_COLORARG1, D3DTA_TEMP | D3DTA_ALPHAREPLICATE),
        0
    );
    assert_eq!(
        temp_probe_pixel(&h, 0xff00_0000),
        0x00bf_bfbf,
        "alpha complement writes TEMP alpha"
    );
}

#[test]
fn texture_stage_temp_stateblock_destinations_restore_pixels() {
    use mtld3d_types::{
        D3DSBT_ALL, D3DSBT_PIXELSTATE, D3DTA_CURRENT, D3DTA_TEMP, D3DTSS_RESULTARG,
    };
    let h = Harness::new();
    for block_type in [D3DSBT_ALL, D3DSBT_PIXELSTATE] {
        assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0xff10_2030), 0);
        temp_probe_stage(&h, 0, D3DTA_TFACTOR, D3DTA_TFACTOR, D3DTA_TEMP);
        temp_probe_stage(&h, 1, D3DTA_DIFFUSE, D3DTA_DIFFUSE, D3DTA_TEMP);
        temp_probe_stage(&h, 2, D3DTA_TEMP, D3DTA_TEMP, D3DTA_CURRENT);
        let block = h.create_state_block(block_type);
        assert_eq!(temp_probe_pixel(&h, 0xff20_4060), 0x0020_4060);
        assert_eq!(
            h.set_texture_stage_state(1, D3DTSS_RESULTARG, D3DTA_CURRENT),
            0
        );
        assert_eq!(temp_probe_pixel(&h, 0xff20_4060), 0x0010_2030);
        assert_eq!(block.apply(), 0);
        assert_eq!(h.texture_stage_state(1, D3DTSS_RESULTARG), D3DTA_TEMP);
        assert_eq!(temp_probe_pixel(&h, 0xff20_4060), 0x0020_4060);
    }
}

#[test]
fn texture_stage_temp_recorded_destination_does_not_change_live_state() {
    use mtld3d_types::{D3DTA_CURRENT, D3DTA_TEMP, D3DTSS_RESULTARG};
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0xff10_2030), 0);
    temp_probe_stage(&h, 0, D3DTA_TFACTOR, D3DTA_TFACTOR, D3DTA_TEMP);
    temp_probe_stage(&h, 1, D3DTA_DIFFUSE, D3DTA_DIFFUSE, D3DTA_CURRENT);
    temp_probe_stage(&h, 2, D3DTA_TEMP, D3DTA_TEMP, D3DTA_CURRENT);
    assert_eq!(h.begin_state_block(), 0);
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_RESULTARG, D3DTA_TEMP),
        0
    );
    let block = h.end_state_block();
    assert_eq!(h.texture_stage_state(1, D3DTSS_RESULTARG), D3DTA_CURRENT);
    assert_eq!(temp_probe_pixel(&h, 0xff20_4060), 0x0010_2030);
    assert_eq!(block.apply(), 0);
    assert_eq!(h.texture_stage_state(1, D3DTSS_RESULTARG), D3DTA_TEMP);
    assert_eq!(temp_probe_pixel(&h, 0xff20_4060), 0x0020_4060);
}

#[test]
fn texture_stage_temp_draw_initialization_and_reset_defaults() {
    use mtld3d_types::{D3DTA_CURRENT, D3DTA_TEMP, D3DTOP_DISABLE, D3DTSS_RESULTARG};
    let h = Harness::new();
    temp_probe_stage(&h, 0, D3DTA_DIFFUSE, D3DTA_DIFFUSE, D3DTA_TEMP);
    temp_probe_stage(&h, 1, D3DTA_TEMP, D3DTA_TEMP, D3DTA_CURRENT);
    assert_eq!(temp_probe_pixel(&h, 0xff20_4060), 0x0020_4060);
    temp_probe_stage(&h, 0, D3DTA_TEMP, D3DTA_TEMP, D3DTA_CURRENT);
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_COLOROP, D3DTOP_DISABLE),
        0
    );
    assert_eq!(
        temp_probe_pixel(&h, 0xff20_4060),
        0,
        "TEMP is not retained between draws"
    );
    assert_eq!(
        h.set_texture_stage_state(7, D3DTSS_RESULTARG, D3DTA_TEMP),
        0
    );
    assert_eq!(h.reset(640, 480), 0);
    for stage in 0..8 {
        assert_eq!(
            h.texture_stage_state(stage, D3DTSS_RESULTARG),
            D3DTA_CURRENT
        );
    }
    temp_probe_stage(&h, 0, D3DTA_TEMP, D3DTA_TEMP, D3DTA_CURRENT);
    assert_eq!(temp_probe_pixel(&h, 0xff20_4060), 0);
}

#[test]
fn texture_stage_temp_second_arguments_and_texture_binding() {
    use mtld3d_types::{
        D3DTA_ALPHAREPLICATE, D3DTA_COMPLEMENT, D3DTA_CURRENT, D3DTA_TEMP, D3DTSS_ALPHAARG2,
    };
    let h = Harness::new();
    let tex = solid_texture(&h, 0x4020_4060);
    assert_eq!(h.set_texture(0, &tex), 0);
    temp_probe_stage(&h, 0, D3DTA_TEXTURE, D3DTA_TEXTURE, D3DTA_TEMP);
    temp_probe_stage(&h, 1, D3DTA_DIFFUSE, D3DTA_DIFFUSE, D3DTA_TEMP);
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_SELECTARG2),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG2),
        (D3DTSS_COLORARG2, D3DTA_TEMP | D3DTA_COMPLEMENT),
        (D3DTSS_ALPHAARG2, D3DTA_TEMP | D3DTA_COMPLEMENT),
    ] {
        assert_eq!(h.set_texture_stage_state(1, state, value), 0);
    }
    temp_probe_stage(&h, 2, D3DTA_TEMP, D3DTA_TEMP, D3DTA_CURRENT);
    assert_eq!(temp_probe_pixel(&h, 0x8040_2010), 0x00df_bf9f);
    assert_eq!(
        h.set_texture_stage_state(2, D3DTSS_COLORARG1, D3DTA_TEMP | D3DTA_ALPHAREPLICATE),
        0
    );
    assert_eq!(temp_probe_pixel(&h, 0x8040_2010), 0x00bf_bfbf);
    assert_eq!(h.clear_texture(0), 0);
    assert_eq!(
        temp_probe_pixel(&h, 0x8040_2010),
        0x007f_7f7f,
        "unbound stage selects CURRENT before writing TEMP"
    );
}

#[test]
fn texture_stage_temp_dotproduct3_writes_alpha_and_preserves_current() {
    use mtld3d_types::{D3DTA_ALPHAREPLICATE, D3DTA_CURRENT, D3DTA_TEMP, D3DTOP_DOTPRODUCT3};
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0x90ff_8080), 0);
    temp_probe_stage(&h, 0, D3DTA_DIFFUSE, D3DTA_DIFFUSE, D3DTA_TEMP);
    temp_probe_stage(&h, 1, D3DTA_TEMP, D3DTA_DIFFUSE, D3DTA_TEMP);
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_COLOROP, D3DTOP_DOTPRODUCT3),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_COLORARG2, D3DTA_TFACTOR),
        0
    );
    temp_probe_stage(
        &h,
        2,
        D3DTA_TEMP | D3DTA_ALPHAREPLICATE,
        D3DTA_CURRENT,
        D3DTA_CURRENT,
    );
    assert_eq!(
        temp_probe_pixel(&h, 0x20ff_8080),
        0x00ff_ffff,
        "effective DOT3 writes all TEMP channels, ignoring the diffuse alpha operation"
    );
    assert_eq!(
        h.set_texture_stage_state(2, D3DTSS_COLORARG1, D3DTA_CURRENT),
        0
    );
    assert_eq!(
        temp_probe_pixel(&h, 0x20ff_8080),
        0x00ff_8080,
        "DOT3 writing TEMP preserves CURRENT"
    );
}

#[test]
fn texture_stage_temp_unbound_dotproduct3_color_keeps_temp_alpha_read() {
    use mtld3d_types::{
        D3DTA_ALPHAREPLICATE, D3DTA_COMPLEMENT, D3DTA_CURRENT, D3DTA_TEMP, D3DTOP_DOTPRODUCT3,
    };
    let h = Harness::new();
    temp_probe_stage(
        &h,
        0,
        D3DTA_TEXTURE,
        D3DTA_TEMP | D3DTA_COMPLEMENT,
        D3DTA_CURRENT,
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_DOTPRODUCT3),
        0
    );
    temp_probe_stage(
        &h,
        1,
        D3DTA_CURRENT | D3DTA_ALPHAREPLICATE,
        D3DTA_CURRENT,
        D3DTA_CURRENT,
    );
    assert_eq!(
        temp_probe_pixel(&h, 0x2040_6080),
        0x00ff_ffff,
        "unbound color fallback leaves the valid zero-TEMP alpha operation active"
    );
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_COLORARG1, D3DTA_CURRENT),
        0
    );
    assert_eq!(temp_probe_pixel(&h, 0x2040_6080), 0x0040_6080);
}

#[test]
fn per_stage_constant_color_and_modifiers() {
    use mtld3d_types::{D3DTA_ALPHAREPLICATE, D3DTA_COMPLEMENT, D3DTA_CONSTANT, D3DTSS_CONSTANT};

    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_CONSTANT, 0x80A1_B2C3),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    let verts = quad(0xFFFF_FFFF);
    for (modifier, expected) in [
        (0, 0x00A1_B2C3),
        (D3DTA_COMPLEMENT, 0x005E_4D3C),
        (D3DTA_ALPHAREPLICATE, 0x0080_8080),
        (D3DTA_ALPHAREPLICATE | D3DTA_COMPLEMENT, 0x007F_7F7F),
    ] {
        assert_eq!(
            h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_CONSTANT | modifier),
            0
        );
        h.render_once(BLACK, |d| {
            assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &verts), 0);
        });
        let actual = h.read_pixel(320, 240) & 0x00FF_FFFF;
        assert_eq!(
            actual, expected,
            "per-stage constant modifier {modifier:#x}: got {actual:#08x}, expected {expected:#08x}"
        );
    }
}

#[test]
fn per_stage_constant_all_stages_color_alpha_and_factor() {
    use mtld3d_types::{
        D3DBLEND_INVSRCALPHA, D3DBLEND_SRCALPHA, D3DRS_ALPHABLENDENABLE, D3DRS_DESTBLEND,
        D3DRS_SRCBLEND, D3DTA_CONSTANT, D3DTA_CURRENT, D3DTSS_ALPHAARG2, D3DTSS_CONSTANT,
        PrimitiveMiscCaps,
    };
    let h = Harness::new();
    assert_ne!(
        h.device_caps().primitive_misc_caps & PrimitiveMiscCaps::PERSTAGECONSTANT.bits(),
        0
    );
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    for stage in 0..8 {
        h.select_diffuse_stage(stage);
        let color = 0x8000_0000 | ((stage + 1) * 0x0010_2030);
        for (state, value) in [
            (D3DTSS_CONSTANT, color),
            (D3DTSS_COLOROP, D3DTOP_SELECTARG2),
            (D3DTSS_COLORARG2, D3DTA_CONSTANT),
            (D3DTSS_ALPHAOP, D3DTOP_SELECTARG2),
            (D3DTSS_ALPHAARG2, D3DTA_CONSTANT),
        ] {
            assert_eq!(h.set_texture_stage_state(stage, state, value), 0);
        }
        assert_eq!(h.texture_stage_state(stage, D3DTSS_CONSTANT), color);
        assert_eq!(temp_probe_pixel(&h, 0xFFFF_0000), color & 0x00FF_FFFF);
        assert_eq!(
            h.set_texture_stage_state(stage, D3DTSS_COLORARG2, D3DTA_CURRENT),
            0
        );
        assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), 0);
        assert_eq!(h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA), 0);
        assert_eq!(h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA), 0);
        h.render_once(0xFF00_00FF, |d| {
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(0xFFFF_0000)),
                0
            );
        });
        let pixel = Rgba8::from_pixel(h.read_pixel(320, 240));
        assert!(
            pixel.r.abs_diff(128) <= 1 && pixel.g == 0 && pixel.b.abs_diff(127) <= 1,
            "stage {stage} alpha: {pixel:?}"
        );
        assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 0), 0);
        h.select_diffuse_stage(stage);
    }
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0xFF10_2030), 0);
    assert_eq!(
        h.set_texture_stage_state(7, D3DTSS_CONSTANT, 0xFF20_4060),
        0
    );
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_ADD),
        (D3DTSS_COLORARG1, D3DTA_CONSTANT),
        (D3DTSS_COLORARG2, D3DTA_TFACTOR),
    ] {
        assert_eq!(h.set_texture_stage_state(7, state, value), 0);
    }
    assert_eq!(temp_probe_pixel(&h, 0xFFFF_FFFF), 0x0030_6090);
}

#[test]
fn per_stage_constant_updates_preserve_queued_draws() {
    use mtld3d_types::{D3DTA_CONSTANT, D3DTSS_CONSTANT};
    let h = Harness::new();
    h.select_diffuse_stage(0);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_CONSTANT),
        0
    );
    h.render_once(BLACK, |d| {
        for (offset, color) in [(-0.5, 0xFF20_4060), (0.5, 0xFF80_A0C0)] {
            assert_eq!(d.set_texture_stage_state(0, D3DTSS_CONSTANT, color), 0);
            let verts = quad(0xFFFF_FFFF).map(|mut v| {
                v.x = v.x.mul_add(0.5, offset);
                v
            });
            assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &verts), 0);
        }
    });
    assert_eq!(h.read_pixel(160, 240) & 0x00FF_FFFF, 0x0020_4060);
    assert_eq!(h.read_pixel(480, 240) & 0x00FF_FFFF, 0x0080_A0C0);
}

#[test]
fn per_stage_constant_stateblock_reset_and_shader_transitions() {
    use mtld3d_types::{D3DSBT_ALL, D3DSBT_PIXELSTATE, D3DTA_CONSTANT, D3DTSS_CONSTANT};
    const PS: &[u32] = &[
        0xFFFF_0200,
        0x0200_0001,
        0x800F_0800,
        0xA0E4_0000,
        0x0000_FFFF,
    ];
    let h = Harness::new();
    for stage in 0..8 {
        assert_eq!(h.texture_stage_state(stage, D3DTSS_CONSTANT), 0);
    }
    h.select_diffuse_stage(0);
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_CONSTANT),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_CONSTANT, 0xFF12_3456),
        0
    );
    {
        let all = h.create_state_block(D3DSBT_ALL);
        let pixel = h.create_state_block(D3DSBT_PIXELSTATE);
        assert_eq!(
            h.set_texture_stage_state(0, D3DTSS_CONSTANT, 0xFF65_4321),
            0
        );
        assert_eq!(temp_probe_pixel(&h, 0xFFFF_FFFF), 0x0065_4321);
        assert_eq!(pixel.apply(), 0);
        // Preserve the existing PIXEL preset's exclusion of CONSTANT.
        assert_eq!(h.texture_stage_state(0, D3DTSS_CONSTANT), 0xFF65_4321);
        assert_eq!(temp_probe_pixel(&h, 0xFFFF_FFFF), 0x0065_4321);
        assert_eq!(all.apply(), 0);
        assert_eq!(temp_probe_pixel(&h, 0xFFFF_FFFF), 0x0012_3456);
        assert_eq!(h.begin_state_block(), 0);
        assert_eq!(
            h.set_texture_stage_state(0, D3DTSS_CONSTANT, 0xFFAB_CDEF),
            0
        );
        let recorded = h.end_state_block();
        assert_eq!(h.texture_stage_state(0, D3DTSS_CONSTANT), 0xFF12_3456);
        assert_eq!(recorded.apply(), 0);
        assert_eq!(temp_probe_pixel(&h, 0xFFFF_FFFF), 0x00AB_CDEF);
    }
    {
        let ps = h.create_pixel_shader(PS);
        assert_eq!(h.set_pixel_shader(&ps), 0);
        assert_eq!(h.set_pixel_shader_constant_f(0, &[0.0, 1.0, 0.0, 1.0]), 0);
        assert_eq!(temp_probe_pixel(&h, 0xFFFF_FFFF), 0x0000_FF00);
        assert_eq!(
            h.set_texture_stage_state(0, D3DTSS_CONSTANT, 0xFF32_5476),
            0
        );
        assert_eq!(h.clear_pixel_shader(), 0);
        assert_eq!(temp_probe_pixel(&h, 0xFFFF_FFFF), 0x0032_5476);
    }
    assert!(h.set_texture_stage_state(8, D3DTSS_CONSTANT, 0xFFFF_FFFF) < 0);
    assert_eq!(h.texture_stage_state(0, D3DTSS_CONSTANT), 0xFF32_5476);
    assert_eq!(h.reset(640, 480), 0);
    for stage in 0..8 {
        assert_eq!(h.texture_stage_state(stage, D3DTSS_CONSTANT), 0);
    }
    h.select_diffuse_stage(0);
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_CONSTANT),
        0
    );
    assert_eq!(temp_probe_pixel(&h, 0xFFFF_FFFF), 0);
}

#[test]
fn per_stage_constant_extent_tracks_texture_occupancy_and_dot3() {
    use mtld3d_types::{
        D3DTA_ALPHAREPLICATE, D3DTA_CONSTANT, D3DTA_CURRENT, D3DTOP_DOTPRODUCT3, D3DTSS_CONSTANT,
    };
    let h = Harness::new();
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_CONSTANT, 0xC020_4060),
        0
    );
    h.select_diffuse_stage(0);
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_ALPHAARG1, D3DTA_CONSTANT),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_DOTPRODUCT3),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE),
        0
    );
    h.select_diffuse_stage(1);
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_COLORARG1, D3DTA_CURRENT | D3DTA_ALPHAREPLICATE),
        0
    );
    assert_eq!(
        temp_probe_pixel(&h, 0x40FF_FFFF),
        0x00FF_FFFF,
        "DOT3 supplies alpha"
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
        0
    );
    assert_eq!(
        temp_probe_pixel(&h, 0x40FF_FFFF),
        0x00C0_C0C0,
        "unbound fallback exposes constant alpha"
    );
    let tex = solid_texture(&h, 0xFFFF_FFFF);
    assert_eq!(h.set_texture(0, &tex), 0);
    assert_eq!(temp_probe_pixel(&h, 0x40FF_FFFF), 0x00FF_FFFF);
    assert_eq!(h.clear_texture(0), 0);
    assert_eq!(temp_probe_pixel(&h, 0x40FF_FFFF), 0x00C0_C0C0);
}

/// Wine's `texop_test` equation, exercised without its missing-capability skip.
///
/// Texture alpha 0x99 scales the factor RGB before adding the green texture.
#[test]
fn modulate_alpha_add_color_wine_equation() {
    use mtld3d_types::D3DTOP_MODULATEALPHA_ADDCOLOR;
    let h = Harness::new();
    let tex = solid_texture(&h, 0x9900_ff00);
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0xdd33_3333), 0);
    let pixel = render_stage(
        &h,
        &tex,
        D3DTOP_MODULATEALPHA_ADDCOLOR,
        D3DTA_TEXTURE,
        D3DTA_TFACTOR,
        0x55ff_0000,
    );
    assert!(
        pixel.r.abs_diff(0x1f) <= 1 && pixel.g == 255 && pixel.b.abs_diff(0x1f) <= 1,
        "Wine MODULATEALPHA_ADDCOLOR expected RGB 0x1fff1f, got {pixel:?}"
    );
}

fn modulate_alpha_add_stage(h: &Harness, stage: u32, arg1: u32, arg2: u32) {
    use mtld3d_types::D3DTOP_MODULATEALPHA_ADDCOLOR;
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_MODULATEALPHA_ADDCOLOR),
        (D3DTSS_COLORARG1, arg1),
        (D3DTSS_COLORARG2, arg2),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_TFACTOR),
    ] {
        assert_eq!(h.set_texture_stage_state(stage, state, value), 0);
    }
}

#[test]
fn modulate_alpha_add_color_modifiers_and_saturation() {
    use mtld3d_tests::assert_pixel_approx;
    use mtld3d_types::{D3DTA_ALPHAREPLICATE, D3DTA_COMPLEMENT};
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0x7010_2030), 0);
    let tex = solid_texture(&h, 0x4020_4080);
    assert_eq!(h.set_texture(0, &tex), 0);
    for (modifier, expected) in [
        (0, 0x0024_488c),
        (D3DTA_COMPLEMENT, 0x00eb_d7a3),
        (D3DTA_ALPHAREPLICATE, 0x0044_484c),
        (D3DTA_ALPHAREPLICATE | D3DTA_COMPLEMENT, 0x00cb_d7e3),
    ] {
        modulate_alpha_add_stage(&h, 0, D3DTA_TEXTURE | modifier, D3DTA_TFACTOR);
        assert_pixel_approx(
            temp_probe_pixel(&h, 0xff00_0000),
            expected,
            1,
            "modified ARG1 supplies both RGB and alpha multiplier",
        );
    }
    let bright = solid_texture(&h, 0xc0e0_2080);
    assert_eq!(h.set_texture(0, &bright), 0);
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0xff80_8080), 0);
    modulate_alpha_add_stage(&h, 0, D3DTA_TEXTURE, D3DTA_TFACTOR);
    assert_pixel_approx(temp_probe_pixel(&h, 0), 0x00ff_80e0, 1, "RGB saturation");
}

#[test]
fn modulate_alpha_add_color_nontexture_alpha_and_independent_alpha() {
    use mtld3d_tests::assert_pixel_approx;
    use mtld3d_types::{D3DTA_ALPHAREPLICATE, D3DTA_CURRENT, D3DTOP_DISABLE};
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0x7010_2030), 0);
    modulate_alpha_add_stage(&h, 0, D3DTA_DIFFUSE, D3DTA_TFACTOR);
    let low = solid_texture(&h, 0x1022_4466);
    let high = solid_texture(&h, 0xe022_4466);
    for tex in [None, Some(&low), Some(&high)] {
        if let Some(tex) = tex {
            assert_eq!(h.set_texture(0, tex), 0);
        } else {
            assert_eq!(h.clear_texture(0), 0);
        }
        for (diffuse, expected) in [(0x4020_4080, 0x0024_488c), (0xc020_4080, 0x002c_58a4)] {
            assert_eq!(
                h.set_texture_stage_state(1, D3DTSS_COLOROP, D3DTOP_DISABLE),
                0
            );
            assert_pixel_approx(
                temp_probe_pixel(&h, diffuse),
                expected,
                1,
                "ARG1 alpha alone controls RGB",
            );
            temp_probe_stage(
                &h,
                1,
                D3DTA_CURRENT | D3DTA_ALPHAREPLICATE,
                D3DTA_CURRENT,
                D3DTA_CURRENT,
            );
            assert_eq!(
                temp_probe_pixel(&h, diffuse),
                0x0070_7070,
                "independent ALPHAOP survives"
            );
        }
    }
}

#[test]
fn modulate_alpha_add_color_unbound_operands_keep_current_and_alpha() {
    use mtld3d_types::{D3DTA_ALPHAREPLICATE, D3DTA_CURRENT, D3DTOP_DISABLE};
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0x7010_2030), 0);
    for (arg1, arg2) in [
        (D3DTA_TEXTURE, D3DTA_TFACTOR),
        (D3DTA_DIFFUSE, D3DTA_TEXTURE),
    ] {
        modulate_alpha_add_stage(&h, 0, arg1, arg2);
        assert_eq!(
            h.set_texture_stage_state(1, D3DTSS_COLOROP, D3DTOP_DISABLE),
            0
        );
        assert_eq!(temp_probe_pixel(&h, 0x8040_6080), 0x0040_6080);
        temp_probe_stage(
            &h,
            1,
            D3DTA_CURRENT | D3DTA_ALPHAREPLICATE,
            D3DTA_CURRENT,
            D3DTA_CURRENT,
        );
        assert_eq!(temp_probe_pixel(&h, 0x8040_6080), 0x0070_7070);
    }
}

#[test]
fn modulate_alpha_add_color_temp_reads_old_value_and_preserves_current() {
    use mtld3d_tests::assert_pixel_approx;
    use mtld3d_types::{D3DTA_ALPHAREPLICATE, D3DTA_CURRENT, D3DTA_TEMP, D3DTSS_RESULTARG};
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0x8040_2010), 0);
    temp_probe_stage(&h, 0, D3DTA_TFACTOR, D3DTA_TFACTOR, D3DTA_TEMP);
    modulate_alpha_add_stage(&h, 1, D3DTA_TEMP, D3DTA_CURRENT);
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_ALPHAARG1, D3DTA_DIFFUSE),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_RESULTARG, D3DTA_TEMP),
        0
    );
    temp_probe_stage(&h, 2, D3DTA_TEMP, D3DTA_TEMP, D3DTA_CURRENT);
    assert_eq!(h.set_texture_stage_state(2, D3DTSS_COLOROP, D3DTOP_ADD), 0);
    assert_eq!(
        h.set_texture_stage_state(2, D3DTSS_COLORARG2, D3DTA_CURRENT),
        0
    );
    assert_pixel_approx(
        temp_probe_pixel(&h, 0x2020_4060),
        0x0070_80a0,
        1,
        "old TEMP alpha and unchanged CURRENT",
    );
    temp_probe_stage(
        &h,
        2,
        D3DTA_TEMP | D3DTA_ALPHAREPLICATE,
        D3DTA_CURRENT,
        D3DTA_CURRENT,
    );
    assert_eq!(
        temp_probe_pixel(&h, 0x2020_4060),
        0x0020_2020,
        "independent alpha writes TEMP"
    );
}

#[test]
fn modulate_alpha_add_color_uses_its_stage_constant_alpha_without_texture() {
    use mtld3d_tests::assert_pixel_approx;
    use mtld3d_types::{D3DTA_CONSTANT, D3DTA_CURRENT, D3DTSS_CONSTANT};
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0x7010_2030), 0);
    temp_probe_stage(&h, 0, D3DTA_DIFFUSE, D3DTA_DIFFUSE, D3DTA_CURRENT);
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_CONSTANT, 0xff00_ff00),
        0
    );
    modulate_alpha_add_stage(&h, 1, D3DTA_CONSTANT, D3DTA_CURRENT);
    for (constant, expected) in [
        (0x4020_4080, 0x0024_488c),
        (0xc020_4080, 0x002c_58a4),
        (0x4020_4080, 0x0024_488c),
    ] {
        assert_eq!(h.set_texture_stage_state(1, D3DTSS_CONSTANT, constant), 0);
        assert_pixel_approx(
            temp_probe_pixel(&h, 0x8010_2030),
            expected,
            1,
            "stage 1 constant alpha changes RGB independently of stage 0 constant and factor",
        );
    }
}
