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
/// Returns the centre pixel. Alpha always comes from the texture.
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
