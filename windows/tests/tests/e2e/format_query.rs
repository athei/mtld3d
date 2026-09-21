//! The `CheckDeviceFormat` usage answers that are rules, not format lists.

use mtld3d_tests::Harness;
use mtld3d_types::{
    D3D_OK, D3DERR_NOTAVAILABLE, D3DFMT_A1R5G5B5, D3DFMT_A2B10G10R10, D3DFMT_A2R10G10B10,
    D3DFMT_A4R4G4B4, D3DFMT_A8, D3DFMT_A8B8G8R8, D3DFMT_A8L8, D3DFMT_A8R8G8B8, D3DFMT_A16B16G16R16,
    D3DFMT_A16B16G16R16F, D3DFMT_A32B32G32R32F, D3DFMT_ATI1, D3DFMT_DXT1, D3DFMT_DXT2, D3DFMT_DXT3,
    D3DFMT_DXT4, D3DFMT_DXT5, D3DFMT_G16R16, D3DFMT_L8, D3DFMT_L16, D3DFMT_Q8W8V8U8,
    D3DFMT_Q16W16V16U16, D3DFMT_R5G6B5, D3DFMT_R8G8B8, D3DFMT_R16F, D3DFMT_R32F, D3DFMT_V8U8,
    D3DFMT_V16U16, D3DFMT_X1R5G5B5, D3DFMT_X8B8G8R8, D3DFMT_X8R8G8B8, D3DRTYPE_CUBETEXTURE,
    D3DRTYPE_SURFACE, D3DRTYPE_TEXTURE, D3DRTYPE_VOLUME, D3DRTYPE_VOLUMETEXTURE,
    D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_QUERY_LEGACYBUMPMAP, D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING,
    D3DUSAGE_QUERY_SRGBREAD, D3DUSAGE_QUERY_SRGBWRITE, D3DUSAGE_RENDERTARGET,
};

/// The mapped colour formats, each with the name a failure prints.
///
/// The spread is what makes the two rules below rules: colour attachments
/// with and without an sRGB twin, sampled formats with and without one, the
/// signed and packed-ten-bit families, and the two packed 16-bit members
/// whose render-target answer depends on the device.
const FORMATS: [(u32, &str); 31] = [
    (D3DFMT_A8R8G8B8, "A8R8G8B8"),
    (D3DFMT_X8R8G8B8, "X8R8G8B8"),
    (D3DFMT_A8B8G8R8, "A8B8G8R8"),
    (D3DFMT_X8B8G8R8, "X8B8G8R8"),
    (D3DFMT_R8G8B8, "R8G8B8"),
    (D3DFMT_R5G6B5, "R5G6B5"),
    (D3DFMT_A1R5G5B5, "A1R5G5B5"),
    (D3DFMT_X1R5G5B5, "X1R5G5B5"),
    (D3DFMT_A4R4G4B4, "A4R4G4B4"),
    (D3DFMT_A8, "A8"),
    (D3DFMT_A8L8, "A8L8"),
    (D3DFMT_L8, "L8"),
    (D3DFMT_L16, "L16"),
    (D3DFMT_G16R16, "G16R16"),
    (D3DFMT_A16B16G16R16, "A16B16G16R16"),
    (D3DFMT_R16F, "R16F"),
    (D3DFMT_A16B16G16R16F, "A16B16G16R16F"),
    (D3DFMT_R32F, "R32F"),
    (D3DFMT_A32B32G32R32F, "A32B32G32R32F"),
    (D3DFMT_ATI1, "ATI1"),
    (D3DFMT_V8U8, "V8U8"),
    (D3DFMT_V16U16, "V16U16"),
    (D3DFMT_Q8W8V8U8, "Q8W8V8U8"),
    (D3DFMT_Q16W16V16U16, "Q16W16V16U16"),
    (D3DFMT_A2R10G10B10, "A2R10G10B10"),
    (D3DFMT_A2B10G10R10, "A2B10G10R10"),
    (D3DFMT_DXT1, "DXT1"),
    (D3DFMT_DXT2, "DXT2"),
    (D3DFMT_DXT3, "DXT3"),
    (D3DFMT_DXT4, "DXT4"),
    (D3DFMT_DXT5, "DXT5"),
];

/// The sRGB-write query answers the render-target question, for every format.
///
/// The encode belongs to the render pass, so a format that can be a colour
/// attachment takes it and one that cannot has nothing to encode into. The
/// expected answer is read from the device's own `D3DUSAGE_RENDERTARGET`
/// answer rather than spelled out, so the case holds on a GPU family without
/// the native packed 16-bit formats too.
#[test]
fn srgb_write_queries_answer_the_render_target_question() {
    let h = Harness::factory_only();
    for (format, name) in FORMATS {
        let renderable = h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_RENDERTARGET,
            D3DRTYPE_TEXTURE,
            format,
        ) == D3D_OK;
        let expected = if renderable {
            D3D_OK
        } else {
            D3DERR_NOTAVAILABLE
        };
        for extra in [
            0,
            D3DUSAGE_RENDERTARGET,
            D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING,
            // A rejected sRGB write may not come back as the NOAUTOGEN
            // success of the mip-generation half of a combined query.
            D3DUSAGE_AUTOGENMIPMAP,
        ] {
            assert_eq!(
                h.check_device_format(
                    D3DFMT_X8R8G8B8,
                    D3DUSAGE_QUERY_SRGBWRITE | extra,
                    D3DRTYPE_TEXTURE,
                    format
                ),
                expected,
                "{name} texture sRGB write with {extra:#x}"
            );
        }
    }
}

/// The sRGB read and the sRGB write questions are answered separately.
///
/// They were one question while the write answer came off the twin table.
/// DXT1 has the twin view its `D3DSAMP_SRGBTEXTURE` decode needs and is no
/// colour attachment; R32F is a colour attachment whose Metal format has no
/// twin, so its write encodes in the pixel shader and its read has nothing
/// to decode with. Each answers yes to one of the two.
#[test]
fn srgb_read_and_write_queries_are_independent() {
    let h = Harness::factory_only();
    let check = |usage: u32, format: u32| {
        h.check_device_format(D3DFMT_X8R8G8B8, usage, D3DRTYPE_TEXTURE, format)
    };
    for format in [
        D3DFMT_DXT1,
        D3DFMT_DXT2,
        D3DFMT_DXT3,
        D3DFMT_DXT4,
        D3DFMT_DXT5,
    ] {
        assert_eq!(check(D3DUSAGE_QUERY_SRGBREAD, format), D3D_OK);
        assert_eq!(check(D3DUSAGE_QUERY_SRGBWRITE, format), D3DERR_NOTAVAILABLE);
    }
    for format in [D3DFMT_R32F, D3DFMT_A16B16G16R16F, D3DFMT_G16R16] {
        assert_eq!(check(D3DUSAGE_QUERY_SRGBREAD, format), D3DERR_NOTAVAILABLE);
        assert_eq!(check(D3DUSAGE_QUERY_SRGBWRITE, format), D3D_OK);
    }
    // The pair an engine probes together before it commits to a
    // gamma-correct pipeline still answers both halves.
    assert_eq!(
        check(
            D3DUSAGE_QUERY_SRGBREAD | D3DUSAGE_QUERY_SRGBWRITE,
            D3DFMT_A8R8G8B8
        ),
        D3D_OK
    );
}

/// No format and no resource type answers the legacy bump-map query.
///
/// `D3DCAPS9::TextureOpCaps` advertises neither `BUMPENVMAP` nor
/// `BUMPENVMAPLUMINANCE`, so an application that probes for legacy bump
/// mapping is told no rather than told yes and then left without an
/// operation to use the format with. The signed formats hardware of the era
/// advertised are no exception.
#[test]
fn legacy_bump_map_queries_are_unavailable() {
    let h = Harness::factory_only();
    for (format, name) in FORMATS {
        for rtype in [
            D3DRTYPE_SURFACE,
            D3DRTYPE_TEXTURE,
            D3DRTYPE_CUBETEXTURE,
            D3DRTYPE_VOLUME,
            D3DRTYPE_VOLUMETEXTURE,
        ] {
            for extra in [0, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_QUERY_SRGBREAD] {
                assert_eq!(
                    h.check_device_format(
                        D3DFMT_X8R8G8B8,
                        D3DUSAGE_QUERY_LEGACYBUMPMAP | extra,
                        rtype,
                        format
                    ),
                    D3DERR_NOTAVAILABLE,
                    "{name} rtype {rtype} legacy bump map with {extra:#x}"
                );
            }
        }
    }
}
