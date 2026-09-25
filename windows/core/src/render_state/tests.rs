use mtld3d_types::{
    D3DBLEND_BOTHSRCALPHA, D3DBLEND_INVSRCCOLOR2, D3DBLEND_SRCALPHA, D3DCMP_GREATER, D3DCULL_CW,
    D3DSTENCILOP_INCR, render_state_defaults,
};

use super::*;

/// Every render state this module classifies as an enum.
const ENUM_STATES: [u32; 18] = [
    D3DRS_ZFUNC,
    D3DRS_ALPHAFUNC,
    D3DRS_STENCILFUNC,
    D3DRS_CCW_STENCILFUNC,
    D3DRS_SRCBLEND,
    D3DRS_SRCBLENDALPHA,
    D3DRS_DESTBLEND,
    D3DRS_DESTBLENDALPHA,
    D3DRS_BLENDOP,
    D3DRS_BLENDOPALPHA,
    D3DRS_CULLMODE,
    D3DRS_FILLMODE,
    D3DRS_STENCILFAIL,
    D3DRS_STENCILZFAIL,
    D3DRS_STENCILPASS,
    D3DRS_CCW_STENCILFAIL,
    D3DRS_CCW_STENCILZFAIL,
    D3DRS_CCW_STENCILPASS,
];

/// Every render state this module classifies as a bit mask.
const MASK_STATES: [u32; 4] = [
    D3DRS_COLORWRITEENABLE,
    D3DRS_COLORWRITEENABLE1,
    D3DRS_COLORWRITEENABLE2,
    D3DRS_COLORWRITEENABLE3,
];

fn narrowed(state: u32, value: u32) -> u8 {
    let mut rs = render_state_defaults();
    rs[state as usize] = value;
    enum_value(&rs, state)
}

#[test]
fn spec_defaults_pass_through() {
    let rs = render_state_defaults();
    for state in ENUM_STATES.iter().chain(&MASK_STATES).copied() {
        let expected = u8::try_from(rs[state as usize]).expect("spec default fits a byte");
        assert_eq!(enum_value(&rs, state), expected, "D3DRS_{state} default");
    }
}

#[test]
fn out_of_range_reads_as_the_spec_default() {
    let defaults = render_state_defaults();
    for state in ENUM_STATES {
        let expected = u8::try_from(defaults[state as usize]).expect("spec default fits a byte");
        // A value wider than a byte is the write that used to end the process.
        assert_eq!(narrowed(state, 0x1_0000), expected, "D3DRS_{state} wide");
        assert_eq!(
            narrowed(state, u32::MAX),
            expected,
            "D3DRS_{state} all ones"
        );
    }
}

#[test]
fn values_inside_a_space_pass_through() {
    assert_eq!(narrowed(D3DRS_ZFUNC, D3DCMP_GREATER), 5);
    assert_eq!(narrowed(D3DRS_CULLMODE, D3DCULL_CW), 2);
    assert_eq!(narrowed(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA), 5);
    assert_eq!(narrowed(D3DRS_STENCILPASS, D3DSTENCILOP_INCR), 7);
}

#[test]
fn blend_space_covers_the_factors_the_layer_does_not_translate() {
    // `D3DBLEND_BOTHSRCALPHA` and the dual-source factors are D3D9 values, so
    // they reach the translation's own unmapped arm rather than being
    // replaced here.
    assert_eq!(narrowed(D3DRS_SRCBLEND, D3DBLEND_BOTHSRCALPHA), 12);
    assert_eq!(narrowed(D3DRS_DESTBLEND, D3DBLEND_INVSRCCOLOR2), 17);
    assert_eq!(
        narrowed(D3DRS_SRCBLEND, D3DBLEND_INVSRCCOLOR2 + 1),
        BLEND_ONE
    );
}

#[test]
fn zero_is_outside_every_enum_space() {
    // D3D9 numbers each of these spaces from one, so zero is a garbage write
    // even though it fits a byte.
    assert_eq!(narrowed(D3DRS_BLENDOPALPHA, 0), BLENDOP_ADD);
    assert_eq!(narrowed(D3DRS_CULLMODE, 0), CULL_CCW);
    assert_eq!(narrowed(D3DRS_STENCILFAIL, 0), STENCILOP_KEEP);
    assert_eq!(narrowed(D3DRS_ZFUNC, 0), CMP_LESSEQUAL);
}

#[test]
fn colour_write_masks_drop_bits_they_do_not_name() {
    assert_eq!(narrowed(D3DRS_COLORWRITEENABLE, 0x0000_0007), 0x07);
    assert_eq!(narrowed(D3DRS_COLORWRITEENABLE, 0x0000_001F), 0x0F);
    assert_eq!(narrowed(D3DRS_COLORWRITEENABLE1, 0xFFFF_FF00), 0x00);
    assert_eq!(narrowed(D3DRS_COLORWRITEENABLE3, 0x1_0009), 0x09);
}

#[test]
fn an_unclassified_state_reads_as_its_low_byte() {
    assert_eq!(narrowed(mtld3d_types::D3DRS_STENCILREF, 0x42), 0x42);
}

#[test]
fn shade_mode_is_consumed() {
    // SHADEMODE keys `VariantFlags::FLAT_SHADE`, which puts `[[flat]]` on the
    // colour varyings, so a FLAT write must not fire the not-consumed warn.
    assert!(
        matches!(
            rs_classify(mtld3d_types::D3DRS_SHADEMODE, mtld3d_types::D3DSHADE_FLAT),
            RsClass::Consumed
        ),
        "D3DRS_SHADEMODE is not classified Consumed"
    );
}

#[test]
fn legacy_render_states_are_obsolete() {
    // Dithering has no effect on the 8-bit and wider targets rendered to,
    // antialiased lines have no Metal rasterizer mode and no advertised cap,
    // and the tessellation slots belong to RT-patch and N-patch tessellation,
    // which is not implemented. Each logs at info, never as a gap.
    let one = 1.0f32.to_bits();
    let writes = [
        (mtld3d_types::D3DRS_DITHERENABLE, 1),
        (mtld3d_types::D3DRS_ANTIALIASEDLINEENABLE, 1),
        (mtld3d_types::D3DRS_MINTESSELLATIONLEVEL, 2.0f32.to_bits()),
        (mtld3d_types::D3DRS_MAXTESSELLATIONLEVEL, 4.0f32.to_bits()),
        (mtld3d_types::D3DRS_ADAPTIVETESS_X, one),
        (mtld3d_types::D3DRS_ADAPTIVETESS_Y, one),
        (mtld3d_types::D3DRS_ADAPTIVETESS_Z, 0),
        (mtld3d_types::D3DRS_ADAPTIVETESS_W, one),
        (mtld3d_types::D3DRS_ENABLEADAPTIVETESSELLATION, 1),
    ];
    for (index, value) in writes {
        assert!(
            matches!(rs_classify(index, value), RsClass::Obsolete(_)),
            "D3DRS_{index} = {value:#x} is not classified Obsolete"
        );
    }
}

#[test]
fn vendor_tokens_on_the_tessellation_slots_keep_their_class() {
    // ATOC on ADAPTIVETESS_Y is alpha to coverage, which is implemented.
    // NVDB on ADAPTIVETESS_X asks for the depth-bounds test, which is a
    // missing feature rather than an obsolete one, so it stays a warning.
    assert!(matches!(
        rs_classify(D3DRS_ADAPTIVETESS_Y, D3DFMT_ATOC),
        RsClass::Consumed
    ));
    assert!(matches!(
        rs_classify(
            mtld3d_types::D3DRS_ADAPTIVETESS_X,
            mtld3d_types::D3DFMT_NVDB
        ),
        RsClass::NotImplemented
    ));
}

#[test]
fn unconsumed_render_states_are_the_known_gaps() {
    // No reader exists for the last-pixel rule or for cylindrical texture
    // wrapping; both are D3D9 features rather than obsolete ones, so a
    // non-default write keeps warning.
    let gaps = [
        mtld3d_types::D3DRS_LASTPIXEL,
        mtld3d_types::D3DRS_WRAP0,
        mtld3d_types::D3DRS_WRAP7,
        mtld3d_types::D3DRS_WRAP8,
        mtld3d_types::D3DRS_WRAP15,
    ];
    for index in gaps {
        assert!(
            matches!(rs_classify(index, 0), RsClass::NotImplemented),
            "D3DRS_{index} is not classified NotImplemented"
        );
    }
}
