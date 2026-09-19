use super::*;

#[test]
fn commands_latch_and_texture_eligibility_is_independent() {
    let mut state = Fetch4State::new();
    state.set_sampler(3, D3DSAMP_MIPMAPLODBIAS as usize, FETCH4_ENABLE);
    assert_eq!(state.masks(), (0, 0));
    state.set_texture(3, Some(D3DFMT_L8), true);
    assert_eq!(state.masks(), (8, 0));
    state.set_sampler(3, D3DSAMP_MIPMAPLODBIAS as usize, 2.0f32.to_bits());
    assert_eq!(state.masks(), (8, 0));
    state.set_texture(3, Some(D3DFMT_A8), true);
    assert_eq!(state.masks(), (8, 8));
    state.set_sampler(3, D3DSAMP_MAGFILTER as usize, mtld3d_types::D3DTEXF_LINEAR);
    assert_eq!(state.masks(), (0, 0));
    state.set_sampler(3, D3DSAMP_MAGFILTER as usize, D3DTEXF_POINT);
    assert_eq!(state.masks(), (8, 8));
    state.set_sampler(3, D3DSAMP_MIPMAPLODBIAS as usize, FETCH4_DISABLE);
    assert_eq!(state.masks(), (0, 0));
}

#[test]
fn unsupported_formats_and_dimensions_keep_ordinary_sampling() {
    let mut state = Fetch4State::new();
    state.restore_enabled(u16::MAX);
    for format in [D3DFMT_L8, D3DFMT_A8, D3DFMT_DF24] {
        state.set_texture(0, Some(format), false);
        assert_eq!(state.masks(), (0, 0));
    }
    for format in [
        mtld3d_types::D3DFMT_ATI1,
        mtld3d_types::D3DFMT_A8R8G8B8,
        mtld3d_types::D3DFMT_D24S8,
    ] {
        state.set_texture(0, Some(format), true);
        assert_eq!(state.masks(), (0, 0));
    }
}
