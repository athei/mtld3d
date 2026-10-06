use mtld3d_types::{
    D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SCRATCH,
    D3DPOOL_SYSTEMMEM, D3DPRESENT_DONOTWAIT, D3DPRESENT_FORCEIMMEDIATE, D3DUSAGE_DEPTHSTENCIL,
    D3DUSAGE_RENDERTARGET, D3DUSAGE_RESTRICT_SHARED_RESOURCE,
    D3DUSAGE_RESTRICT_SHARED_RESOURCE_DRIVER, D3DUSAGE_RESTRICTED_CONTENT, E_NOTIMPL,
};

use super::*;

fn refusal(verdict: &SharedHandleVerdict) -> &SharedRefusal {
    match verdict {
        SharedHandleVerdict::Refuse(refusal) => refusal,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_null_shared_handle_changes_nothing_on_either_device() {
    for extended in [false, true] {
        for pool in [
            D3DPOOL_DEFAULT,
            D3DPOOL_MANAGED,
            D3DPOOL_SYSTEMMEM,
            D3DPOOL_SCRATCH,
        ] {
            assert_eq!(
                shared_handle_verdict(&CreateKind::Texture { levels: 1 }, pool, false, extended),
                SharedHandleVerdict::Proceed
            );
        }
    }
}

#[test]
fn a_plain_device_refuses_every_shared_handle_with_e_notimpl() {
    let kinds = [
        CreateKind::Texture { levels: 1 },
        CreateKind::CubeTexture,
        CreateKind::VolumeTexture,
        CreateKind::VertexBuffer,
        CreateKind::IndexBuffer,
        CreateKind::RenderTarget,
        CreateKind::DepthStencil,
        CreateKind::OffscreenPlain,
    ];
    for kind in &kinds {
        for pool in [D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM] {
            let verdict = shared_handle_verdict(kind, pool, true, false);
            assert_eq!(refusal(&verdict), &SharedRefusal::PlainDevice);
            assert_eq!(refusal(&verdict).hresult(), E_NOTIMPL);
        }
    }
}

#[test]
fn an_extended_device_takes_user_memory_for_one_level_system_memory_textures_and_plains() {
    assert_eq!(
        shared_handle_verdict(
            &CreateKind::Texture { levels: 1 },
            D3DPOOL_SYSTEMMEM,
            true,
            true
        ),
        SharedHandleVerdict::UserMemory
    );
    assert_eq!(
        shared_handle_verdict(&CreateKind::OffscreenPlain, D3DPOOL_SYSTEMMEM, true, true),
        SharedHandleVerdict::UserMemory
    );
}

#[test]
fn user_memory_outside_its_one_shape_is_an_invalid_call() {
    for levels in [0, 2] {
        let verdict = shared_handle_verdict(
            &CreateKind::Texture { levels },
            D3DPOOL_SYSTEMMEM,
            true,
            true,
        );
        assert_eq!(
            refusal(&verdict).hresult(),
            D3DERR_INVALIDCALL,
            "levels {levels}"
        );
    }
    for kind in [CreateKind::CubeTexture, CreateKind::VolumeTexture] {
        let verdict = shared_handle_verdict(&kind, D3DPOOL_SYSTEMMEM, true, true);
        assert_eq!(refusal(&verdict).hresult(), D3DERR_INVALIDCALL, "{kind:?}");
    }
    for kind in [
        CreateKind::Texture { levels: 1 },
        CreateKind::OffscreenPlain,
    ] {
        let verdict = shared_handle_verdict(&kind, D3DPOOL_SCRATCH, true, true);
        assert_eq!(refusal(&verdict).hresult(), D3DERR_INVALIDCALL, "{kind:?}");
    }
}

#[test]
fn buffers_outside_the_default_pool_take_no_user_memory() {
    for kind in [CreateKind::VertexBuffer, CreateKind::IndexBuffer] {
        let verdict = shared_handle_verdict(&kind, D3DPOOL_SYSTEMMEM, true, true);
        assert_eq!(refusal(&verdict), &SharedRefusal::BufferUserMemory);
        assert_eq!(refusal(&verdict).hresult(), D3DERR_NOTAVAILABLE);
    }
}

#[test]
fn a_shared_default_pool_resource_is_not_available() {
    for kind in [
        CreateKind::Texture { levels: 1 },
        CreateKind::CubeTexture,
        CreateKind::VolumeTexture,
        CreateKind::VertexBuffer,
        CreateKind::IndexBuffer,
        CreateKind::RenderTarget,
        CreateKind::DepthStencil,
        CreateKind::OffscreenPlain,
    ] {
        let verdict = shared_handle_verdict(&kind, D3DPOOL_DEFAULT, true, true);
        assert_eq!(
            refusal(&verdict),
            &SharedRefusal::SharedResource,
            "{kind:?}"
        );
        assert_eq!(refusal(&verdict).hresult(), D3DERR_NOTAVAILABLE);
        assert!(!refusal(&verdict).reason().is_empty());
    }
}

#[test]
fn the_ex_surface_creates_take_only_the_restriction_usages() {
    assert!(ex_create_usage_valid(0, false));
    assert!(ex_create_usage_valid(D3DUSAGE_RESTRICTED_CONTENT, false));
    assert!(!ex_create_usage_valid(D3DUSAGE_RENDERTARGET, false));
    assert!(!ex_create_usage_valid(D3DUSAGE_DEPTHSTENCIL, false));
    for restriction in [
        D3DUSAGE_RESTRICT_SHARED_RESOURCE,
        D3DUSAGE_RESTRICT_SHARED_RESOURCE_DRIVER,
    ] {
        assert!(!ex_create_usage_valid(restriction, false));
        assert!(ex_create_usage_valid(restriction, true));
    }
}

#[test]
fn reset_ex_names_a_mode_exactly_when_fullscreen_and_at_the_back_buffer_size() {
    assert!(reset_ex_mode_valid(true, None, (400, 300)));
    assert!(!reset_ex_mode_valid(true, Some((400, 300)), (400, 300)));
    assert!(!reset_ex_mode_valid(false, None, (800, 600)));
    assert!(reset_ex_mode_valid(false, Some((800, 600)), (800, 600)));
    assert!(!reset_ex_mode_valid(false, Some((800, 600)), (799, 600)));
    assert!(!reset_ex_mode_valid(false, Some((800, 600)), (800, 599)));
    assert!(!reset_ex_mode_valid(false, Some((0, 0)), (800, 600)));
}

#[test]
fn frame_latency_defaults_to_three_and_stops_at_thirty() {
    assert_eq!(frame_latency(0), Some(DEFAULT_FRAME_LATENCY));
    assert_eq!(frame_latency(1), Some(1));
    assert_eq!(frame_latency(MAX_FRAME_LATENCY), Some(30));
    assert_eq!(frame_latency(MAX_FRAME_LATENCY + 1), None);
}

#[test]
fn present_flags_are_named_bit_by_bit() {
    let flags = D3DPRESENT_DONOTWAIT | D3DPRESENT_FORCEIMMEDIATE | 0x8000;
    let bits: Vec<u32> = present_flag_bits(flags).collect();
    assert_eq!(
        bits,
        [D3DPRESENT_DONOTWAIT, D3DPRESENT_FORCEIMMEDIATE, 0x8000]
    );
    assert_eq!(
        present_flag_name(D3DPRESENT_DONOTWAIT),
        "D3DPRESENT_DONOTWAIT"
    );
    assert_eq!(
        present_flag_name(D3DPRESENT_FORCEIMMEDIATE),
        "D3DPRESENT_FORCEIMMEDIATE"
    );
    assert_eq!(present_flag_name(0x8000), "an unknown D3DPRESENT flag");
    assert_eq!(present_flag_bits(0).count(), 0);
}

#[test]
fn packed_rows_are_texels_or_blocks_without_padding() {
    // L8, 33 wide: 33 bytes a row, where the lock pitch rounds to 36.
    let l8 = PackedRows::of_level(33, 33, 1, (1, 1, 1));
    assert_eq!(
        l8,
        PackedRows {
            row_bytes: 33,
            rows: 33
        }
    );
    assert_eq!(l8.total_bytes(), 33 * 33);
    // DXT1 at 8x12: two blocks of eight bytes a row, three block rows.
    let dxt1 = PackedRows::of_level(8, 12, 0, (4, 4, 8));
    assert_eq!(
        dxt1,
        PackedRows {
            row_bytes: 16,
            rows: 3
        }
    );
}

#[test]
fn packed_rows_land_at_the_destination_pitch() {
    let rows = PackedRows {
        row_bytes: 3,
        rows: 2,
    };
    let src = [1u8, 2, 3, 4, 5, 6];
    let mut dst = [0u8; 8];
    assert!(copy_packed_rows(&src, &rows, &mut dst, 4));
    assert_eq!(dst, [1, 2, 3, 0, 4, 5, 6, 0]);
}

#[test]
fn a_short_side_copies_nothing() {
    let rows = PackedRows {
        row_bytes: 3,
        rows: 2,
    };
    let mut dst = [0u8; 8];
    assert!(!copy_packed_rows(&[1, 2, 3], &rows, &mut dst, 4));
    assert!(!copy_packed_rows(&[0; 6], &rows, &mut dst[..6], 4));
    assert!(!copy_packed_rows(&[0; 6], &rows, &mut dst, 2));
    assert_eq!(dst, [0; 8]);
    // The last row needs only its own bytes, not a whole pitch.
    assert!(copy_packed_rows(&[9; 6], &rows, &mut dst[..7], 4));
}

#[test]
fn an_extended_device_reports_the_whole_texture_budget() {
    assert_eq!(available_texture_mem(1 << 30, 1 << 20, true), 1 << 30);
    assert_eq!(
        available_texture_mem(1 << 30, 1 << 20, false),
        (1 << 30) - (1 << 20)
    );
    assert_eq!(available_texture_mem(1 << 34, 0, true), u32::MAX);
    assert_eq!(available_texture_mem(1 << 20, 1 << 30, false), 0);
}
