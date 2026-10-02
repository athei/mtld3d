use super::{EncoderControl, EncoderOpcode, EncoderSubmitMode};
use crate::encoder_wire::WireError;

#[test]
fn operation_tags_keep_the_existing_wire_numbers() {
    let operations = [
        (1, EncoderOpcode::SetVsConstRange),
        (2, EncoderOpcode::SetPsConstRange),
        (3, EncoderOpcode::SetFfVsConstRange),
        (4, EncoderOpcode::Draw),
        (6, EncoderOpcode::SetViewport),
        (7, EncoderOpcode::SetVertexSampler),
        (8, EncoderOpcode::SetVertexTexture),
        (9, EncoderOpcode::BindDepth),
        (10, EncoderOpcode::BindColor),
        (11, EncoderOpcode::GenerateMipmapsOrdered),
        (12, EncoderOpcode::UnbindExtraColor),
        (13, EncoderOpcode::DestroyTexture),
        (14, EncoderOpcode::ReadColorHandle),
        (15, EncoderOpcode::NoteColorRead),
        (16, EncoderOpcode::ResolveDepthSurface),
        (17, EncoderOpcode::StretchBlit),
        (18, EncoderOpcode::ColorFill),
        (19, EncoderOpcode::CarryDepth),
        (20, EncoderOpcode::ClearColor),
        (21, EncoderOpcode::ClearColorRects),
        (22, EncoderOpcode::ClearDepthStencilRects),
        (23, EncoderOpcode::ClearDepthStencil),
        (24, EncoderOpcode::ResolveDynamicDepth),
        (25, EncoderOpcode::ResolveDepthTexture),
        (26, EncoderOpcode::ReadDeviceBuffer),
        (27, EncoderOpcode::AdoptProgram),
        (28, EncoderOpcode::BeginVisibility),
        (29, EncoderOpcode::EndVisibility),
        (30, EncoderOpcode::RetireColor),
        (31, EncoderOpcode::RetireDepth),
        (32, EncoderOpcode::UploadColor),
        (33, EncoderOpcode::UploadResampled),
        (34, EncoderOpcode::ReadTextureHandle),
        (35, EncoderOpcode::GenerateMipmaps),
        (36, EncoderOpcode::ReadTextureColorHandle),
        (37, EncoderOpcode::UploadTextureAndMips),
        (38, EncoderOpcode::UploadTexture),
        (39, EncoderOpcode::SetDumpDraw),
        (40, EncoderOpcode::StageUpload),
        (41, EncoderOpcode::SetSnapshot),
        (42, EncoderOpcode::WarmupTexture),
        (43, EncoderOpcode::WarmupBuffer),
        (45, EncoderOpcode::RetainVbib),
        (46, EncoderOpcode::SetLayerPacing),
        (47, EncoderOpcode::SetGamma),
        (48, EncoderOpcode::UpdateColorRegion),
        (49, EncoderOpcode::DestroyBuffer),
    ];
    for (raw, operation) in operations {
        assert_eq!(u16::from(operation), raw);
        assert_eq!(u16::from(EncoderOpcode::try_from(raw).unwrap()), raw);
    }
}

#[test]
fn every_unknown_operation_tag_is_rejected_before_construction() {
    for raw in u16::MIN..=u16::MAX {
        let valid = (1..=49).contains(&raw) && !matches!(raw, 5 | 44);
        if valid {
            assert!(EncoderOpcode::try_from(raw).is_ok());
        } else {
            assert!(matches!(
                EncoderOpcode::try_from(raw),
                Err(WireError::InvalidValue)
            ));
        }
    }
}

#[test]
fn lifecycle_values_preserve_zero_one_two_and_reject_other_integers() {
    let modes = [
        EncoderSubmitMode::Queue,
        EncoderSubmitMode::WaitForSubmit,
        EncoderSubmitMode::WaitForGpu,
    ];
    for (index, mode) in modes.into_iter().enumerate() {
        let raw = u32::try_from(index).unwrap();
        assert_eq!(u32::from(mode), raw);
        assert_eq!(u32::from(EncoderSubmitMode::try_from(raw).unwrap()), raw);
    }
    let controls = [
        EncoderControl::DrainRetention,
        EncoderControl::IntakeVisibility,
        EncoderControl::Reset,
    ];
    for (index, control) in controls.into_iter().enumerate() {
        let raw = u32::try_from(index).unwrap();
        assert_eq!(u32::from(control), raw);
        assert_eq!(u32::from(EncoderControl::try_from(raw).unwrap()), raw);
    }
    for raw in [3, 4, u32::from(u16::MAX), u32::MAX] {
        assert!(matches!(
            EncoderSubmitMode::try_from(raw),
            Err(WireError::InvalidValue)
        ));
        assert!(matches!(
            EncoderControl::try_from(raw),
            Err(WireError::InvalidValue)
        ));
    }
}
