//! Symbolic values in the internal encoder frame and lifecycle protocol.
//!
//! Wire records retain raw integers until validation succeeds. These types never
//! reinterpret unchecked bytes as an enum, and matching PE/Unix binaries agree
//! on every explicit discriminant below.

use strum::FromRepr;

use crate::encoder_wire::WireError;

#[cfg(test)]
mod tests;

/// One frame operation. Tag 5 was retired with the executable snapshot handoff.
#[repr(u16)]
#[derive(FromRepr)]
pub enum EncoderOpcode {
    SetVsConstRange = 1,
    SetPsConstRange = 2,
    SetFfVsConstRange = 3,
    Draw = 4,
    SetViewport = 6,
    SetVertexSampler = 7,
    SetVertexTexture = 8,
    BindDepth = 9,
    BindColor = 10,
    GenerateMipmapsOrdered = 11,
    UnbindExtraColor = 12,
    DestroyTexture = 13,
    ReadColorHandle = 14,
    NoteColorRead = 15,
    ResolveDepthSurface = 16,
    StretchBlit = 17,
    ColorFill = 18,
    CarryDepth = 19,
    ClearColor = 20,
    ClearColorRects = 21,
    ClearDepthStencilRects = 22,
    ClearDepthStencil = 23,
    ResolveDynamicDepth = 24,
    ResolveDepthTexture = 25,
    ReadDeviceBuffer = 26,
    AdoptProgram = 27,
    BeginVisibility = 28,
    EndVisibility = 29,
    RetireColor = 30,
    RetireDepth = 31,
    UploadColor = 32,
    UploadResampled = 33,
    ReadTextureHandle = 34,
    GenerateMipmaps = 35,
    ReadTextureColorHandle = 36,
    UploadTextureAndMips = 37,
    UploadTexture = 38,
    SetDumpDraw = 39,
    StageUpload = 40,
    SetSnapshot = 41,
    WarmupTexture = 42,
    WarmupBuffer = 43,
    RetainVbib = 45,
    SetLayerPacing = 46,
    SetGamma = 47,
    UpdateColorRegion = 48,
    DestroyBuffer = 49,
}

/// How far a frame submission waits on the native pipeline.
///
/// Copy is needed when the native worker reads one retained frame's mode at
/// admission, submission, and retirement. Equality selects those wait stages.
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, FromRepr)]
pub enum EncoderSubmitMode {
    Queue = 0,
    WaitForSubmit = 1,
    WaitForGpu = 2,
}

/// Ordered native work requested without an accompanying frame.
#[repr(u32)]
#[derive(FromRepr)]
pub enum EncoderControl {
    DrainRetention = 0,
    IntakeVisibility = 1,
    Reset = 2,
}

impl TryFrom<u16> for EncoderOpcode {
    type Error = WireError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Self::from_repr(value).ok_or(WireError::InvalidValue)
    }
}

impl From<EncoderOpcode> for u16 {
    fn from(value: EncoderOpcode) -> Self {
        value as Self
    }
}

impl TryFrom<u32> for EncoderSubmitMode {
    type Error = WireError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::from_repr(value).ok_or(WireError::InvalidValue)
    }
}

impl From<EncoderSubmitMode> for u32 {
    fn from(value: EncoderSubmitMode) -> Self {
        value as Self
    }
}

impl TryFrom<u32> for EncoderControl {
    type Error = WireError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::from_repr(value).ok_or(WireError::InvalidValue)
    }
}

impl From<EncoderControl> for u32 {
    fn from(value: EncoderControl) -> Self {
        value as Self
    }
}

const _: () = {
    assert!(size_of::<EncoderOpcode>() == size_of::<u16>());
    assert!(size_of::<EncoderSubmitMode>() == size_of::<u32>());
    assert!(size_of::<EncoderControl>() == size_of::<u32>());
};
