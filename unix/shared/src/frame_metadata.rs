//! Fixed frame metadata borrowed from the publishing runtime until replay completion.

/// An immutable, initialized array retained by the frame owner.
#[repr(C, align(8))]
#[derive(Default)]
pub struct ArrayRef {
    pub address: u64,
    pub count: u32,
    pub reserved: u32,
}

/// Matched-build frame state with explicit padding and architecture-independent offsets.
#[repr(C, align(8))]
#[derive(Default)]
pub struct FrameMetadata {
    pub backbuffer_handle: u64,
    pub backbuffer_srgb_handle: u64,
    pub backbuffer_msaa_handle: u64,
    pub backbuffer_msaa_srgb_handle: u64,
    pub layer_handle: u64,
    pub view_handle: u64,
    pub depth_texture: u64,
    pub submit_seq: u64,
    pub perf: ArrayRef,
    pub flags: u32,
    pub backbuffer_width: u32,
    pub backbuffer_height: u32,
    pub backbuffer_format: u32,
    pub render_scale_percent: u32,
    pub backbuffer_sample_count: u32,
    pub backbuffer_contents: u32,
    pub reserved: u32,
}

const _: () = {
    assert!(size_of::<ArrayRef>() == 16);
    assert!(align_of::<ArrayRef>() == 8);
    assert!(core::mem::offset_of!(ArrayRef, address) == 0);
    assert!(core::mem::offset_of!(ArrayRef, count) == 8);
    assert!(core::mem::offset_of!(ArrayRef, reserved) == 12);
    assert!(size_of::<FrameMetadata>() == 112);
    assert!(align_of::<FrameMetadata>() == 8);
    assert!(core::mem::offset_of!(FrameMetadata, backbuffer_handle) == 0);
    assert!(core::mem::offset_of!(FrameMetadata, backbuffer_srgb_handle) == 8);
    assert!(core::mem::offset_of!(FrameMetadata, backbuffer_msaa_handle) == 16);
    assert!(core::mem::offset_of!(FrameMetadata, backbuffer_msaa_srgb_handle) == 24);
    assert!(core::mem::offset_of!(FrameMetadata, layer_handle) == 32);
    assert!(core::mem::offset_of!(FrameMetadata, view_handle) == 40);
    assert!(core::mem::offset_of!(FrameMetadata, depth_texture) == 48);
    assert!(core::mem::offset_of!(FrameMetadata, submit_seq) == 56);
    assert!(core::mem::offset_of!(FrameMetadata, perf) == 64);
    assert!(core::mem::offset_of!(FrameMetadata, flags) == 80);
    assert!(core::mem::offset_of!(FrameMetadata, backbuffer_width) == 84);
    assert!(core::mem::offset_of!(FrameMetadata, backbuffer_height) == 88);
    assert!(core::mem::offset_of!(FrameMetadata, backbuffer_format) == 92);
    assert!(core::mem::offset_of!(FrameMetadata, render_scale_percent) == 96);
    assert!(core::mem::offset_of!(FrameMetadata, backbuffer_sample_count) == 100);
    assert!(core::mem::offset_of!(FrameMetadata, backbuffer_contents) == 104);
    assert!(core::mem::offset_of!(FrameMetadata, reserved) == 108);
};
