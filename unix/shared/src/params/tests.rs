//! Size and alignment checks for a dozen of the PE/unix thunk parameter structs.
//!
//! Each covered struct pins `size_of` against a field-by-field tally in a comment, and `align_of`
//! against 8 (`ExtraColorDesc` gets only the size check); the other param structs are not covered.
//! No test reads `offset_of`, so swapping two same-width fields still passes here; the parent
//! module asserts field offsets at compile time only for `CreateDepthStencilStateParams`. One
//! test also pins `PassDescriptor::pack_flags`, where an ordinary pass encodes as zero.

use super::{
    BufferCreateDesc, CreateBuffersBatchParams, CreateTexturesBatchParams,
    DestroyResourcesBulkParams, ExtraColorDesc, PassDescriptor, SubmitFrameParams,
    TextureCreateDesc,
};

#[test]
fn buffer_param_layouts_match_wow64() {
    // All thunk params must be 8-byte aligned and contain only u32/u64
    // fields so 32-bit PE and 64-bit Unix agree on layout.
    assert_eq!(core::mem::align_of::<CreateBuffersBatchParams>(), 8);
    assert_eq!(core::mem::align_of::<BufferCreateDesc>(), 8);
    assert_eq!(core::mem::align_of::<DestroyResourcesBulkParams>(), 8);

    // Sizes: sum of fields with repr(C, align(8)) padding:
    //   CreateBuffersBatchParams   = 8 + 4 + 4 + 8 + 8     = 32
    //   BufferCreateDesc           = 8 + 8 + 8 + 4 + 4     = 32
    //   DestroyResourcesBulkParams = 4 + 4 + 8 + 4 + 4     = 24
    assert_eq!(core::mem::size_of::<CreateBuffersBatchParams>(), 32);
    assert_eq!(core::mem::size_of::<BufferCreateDesc>(), 32);
    assert_eq!(core::mem::size_of::<DestroyResourcesBulkParams>(), 24);
}

#[test]
fn attach_metal_layer_layout() {
    use super::AttachMetalLayerParams;
    // 2*u64 + 2*u32 + 2*u64 + 2*u32 + 1*u32 + 1*ColorSpacePolicy
    // + 2*u32 + 1*u64 = 16 + 8 + 16 + 8 + 4 + 4 + 8 + 8 = 72 (the
    // six 4-byte fields pair up, so the trailing u64 needs no pad).
    assert_eq!(core::mem::align_of::<AttachMetalLayerParams>(), 8);
    assert_eq!(core::mem::size_of::<AttachMetalLayerParams>(), 72);
}

#[test]
fn open_log_layout() {
    use super::OpenLogParams;
    // 2*u64 + 4*u32 = 16 + 16 = 32
    assert_eq!(core::mem::align_of::<OpenLogParams>(), 8);
    assert_eq!(core::mem::size_of::<OpenLogParams>(), 32);
}

#[test]
fn set_display_sync_enabled_layout() {
    use super::SetDisplaySyncEnabledParams;
    // u64 + u32 + u32 = 8 + 4 + 4 = 16
    assert_eq!(core::mem::align_of::<SetDisplaySyncEnabledParams>(), 8);
    assert_eq!(core::mem::size_of::<SetDisplaySyncEnabledParams>(), 16);
}

#[test]
fn blit_texture_to_buffer_layout() {
    use super::BlitTextureToBufferParams;
    // 3*u64 (handles) + 2*u64 (dst ptr/len) + 10*u32
    // = 24 + 16 + 40 = 80, a multiple of the align-8.
    assert_eq!(core::mem::align_of::<BlitTextureToBufferParams>(), 8);
    assert_eq!(core::mem::size_of::<BlitTextureToBufferParams>(), 80);
}

#[test]
fn create_texture_slice_view_layout() {
    use super::CreateTextureSliceViewParams;
    // 2*u64 (handles) + 2*u32 = 16 + 8 = 24
    assert_eq!(core::mem::align_of::<CreateTextureSliceViewParams>(), 8);
    assert_eq!(core::mem::size_of::<CreateTextureSliceViewParams>(), 24);
}

#[test]
fn wait_for_gpu_retire_layout() {
    use super::WaitForGpuRetireParams;
    // 3 * u64 = 24
    assert_eq!(core::mem::align_of::<WaitForGpuRetireParams>(), 8);
    assert_eq!(core::mem::size_of::<WaitForGpuRetireParams>(), 24);
}

#[test]
fn frame_param_layouts_match_wow64() {
    assert_eq!(core::mem::align_of::<PassDescriptor>(), 8);
    assert_eq!(core::mem::align_of::<SubmitFrameParams>(), 8);
    assert_eq!(core::mem::align_of::<CreateTexturesBatchParams>(), 8);
    assert_eq!(core::mem::align_of::<TextureCreateDesc>(), 8);

    // PassDescriptor: 7 * u64 + 16 * u32 + 3 * ExtraColorDesc = 56 + 64 + 96 = 216.
    assert_eq!(core::mem::size_of::<PassDescriptor>(), 216);
    // ExtraColorDesc: 8 texture + 8 resolve + 4 subresource + 4 load + 4 store
    // + 4 reserved = 32.
    assert_eq!(core::mem::size_of::<ExtraColorDesc>(), 32);

    // SubmitFrameParams:
    //   8 queue_handle
    //   + 8 blit_commands_ptr + 4 blit_command_count + 4 blit_commands_need_encoder
    //   + 8 passes_ptr + 4 pass_count + 4 _pad1
    //   + 8 present_layer + 8 present_texture
    //   + 8 submit_seq + 8 coherent_seq_ptr + 8 upload_coherent_seq_ptr
    //   + 8 failed_submit_seq_ptr
    //   + 8 drawable_wait_ns + 8 present_view
    //   = 104
    assert_eq!(core::mem::size_of::<SubmitFrameParams>(), 104);

    // CreateTexturesBatchParams:
    //   8 device_handle + 4 count + 4 _pad0 + 8 descs_ptr + 8 handles_out_ptr
    //   + 8 srgb_handles_out_ptr = 40
    assert_eq!(core::mem::size_of::<CreateTexturesBatchParams>(), 40);

    // TextureCreateDesc:
    //   8 tex_id
    //   + 4 width + 4 height + 4 depth + 4 levels (16)
    //   + 4 pixel_format + 4 storage_mode + 4 flags + 4 swizzle_r (16)
    //   + 4 swizzle_g + 4 swizzle_b + 4 swizzle_a + 4 usage_flags (16)
    //   = 56
    assert_eq!(core::mem::size_of::<TextureCreateDesc>(), 56);
}

#[test]
fn pass_descriptor_flags_preserve_ordinary_pass_bytes() {
    assert_eq!(PassDescriptor::pack_flags(false, 0, 0, 0), 0);
    assert_eq!(PassDescriptor::pack_flags(true, 0, 0, 0), 1);
    assert_eq!(
        PassDescriptor::pack_flags(true, 5, 9, 3),
        1 | (5 << 1) | (9 << 4) | (3 << 8)
    );
    assert_eq!(core::mem::size_of::<PassDescriptor>(), 216);
}
