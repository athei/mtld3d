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
    // + 2*u32 = 16 + 8 + 16 + 8 + 4 + 4 + 8 = 64 (explicit pad0
    // keeps the size a multiple of the align-8).
    assert_eq!(core::mem::align_of::<AttachMetalLayerParams>(), 8);
    assert_eq!(core::mem::size_of::<AttachMetalLayerParams>(), 64);
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
    // 3*u64 (handles) + 2*u64 (dst ptr/len) + 8*u32
    // = 24 + 16 + 32 = 72, already a multiple of the align-8 so the
    // struct carries no trailing pad.
    assert_eq!(core::mem::align_of::<BlitTextureToBufferParams>(), 8);
    assert_eq!(core::mem::size_of::<BlitTextureToBufferParams>(), 72);
}

#[test]
fn wait_for_gpu_retire_layout() {
    use super::WaitForGpuRetireParams;
    // 2 * u64 = 16
    assert_eq!(core::mem::align_of::<WaitForGpuRetireParams>(), 8);
    assert_eq!(core::mem::size_of::<WaitForGpuRetireParams>(), 16);
}

#[test]
fn frame_param_layouts_match_wow64() {
    assert_eq!(core::mem::align_of::<PassDescriptor>(), 8);
    assert_eq!(core::mem::align_of::<SubmitFrameParams>(), 8);
    assert_eq!(core::mem::align_of::<CreateTexturesBatchParams>(), 8);
    assert_eq!(core::mem::align_of::<TextureCreateDesc>(), 8);

    // PassDescriptor: 5 * u64 + 14 * u32 + 3 * ExtraColorDesc = 40 + 56 + 72 = 168.
    assert_eq!(core::mem::size_of::<PassDescriptor>(), 168);
    // ExtraColorDesc: 8 texture + 4 subresource + 4 load + 4 store + 4 reserved = 24.
    assert_eq!(core::mem::size_of::<ExtraColorDesc>(), 24);

    // SubmitFrameParams:
    //   8 queue_handle
    //   + 8 blit_commands_ptr + 4 blit_command_count + 4 blit_commands_need_encoder
    //   + 8 passes_ptr + 4 pass_count + 4 _pad1
    //   + 8 present_layer + 8 present_texture
    //   + 8 submit_seq + 8 coherent_seq_ptr + 8 upload_coherent_seq_ptr
    //   + 8 drawable_wait_ns + 8 present_view
    //   = 96
    assert_eq!(core::mem::size_of::<SubmitFrameParams>(), 96);

    // CreateTexturesBatchParams:
    //   8 device_handle + 4 count + 4 _pad0 + 8 descs_ptr + 8 handles_out_ptr = 32
    assert_eq!(core::mem::size_of::<CreateTexturesBatchParams>(), 32);

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
    assert_eq!(PassDescriptor::pack_flags(false, 0, 0), 0);
    assert_eq!(PassDescriptor::pack_flags(true, 0, 0), 1);
    assert_eq!(
        PassDescriptor::pack_flags(true, 5, 9),
        1 | (5 << 1) | (9 << 4)
    );
    assert_eq!(core::mem::size_of::<PassDescriptor>(), 168);
}
