use strum::FromRepr;

use super::mtl::{CullMode, IndexType, PrimitiveType, VisibilityResultMode};

/// Metal render command encoder commands.
///
/// Each variant maps 1:1 to a Metal `MTLRenderCommandEncoder` method.
/// The encoding thread walks an array of `Command` structs and replays
/// each as the corresponding Metal API call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, FromRepr)]
#[repr(u32)]
pub enum CommandType {
    /// `encoder.setRenderPipelineState(pipeline)`
    SetRenderPipelineState = 1,
    /// `encoder.setViewport(viewport)`
    SetViewport = 2,
    /// `encoder.setVertexBytes(ptr, length, index)`
    SetVertexBytes = 3,
    /// `encoder.drawPrimitives(type, vertexStart, vertexCount)`
    DrawPrimitives = 4,
    /// `encoder.setDepthStencilState(state)`
    SetDepthStencilState = 5,
    /// `encoder.setCullMode(mode)`
    SetCullMode = 6,
    /// `encoder.setFragmentTexture(texture, index)`
    SetFragmentTexture = 7,
    /// `encoder.setFragmentSamplerState(sampler, index)`
    SetFragmentSamplerState = 8,
    /// `encoder.setVertexBytes(ptr, length, index)` — for small buffers like constant tables.
    SetVertexBytesAt = 9,
    /// `encoder.setFragmentBytes(ptr, length, index)`
    SetFragmentBytesAt = 10,
    /// `encoder.setScissorRect(MTLScissorRect { x, y, width, height })`
    SetScissorRect = 11,
    /// `encoder.setVertexBuffer(buffer, offset, index)`
    SetVertexBuffer = 12,
    /// `encoder.drawIndexedPrimitives(...)` — indexed draw from a bound index buffer.
    ///
    /// Metal argument order: `type`, `indexCount`, `indexType`,
    /// `indexBuffer`, `offset`, `baseVertex`.
    DrawIndexedPrimitives = 13,
    /// `encoder.setVisibilityResultMode(mode, offset)`
    ///
    /// Arms / disarms per-fragment counting for occlusion queries.
    /// `offset` is a byte offset into the pass's
    /// `visibilityResultBuffer`.
    SetVisibilityResultMode = 14,
    /// `encoder.setBlendColorRed:green:blue:alpha:`
    ///
    /// The constant RGBA referenced by `MTLBlendFactor::BlendColor` /
    /// `OneMinusBlendColor`. Emitted whenever `D3DRS_BLENDFACTOR`
    /// differs from its default (opaque white) so games that drive a
    /// constant-color blend (fades, decals) get the right tint.
    SetBlendColor = 15,
    /// `encoder.setDepthBias:slopeScale:clamp:`
    ///
    /// The per-encoder rasterizer offset that pushes fragments toward /
    /// away from the camera. Backs `D3DRS_DEPTHBIAS` and
    /// `D3DRS_SLOPESCALEDEPTHBIAS` so games that draw ground-projected
    /// decals (selection circles, shadows, blob projectors) don't z-fight
    /// with the surface they sit on. Clamp is hardcoded to 0.0 (D3D9 has
    /// no clamp control).
    SetDepthBias = 16,
    /// `encoder.drawIndexedPrimitives(...)` with an inline (user-pointer) index stream.
    ///
    /// Backs `DrawIndexedPrimitiveUP`. The index bytes live in the per-frame
    /// scratch arena; the unix side copies them into a transient `MTLBuffer`
    /// (`newBufferWithBytes`) for the draw, since Metal has no inline-index
    /// form.
    DrawIndexedPrimitivesUp = 17,
    // 18 was SetDepthClipMode; the D3D9 depth-clamp rule is realized in the
    // FF vertex shader now (`pos_fixup.z`), because encoder-level clamp is
    // not honoured by every Metal device.
    /// `encoder.setStencilReferenceValue:`
    ///
    /// The reference value the stencil test compares against, and the source
    /// for `D3DSTENCILOP_REPLACE`. Metal carries it on the encoder rather
    /// than the depth/stencil state object, so `D3DRS_STENCILREF` changes
    /// cost a command instead of a new `MTLDepthStencilState`.
    SetStencilReference = 19,
    /// Bind the shared 1×1 opaque-black texture + a default sampler at an index.
    ///
    /// A pixel shader declares a sampler (`dcl_2d`/`dcl_cube`/`dcl_volume`),
    /// so the fragment function carries the matching `[[texture(n)]]` /
    /// `[[sampler(n)]]` argument, but the game bound no texture to that stage.
    /// D3D9 requires such a sample to read opaque black `(0,0,0,1)`, and Metal
    /// requires every declared argument to be bound. `param_a` is the index;
    /// `param_b` is a [`NullTextureKind`], selecting the 2D / cube / 3D black
    /// texture whose type matches the declaration.
    SetFragmentNullTexture = 20,
    /// `encoder.setVertexTexture(texture, index)`
    ///
    /// Vertex texture fetch: a `vs_3_0` shader declaring a sampler reads it
    /// with `texldl` (explicit LOD, the only form the model allows), so the
    /// vertex function carries `[[texture(n)]]` / `[[sampler(n)]]` arguments
    /// bound by this pair of commands.
    SetVertexTexture = 21,
    /// `encoder.setVertexSamplerState(sampler, index)`
    SetVertexSamplerState = 22,
    /// Opaque-black fallback for a declared-but-unbound VERTEX sampler.
    ///
    /// Same contract as [`CommandType::SetFragmentNullTexture`], on the
    /// vertex stage: `param_a` is the slot, `param_b` a [`NullTextureKind`].
    SetVertexNullTexture = 23,
    /// `encoder.setFragmentBuffer(buffer, offset, index)`
    ///
    /// Binds a whole `MTLBuffer` (not inline bytes) to a fragment buffer
    /// slot. The texture-upload quad reads its packed staging source through
    /// this: the staging slab is far past the inline-bytes limit, and reading
    /// it as a shader argument is what lets the upload ignore the linear
    /// texture row alignment a blit copy would have to satisfy.
    SetFragmentBuffer = 24,
}

/// Dimensionality of the opaque-black texture a null-texture bind selects.
///
/// The value matches the shader's declared sampler type
/// ([`CommandType::SetFragmentNullTexture`]) so the bound texture satisfies the
/// `[[texture(n)]]` argument's type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, FromRepr)]
#[repr(u32)]
pub enum NullTextureKind {
    /// `texture2d<float>` — a `dcl_2d` sampler.
    Texture2D = 0,
    /// `texturecube<float>` — a `dcl_cube` sampler.
    TextureCube = 1,
    /// `texture3d<float>` — a `dcl_volume` sampler.
    Texture3D = 2,
}

/// Fixed-size command struct written by the API thread and read by the encoding thread.
///
/// 32 bytes, aligned to 8. Field semantics depend on `cmd`
/// (see [`CommandType`]).
#[derive(Clone, Copy)]
#[repr(C, align(8))]
pub struct Command {
    pub cmd: u32,
    pub param_a: u32,
    pub param_b: u64,
    pub param_c: u64,
    pub param_d: u64,
}

impl Command {
    /// `encoder.setRenderPipelineState(pipeline)`
    #[must_use]
    pub const fn set_render_pipeline_state(pipeline_handle: u64) -> Self {
        Self {
            cmd: CommandType::SetRenderPipelineState as u32,
            param_a: 0,
            param_b: pipeline_handle,
            param_c: 0,
            param_d: 0,
        }
    }

    /// `encoder.setViewport(MTLViewport { x, y, width, height, min_z, max_z })`.
    ///
    /// D3DVIEWPORT9 has non-zero `x, y` when the game renders UI through a
    /// sub-rect of the render target; dropping them shifts every XYZRHW
    /// draw by the origin and breaks UI layout. `min_z`/`max_z` are D3D9's
    /// per-viewport depth range — games partition the depth buffer (sky /
    /// world / weapon) by shifting this window, and depth-bias is scaled
    /// against the active range, so ignoring them silently mis-Z-tests
    /// decals and shadow-blending draws. Packed into the unused high
    /// halves of `param_b` / `param_c` via `f32::to_bits`; `param_a` and
    /// `param_d` keep width / y unchanged.
    #[must_use]
    pub const fn set_viewport(
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        min_z: f32,
        max_z: f32,
    ) -> Self {
        Self {
            cmd: CommandType::SetViewport as u32,
            param_a: width,
            param_b: (min_z.to_bits() as u64) << 32 | height as u64,
            param_c: (max_z.to_bits() as u64) << 32 | x as u64,
            param_d: y as u64,
        }
    }

    /// `encoder.setVertexBytes(data_ptr, data_size, buffer_index)`
    #[must_use]
    pub const fn set_vertex_bytes(data_ptr: u64, data_size: u32, buffer_index: u32) -> Self {
        Self {
            cmd: CommandType::SetVertexBytes as u32,
            param_a: buffer_index,
            param_b: data_ptr,
            param_c: data_size as u64,
            param_d: 0,
        }
    }

    /// `encoder.drawPrimitives(primitive_type, vertex_start, vertex_count)`
    #[must_use]
    pub const fn draw_primitives(
        primitive_type: PrimitiveType,
        vertex_start: u32,
        vertex_count: u32,
    ) -> Self {
        Self {
            cmd: CommandType::DrawPrimitives as u32,
            param_a: primitive_type as u32,
            param_b: vertex_start as u64,
            param_c: vertex_count as u64,
            param_d: 0,
        }
    }

    /// `encoder.setDepthStencilState(state)`
    #[must_use]
    pub const fn set_depth_stencil_state(state_handle: u64) -> Self {
        Self {
            cmd: CommandType::SetDepthStencilState as u32,
            param_a: 0,
            param_b: state_handle,
            param_c: 0,
            param_d: 0,
        }
    }

    /// `encoder.setCullMode(mode)`
    #[must_use]
    pub const fn set_cull_mode(mode: CullMode) -> Self {
        Self {
            cmd: CommandType::SetCullMode as u32,
            param_a: mode as u32,
            param_b: 0,
            param_c: 0,
            param_d: 0,
        }
    }

    /// `encoder.setFragmentTexture(texture, index)`
    #[must_use]
    pub const fn set_fragment_texture(texture_handle: u64, index: u32) -> Self {
        Self {
            cmd: CommandType::SetFragmentTexture as u32,
            param_a: index,
            param_b: texture_handle,
            param_c: 0,
            param_d: 0,
        }
    }

    /// `encoder.setVertexTexture(texture, index)`
    #[must_use]
    pub const fn set_vertex_texture(texture_handle: u64, index: u32) -> Self {
        Self {
            cmd: CommandType::SetVertexTexture as u32,
            param_a: index,
            param_b: texture_handle,
            param_c: 0,
            param_d: 0,
        }
    }

    /// `encoder.setVertexSamplerState(sampler, index)`
    #[must_use]
    pub const fn set_vertex_sampler_state(sampler_handle: u64, index: u32) -> Self {
        Self {
            cmd: CommandType::SetVertexSamplerState as u32,
            param_a: index,
            param_b: sampler_handle,
            param_c: 0,
            param_d: 0,
        }
    }

    /// Opaque-black fallback texture + default sampler for a vertex slot.
    #[must_use]
    pub const fn set_vertex_null_texture(kind: NullTextureKind, index: u32) -> Self {
        Self {
            cmd: CommandType::SetVertexNullTexture as u32,
            param_a: index,
            param_b: kind as u64,
            param_c: 0,
            param_d: 0,
        }
    }

    /// `encoder.setFragmentSamplerState(sampler, index)`
    #[must_use]
    pub const fn set_fragment_sampler_state(sampler_handle: u64, index: u32) -> Self {
        Self {
            cmd: CommandType::SetFragmentSamplerState as u32,
            param_a: index,
            param_b: sampler_handle,
            param_c: 0,
            param_d: 0,
        }
    }

    /// Bind the shared opaque-black texture of `kind` and a default sampler at `index`.
    #[must_use]
    pub const fn set_fragment_null_texture(kind: NullTextureKind, index: u32) -> Self {
        Self {
            cmd: CommandType::SetFragmentNullTexture as u32,
            param_a: index,
            param_b: kind as u64,
            param_c: 0,
            param_d: 0,
        }
    }

    /// `encoder.setVertexBytes(data_ptr, data_size, buffer_index)`
    #[must_use]
    pub const fn set_vertex_bytes_at(data_ptr: u64, data_size: u32, buffer_index: u32) -> Self {
        Self {
            cmd: CommandType::SetVertexBytesAt as u32,
            param_a: buffer_index,
            param_b: data_ptr,
            param_c: data_size as u64,
            param_d: 0,
        }
    }

    /// `encoder.setFragmentBytes(data_ptr, data_size, buffer_index)`
    #[must_use]
    pub const fn set_fragment_bytes_at(data_ptr: u64, data_size: u32, buffer_index: u32) -> Self {
        Self {
            cmd: CommandType::SetFragmentBytesAt as u32,
            param_a: buffer_index,
            param_b: data_ptr,
            param_c: data_size as u64,
            param_d: 0,
        }
    }

    /// `encoder.setScissorRect(MTLScissorRect { x, y, width, height })`
    #[must_use]
    pub const fn set_scissor_rect(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            cmd: CommandType::SetScissorRect as u32,
            param_a: x,
            // Pack y into low 32 bits of param_b; width/height into param_c.
            param_b: y as u64,
            param_c: ((width as u64) << 32) | (height as u64),
            param_d: 0,
        }
    }

    /// `encoder.setVertexBuffer(buffer, offset, atIndex: index)`
    #[must_use]
    pub const fn set_vertex_buffer(buffer_handle: u64, offset: u32, buffer_index: u32) -> Self {
        Self {
            cmd: CommandType::SetVertexBuffer as u32,
            param_a: buffer_index,
            param_b: buffer_handle,
            param_c: offset as u64,
            param_d: 0,
        }
    }

    /// `encoder.setFragmentBuffer(buffer_handle, offset, buffer_index)`
    #[must_use]
    pub const fn set_fragment_buffer(buffer_handle: u64, offset: u32, buffer_index: u32) -> Self {
        Self {
            cmd: CommandType::SetFragmentBuffer as u32,
            param_a: buffer_index,
            param_b: buffer_handle,
            param_c: offset as u64,
            param_d: 0,
        }
    }

    /// `encoder.drawIndexedPrimitives(...)` — indexed draw from a bound index buffer.
    ///
    /// Metal argument order: `type`, `indexCount`, `indexType`,
    /// `indexBuffer`, `indexBufferOffset`, `instanceCount`, `baseVertex`,
    /// `baseInstance`.
    ///
    /// `param_d` packs `index_type` into bits 0..8, `index_count` into bits
    /// 8..40 and `instance_count` into bits 40..64 (see
    /// [`Self::pack_indexed_draw_counts`]). `param_c` packs `offset` into the
    /// high 32 bits and `base_vertex` (signed, via bitcast) into the low 32 so
    /// both fit in one u64. The unix side decodes with
    /// `offset = (param_c >> 32) as usize` and
    /// `base_vertex = (param_c as u32) as i32 as isize`.
    #[must_use]
    pub const fn draw_indexed_primitives(
        primitive_type: PrimitiveType,
        index_count: u32,
        index_type: IndexType,
        index_buffer: u64,
        index_buffer_offset: u32,
        base_vertex: i32,
        instance_count: u32,
    ) -> Self {
        // Bitcast i32 → u32 so the sign pattern round-trips; `as i32 as isize`
        // on the unix side re-sign-extends.
        let base_vertex_bits = base_vertex.cast_unsigned() as u64;
        let offset_bits = (index_buffer_offset as u64) << 32;
        Self {
            cmd: CommandType::DrawIndexedPrimitives as u32,
            param_a: primitive_type as u32,
            param_b: index_buffer,
            param_c: offset_bits | base_vertex_bits,
            param_d: Self::pack_indexed_draw_counts(index_count, index_type, instance_count),
        }
    }

    /// `DrawIndexedPrimitiveUP`: draw with an inline (user-pointer) index stream.
    ///
    /// `index_ptr` points into the per-frame scratch arena and
    /// `index_bytes` is its length; the unix side copies it into a transient
    /// `MTLBuffer` (`newBufferWithBytes`) for the draw, since Metal has no
    /// inline-index form. `param_d` carries the counts packed by
    /// [`Self::pack_indexed_draw_counts`]. Base vertex is always 0 (UP indices
    /// are absolute).
    #[must_use]
    pub const fn draw_indexed_primitives_up(
        primitive_type: PrimitiveType,
        index_count: u32,
        index_type: IndexType,
        index_ptr: u64,
        index_bytes: u32,
        instance_count: u32,
    ) -> Self {
        Self {
            cmd: CommandType::DrawIndexedPrimitivesUp as u32,
            param_a: primitive_type as u32,
            param_b: index_ptr,
            param_c: index_bytes as u64,
            param_d: Self::pack_indexed_draw_counts(index_count, index_type, instance_count),
        }
    }

    /// Pack the counts of an indexed draw into one `param_d`.
    ///
    /// Bits 0..8 hold the `IndexType` discriminant, bits 8..40 the full
    /// 32-bit index count and bits 40..64 the instance count, which D3D9 caps
    /// at 23 bits (`D3DSTREAMSOURCE_INDEXEDDATA | n` masks `n` with
    /// `0x7FFFFF`), so the three never overlap. Decode with
    /// [`Self::unpack_indexed_draw_counts`].
    #[must_use]
    pub const fn pack_indexed_draw_counts(
        index_count: u32,
        index_type: IndexType,
        instance_count: u32,
    ) -> u64 {
        ((instance_count as u64) << 40) | ((index_count as u64) << 8) | ((index_type as u64) & 0xFF)
    }

    /// Decode a `param_d` packed by [`Self::pack_indexed_draw_counts`].
    ///
    /// Returns `(index_count, raw index type, instance_count)`; the raw index
    /// type goes through `IndexType::from_repr` at the decode site.
    #[must_use]
    pub const fn unpack_indexed_draw_counts(param_d: u64) -> (u32, u32, u32) {
        let index_count = ((param_d >> 8) & 0xFFFF_FFFF) as u32;
        let index_type_raw = (param_d & 0xFF) as u32;
        let instance_count = (param_d >> 40) as u32;
        (index_count, index_type_raw, instance_count)
    }

    /// `encoder.setVisibilityResultMode(mode, offset)`.
    ///
    /// `offset` is the byte offset into the pass's
    /// `visibilityResultBuffer` (slot index × 8).
    #[must_use]
    pub const fn set_visibility_result_mode(mode: VisibilityResultMode, offset_bytes: u32) -> Self {
        Self {
            cmd: CommandType::SetVisibilityResultMode as u32,
            param_a: mode as u32,
            param_b: offset_bytes as u64,
            param_c: 0,
            param_d: 0,
        }
    }

    #[must_use]
    pub const fn set_stencil_reference(value: u32) -> Self {
        Self {
            cmd: CommandType::SetStencilReference as u32,
            param_a: value,
            param_b: 0,
            param_c: 0,
            param_d: 0,
        }
    }

    /// `encoder.setBlendColorRed:green:blue:alpha:`.
    ///
    /// Each f32 lane is stored as its bit pattern in the low 32 bits of the
    /// matching param slot (`param_a` is u32, so it carries `r` directly).
    #[must_use]
    pub const fn set_blend_color(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            cmd: CommandType::SetBlendColor as u32,
            param_a: r.to_bits(),
            param_b: g.to_bits() as u64,
            param_c: b.to_bits() as u64,
            param_d: a.to_bits() as u64,
        }
    }

    /// `encoder.setDepthBias(bias, slopeScale: slope, clamp: 0.0)`.
    ///
    /// `depth_bias` is already scaled to the active depth format's ULP
    /// (every depth surface in mtld3d resolves to `Depth32Float` or
    /// `Depth32Float_Stencil8`, so callers multiply the raw D3D9 float
    /// by 2^23 before constructing the command — see
    /// `mtld3d_core::convert::d3d_depth_bias_to_metal`). `slope_scale` is
    /// passed through unchanged. Clamp is hardcoded to 0.0 unix-side.
    #[must_use]
    pub const fn set_depth_bias(depth_bias: f32, slope_scale: f32) -> Self {
        Self {
            cmd: CommandType::SetDepthBias as u32,
            param_a: depth_bias.to_bits(),
            param_b: slope_scale.to_bits() as u64,
            param_c: 0,
            param_d: 0,
        }
    }
}

/// Metal blit command encoder commands.
///
/// Replayed inside a leading `MTLBlitCommandEncoder` before any render
/// pass in the frame.
///
/// Kept as a separate enum (and struct) from `Command` / `CommandType`
/// so the unix side can dispatch on the correct encoder without
/// runtime probing.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, FromRepr)]
pub enum BlitCommandType {
    /// `blit.copyFromBuffer(...).toTexture(...)`
    ///
    /// Sub-rect upload from a Shared staging `MTLBuffer` into an
    /// `MTLTexture`.
    CopyBufferToTexture = 1,
    /// `blit.copyFromTexture(src).toTexture(dst)`
    ///
    /// Full-mip tile-to-tile preserve blit for the default-flag contended
    /// Lock path.
    CopyTextureToTexture = 2,
    /// `blit.copyFromBuffer(src).toBuffer(dst)`
    ///
    /// Async preserve of unchanged head/tail ranges for the WRITEONLY
    /// contended VB/IB rename path. `src_handle` / `dst_handle` =
    /// `MTLBuffers`, `src_offset` / `dst_offset` = byte offsets,
    /// `byte_size` = number of bytes to copy. Region / mip fields unused.
    CopyBufferToBuffer = 3,
    /// `[buffer didModifyRange:NSMakeRange(offset, length)]`
    ///
    /// Signals to Metal that the CPU has just written `length` bytes
    /// at `offset` into a `MTLStorageModeManaged` buffer, so the
    /// driver knows to copy those bytes from system memory to VRAM
    /// before the next GPU read. No-op on `MTLStorageModeShared`
    /// (UMA Macs). Not actually a render-encoder method — the unix
    /// dispatcher calls it directly on the buffer outside the blit
    /// encoder, but it rides the blit-command list so it ships in the
    /// same `SubmitFrame` thunk and gets ordered against other
    /// frame-leading ops. `src_handle` = `MTLBuffer`, `src_offset` =
    /// offset, `byte_size` = length. All other fields unused.
    NotifyBufferDidModifyRange = 4,
    /// `blit.generateMipmapsForTexture(tex)`
    ///
    /// Regenerate mip levels 1..N from level 0 on the shared
    /// frame-leading blit encoder. `dst_handle` = `MTLTexture`. All
    /// other fields unused.
    GenerateMipmaps = 5,
}

/// Fixed-size blit command struct.
///
/// 80 bytes, aligned to 8. Field semantics depend on `cmd`:
///
/// - `CopyBufferToTexture`: `src_handle` = buffer, `dst_handle` =
///   texture, `mip_level` / `origin_x,y` / `region_w,h` /
///   `bytes_per_row` / `src_offset` describe the copy. `depth` is the
///   slice count (1 for a 2D texture, >1 for a volume/3D texture) and
///   `bytes_per_image` is the byte stride between slices (for a 2D copy
///   it equals `bytes_per_row * region_h`, matching the implicit
///   single-slice size). `dst_offset` / `byte_size` unused.
/// - `CopyTextureToTexture`: `src_handle` / `dst_handle` = textures,
///   `mip_level` selects the mip. `origin_x` / `origin_y` are the
///   *source* origin; `region_w` / `region_h` are the region size; the
///   *destination* origin is packed into `dst_offset` as `(dst_y as
///   u64) << 32 | dst_x as u64`. For a full-mip preserve blit emit
///   src origin = (0, 0), region = (`mip_w`, `mip_h`), `dst_offset` = 0
///   (dst origin (0, 0)). `bytes_per_row` / `byte_size` unused.
/// - `CopyBufferToBuffer`: `src_handle` / `dst_handle` = buffers,
///   `src_offset` / `dst_offset` = byte offsets, `byte_size` = copy
///   size in bytes. `mip_level` / `origin_*` / `region_*` /
///   `bytes_per_row` unused.
/// - `GenerateMipmaps`: `dst_handle` = texture. All other fields
///   unused — the encoder reads dimensions / mip count / pixel
///   format off the `MTLTexture` itself.
#[derive(Clone, Copy)]
#[repr(C, align(8))]
pub struct BlitCommand {
    pub cmd: u32,
    pub mip_level: u32,
    pub src_handle: u64,
    pub dst_handle: u64,
    pub src_offset: u64,
    pub bytes_per_row: u64,
    pub origin_x: u32,
    pub origin_y: u32,
    pub region_w: u32,
    pub region_h: u32,
    pub dst_offset: u64,
    pub byte_size: u64,
    /// Slice count for `CopyBufferToTexture` (1 = 2D, >1 = volume/3D).
    ///
    /// Unused (0) for the other command types.
    pub depth: u32,
    /// Byte stride between slices for `CopyBufferToTexture`.
    ///
    /// Unused (0) for the other command types.
    pub bytes_per_image: u32,
    /// Destination mip level for `CopyTextureToTexture`.
    ///
    /// `mip_level` is the source level; a texture-to-texture copy may land on a
    /// different level of the destination. Unused (0) for the other command
    /// types.
    pub dst_mip_level: u32,
    /// Explicit tail padding to the 8-byte stride.
    pub pad0: u32,
}

/// Inputs for `BlitCommand::copy_buffer_to_texture`.
///
/// Grouping the sub-rect + layout params keeps the constructor from
/// tripping the `too_many_arguments` lint and makes call sites
/// self-documenting.
#[derive(Clone, Copy)]
pub struct CopyBufferToTextureInfo {
    pub buffer_handle: u64,
    pub buffer_offset: u64,
    pub bytes_per_row: u32,
    pub texture_handle: u64,
    /// Destination texture slice.
    ///
    /// Zero for 2D and volume textures; cube faces use their
    /// `D3DCUBEMAP_FACES` index.
    pub destination_slice: u32,
    pub mip_level: u32,
    pub origin_x: u32,
    pub origin_y: u32,
    pub region_w: u32,
    pub region_h: u32,
    /// Slice count: 1 for a 2D texture, >1 for a volume (3D) texture.
    pub depth: u32,
    /// Byte stride between slices.
    ///
    /// For a 2D copy (`depth == 1`) callers pass `bytes_per_row * region_h` —
    /// the implicit single-slice size. For a volume it is the box's slice
    /// pitch.
    pub bytes_per_image: u32,
}

impl BlitCommand {
    /// `blit.copyFromBuffer(...).toTexture(...)` with `size.depth = info.depth`.
    ///
    /// Full form: `blit.copyFromBuffer(buffer, offset, bytesPerRow,
    /// bytesPerImage, size, toTexture: texture, destSlice: 0, level: mip,
    /// origin: (x, y, 0))`.
    #[must_use]
    pub const fn copy_buffer_to_texture(info: &CopyBufferToTextureInfo) -> Self {
        Self {
            cmd: BlitCommandType::CopyBufferToTexture as u32,
            mip_level: info.mip_level,
            src_handle: info.buffer_handle,
            dst_handle: info.texture_handle,
            src_offset: info.buffer_offset,
            bytes_per_row: info.bytes_per_row as u64,
            origin_x: info.origin_x,
            origin_y: info.origin_y,
            region_w: info.region_w,
            region_h: info.region_h,
            dst_offset: info.destination_slice as u64,
            byte_size: 0,
            depth: info.depth,
            bytes_per_image: info.bytes_per_image,
            dst_mip_level: 0,
            pad0: 0,
        }
    }

    /// Full-mip `blit.copyFromTexture(...).toTexture(...)`.
    ///
    /// `blit.copyFromTexture(src, sourceSlice: 0, level: mip, origin:
    /// (0,0,0), size: (w,h,1), toTexture: dst, destSlice: 0, level: mip,
    /// origin: (0,0,0))`. Used only for default-flag contended Lock
    /// preserve.
    #[must_use]
    pub const fn copy_texture_to_texture_full_mip(
        src_texture: u64,
        dst_texture: u64,
        mip_level: u32,
        mip_w: u32,
        mip_h: u32,
    ) -> Self {
        Self {
            cmd: BlitCommandType::CopyTextureToTexture as u32,
            mip_level,
            src_handle: src_texture,
            dst_handle: dst_texture,
            src_offset: 0,
            bytes_per_row: 0,
            origin_x: 0,
            origin_y: 0,
            region_w: mip_w,
            region_h: mip_h,
            dst_offset: 0,
            byte_size: 0,
            depth: 0,
            bytes_per_image: 0,
            dst_mip_level: mip_level,
            pad0: 0,
        }
    }

    /// Sub-rect `blit.copyFromTexture(...).toTexture(...)`.
    ///
    /// `blit.copyFromTexture(src, sourceSlice: 0, level: mip, origin:
    /// (src_x, src_y, 0), size: (w, h, 1), toTexture: dst, destSlice: 0,
    /// level: dst_mip, origin: (dst_x, dst_y, 0))`. Used by
    /// `IDirect3DDevice9::StretchRect` for 1:1 same-format copies
    /// between two textures (scaling is not supported).
    #[must_use]
    pub const fn copy_texture_to_texture_sub_rect(info: &CopyTextureSubRectInfo) -> Self {
        Self {
            cmd: BlitCommandType::CopyTextureToTexture as u32,
            mip_level: info.mip_level,
            src_handle: info.src_texture,
            dst_handle: info.dst_texture,
            src_offset: 0,
            bytes_per_row: 0,
            origin_x: info.src_origin_x,
            origin_y: info.src_origin_y,
            region_w: info.region_w,
            region_h: info.region_h,
            dst_offset: ((info.dst_origin_y as u64) << 32) | (info.dst_origin_x as u64),
            byte_size: 0,
            depth: 0,
            bytes_per_image: 0,
            dst_mip_level: info.dst_mip_level,
            pad0: 0,
        }
    }

    /// `blit.copyFromBuffer(src, sourceOffset, toBuffer: dst, destinationOffset, size)`.
    ///
    /// Used by the WRITEONLY contended VB/IB rename path to preserve
    /// head/tail ranges async on the encoder thread instead of
    /// synchronously memcpying on the API thread.
    #[must_use]
    pub const fn copy_buffer_to_buffer(info: &CopyBufferToBufferInfo) -> Self {
        Self {
            cmd: BlitCommandType::CopyBufferToBuffer as u32,
            mip_level: 0,
            src_handle: info.src_buffer,
            dst_handle: info.dst_buffer,
            src_offset: info.src_offset,
            bytes_per_row: 0,
            origin_x: 0,
            origin_y: 0,
            region_w: 0,
            region_h: 0,
            dst_offset: info.dst_offset,
            byte_size: info.byte_size,
            depth: 0,
            bytes_per_image: 0,
            dst_mip_level: 0,
            pad0: 0,
        }
    }

    /// `[buffer didModifyRange:NSMakeRange(offset, length)]`.
    ///
    /// Encoded into the blit-command stream so the unix dispatcher can call
    /// it outside any encoder right before the blit + render passes that
    /// will read the buffer. No-op on UMA Macs (`Shared` storage).
    #[must_use]
    pub const fn notify_buffer_did_modify_range(buffer: u64, offset: u64, length: u64) -> Self {
        Self {
            cmd: BlitCommandType::NotifyBufferDidModifyRange as u32,
            mip_level: 0,
            src_handle: buffer,
            dst_handle: 0,
            src_offset: offset,
            bytes_per_row: 0,
            origin_x: 0,
            origin_y: 0,
            region_w: 0,
            region_h: 0,
            dst_offset: 0,
            byte_size: length,
            depth: 0,
            bytes_per_image: 0,
            dst_mip_level: 0,
            pad0: 0,
        }
    }

    /// `blit.generateMipmapsForTexture(tex)`.
    ///
    /// Encoded into the shared frame-leading blit encoder so an
    /// autogen-opt-in texture's mip-1..N regeneration runs inside the
    /// frame's own command buffer right after its mip-0
    /// `CopyBufferToTexture`, instead of in a dedicated per-texture
    /// command buffer on the queue.
    #[must_use]
    pub const fn generate_mipmaps(texture: u64) -> Self {
        Self {
            cmd: BlitCommandType::GenerateMipmaps as u32,
            mip_level: 0,
            src_handle: 0,
            dst_handle: texture,
            src_offset: 0,
            bytes_per_row: 0,
            origin_x: 0,
            origin_y: 0,
            region_w: 0,
            region_h: 0,
            dst_offset: 0,
            byte_size: 0,
            depth: 0,
            bytes_per_image: 0,
            dst_mip_level: 0,
            pad0: 0,
        }
    }
}

/// Inputs for `BlitCommand::copy_buffer_to_buffer`.
pub struct CopyBufferToBufferInfo {
    pub src_buffer: u64,
    pub dst_buffer: u64,
    pub src_offset: u64,
    pub dst_offset: u64,
    pub byte_size: u64,
}

/// Inputs for `BlitCommand::copy_texture_to_texture_sub_rect`.
///
/// `region_w` / `region_h` describe both the source size and the
/// destination size — sub-rect `StretchRect` is 1:1 only.
pub struct CopyTextureSubRectInfo {
    pub src_texture: u64,
    pub dst_texture: u64,
    /// Source mip level.
    pub mip_level: u32,
    /// Destination mip level.
    pub dst_mip_level: u32,
    pub src_origin_x: u32,
    pub src_origin_y: u32,
    pub dst_origin_x: u32,
    pub dst_origin_y: u32,
    pub region_w: u32,
    pub region_h: u32,
}

#[cfg(test)]
mod tests;
