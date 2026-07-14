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
    /// `encoder.setDepthClipMode:` — Clip (Metal's default) or Clamp.
    ///
    /// D3D9 z-clips primitives only while the depth test is live
    /// (`D3DRS_ZENABLE` on AND a depth attachment bound); with the test
    /// inactive, out-of-range-z fragments are depth-clamped and drawn —
    /// an XYZRHW quad with z in [-0.5, 1.5] and no depth surface /
    /// ZENABLE off rasterizes in full.
    SetDepthClipMode = 18,
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

    /// `encoder.drawIndexedPrimitives(...)` — indexed draw from a bound index buffer.
    ///
    /// Metal argument order: `type`, `indexCount`, `indexType`,
    /// `indexBuffer`, `indexBufferOffset`, `instanceCount`, `baseVertex`,
    /// `baseInstance`.
    ///
    /// `index_count` is packed into the high 56 bits of `param_d`; `index_type`
    /// takes the low 8 bits. `param_c` packs `offset` into the high 32 bits
    /// and `base_vertex` (signed, via bitcast) into the low 32 so both fit in
    /// one u64. The unix side decodes with `offset = (param_c >> 32) as usize`
    /// and `base_vertex = (param_c as u32) as i32 as isize`.
    #[must_use]
    pub const fn draw_indexed_primitives(
        primitive_type: PrimitiveType,
        index_count: u32,
        index_type: IndexType,
        index_buffer: u64,
        index_buffer_offset: u32,
        base_vertex: i32,
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
            param_d: ((index_count as u64) << 8) | ((index_type as u64) & 0xFF),
        }
    }

    /// `DrawIndexedPrimitiveUP`: draw with an inline (user-pointer) index stream.
    ///
    /// `index_ptr` points into the per-frame scratch arena and
    /// `index_bytes` is its length; the unix side copies it into a transient
    /// `MTLBuffer` (`newBufferWithBytes`) for the draw, since Metal has no
    /// inline-index form. `index_count` packs into the high 56 bits of
    /// `param_d`, `index_type` into the low 8. Base vertex is always 0 (UP
    /// indices are absolute), single instance.
    #[must_use]
    pub const fn draw_indexed_primitives_up(
        primitive_type: PrimitiveType,
        index_count: u32,
        index_type: IndexType,
        index_ptr: u64,
        index_bytes: u32,
    ) -> Self {
        Self {
            cmd: CommandType::DrawIndexedPrimitivesUp as u32,
            param_a: primitive_type as u32,
            param_b: index_ptr,
            param_c: index_bytes as u64,
            param_d: ((index_count as u64) << 8) | ((index_type as u64) & 0xFF),
        }
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

    /// `encoder.setDepthClipMode(clip ? Clip : Clamp)` — see [`CommandType::SetDepthClipMode`].
    #[must_use]
    pub const fn set_depth_clip_mode(clip: bool) -> Self {
        Self {
            cmd: CommandType::SetDepthClipMode as u32,
            param_a: clip as u32,
            param_b: 0,
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
            dst_offset: 0,
            byte_size: 0,
            depth: info.depth,
            bytes_per_image: info.bytes_per_image,
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
        }
    }

    /// Sub-rect `blit.copyFromTexture(...).toTexture(...)`.
    ///
    /// `blit.copyFromTexture(src, sourceSlice: 0, level: mip, origin:
    /// (src_x, src_y, 0), size: (w, h, 1), toTexture: dst, destSlice: 0,
    /// level: mip, origin: (dst_x, dst_y, 0))`. Used by
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
    pub mip_level: u32,
    pub src_origin_x: u32,
    pub src_origin_y: u32,
    pub dst_origin_x: u32,
    pub dst_origin_y: u32,
    pub region_w: u32,
    pub region_h: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode the packing that the unix-side `CommandType::DrawIndexedPrimitives` handler uses.
    ///
    /// Kept in one place so the encoder and decoder stay in sync.
    fn decode_draw_indexed(cmd: &Command) -> (u32, u32, u32, u64, u32, i32) {
        let primitive_type = cmd.param_a;
        let index_buffer = cmd.param_b;
        let offset = u32::try_from(cmd.param_c >> 32).expect("high 32 bits fit u32");
        let base_vertex = u32::try_from(cmd.param_c & 0xFFFF_FFFF)
            .expect("low 32 bits fit u32")
            .cast_signed();
        let index_count = u32::try_from(cmd.param_d >> 8).expect("u64 >> 8 fits u32 by wire shape");
        let index_type = u32::try_from(cmd.param_d & 0xFF).expect("8-bit mask fits u32");
        (
            primitive_type,
            index_count,
            index_type,
            index_buffer,
            offset,
            base_vertex,
        )
    }

    #[test]
    fn draw_indexed_primitives_roundtrip_zero_base() {
        let cmd = Command::draw_indexed_primitives(
            PrimitiveType::Triangle,
            42,
            IndexType::UInt16,
            0xDEAD_BEEF_0000_0000,
            128,
            0,
        );
        let (prim, cnt, ty, buf, off, base) = decode_draw_indexed(&cmd);
        assert_eq!(prim, PrimitiveType::Triangle as u32);
        assert_eq!(cnt, 42);
        assert_eq!(ty, IndexType::UInt16 as u32);
        assert_eq!(buf, 0xDEAD_BEEF_0000_0000);
        assert_eq!(off, 128);
        assert_eq!(base, 0);
    }

    #[test]
    fn draw_indexed_primitives_roundtrip_positive_base() {
        let cmd = Command::draw_indexed_primitives(
            PrimitiveType::TriangleStrip,
            6,
            IndexType::UInt32,
            1,
            42,
            100_000,
        );
        let (_, _, _, _, off, base) = decode_draw_indexed(&cmd);
        assert_eq!(off, 42);
        assert_eq!(base, 100_000);
    }

    #[test]
    fn draw_indexed_primitives_roundtrip_negative_base() {
        // Sign-extension guard: `-100_000` must survive as `-100_000` on
        // the other end, not as a large positive u32 pattern.
        let cmd = Command::draw_indexed_primitives(
            PrimitiveType::TriangleStrip,
            6,
            IndexType::UInt32,
            1,
            0,
            -100_000,
        );
        let (_, _, _, _, off, base) = decode_draw_indexed(&cmd);
        assert_eq!(off, 0);
        assert_eq!(base, -100_000);
    }

    #[test]
    fn blit_command_layout_matches_wow64() {
        assert_eq!(core::mem::align_of::<BlitCommand>(), 8);
        // 4 cmd + 4 mip_level + 8 src_handle + 8 dst_handle + 8
        // src_offset + 8 bytes_per_row + 4 origin_x + 4 origin_y +
        // 4 region_w + 4 region_h + 8 dst_offset + 8 byte_size +
        // 4 depth + 4 bytes_per_image = 80
        assert_eq!(core::mem::size_of::<BlitCommand>(), 80);
    }

    #[test]
    fn copy_buffer_to_buffer_round_trip() {
        let cmd = BlitCommand::copy_buffer_to_buffer(&CopyBufferToBufferInfo {
            src_buffer: 0xAAAA_1111_2222_3333,
            dst_buffer: 0xBBBB_4444_5555_6666,
            src_offset: 0x1000,
            dst_offset: 0x2000,
            byte_size: 0x8000,
        });
        assert_eq!(cmd.cmd, BlitCommandType::CopyBufferToBuffer as u32);
        assert_eq!(cmd.src_handle, 0xAAAA_1111_2222_3333);
        assert_eq!(cmd.dst_handle, 0xBBBB_4444_5555_6666);
        assert_eq!(cmd.src_offset, 0x1000);
        assert_eq!(cmd.dst_offset, 0x2000);
        assert_eq!(cmd.byte_size, 0x8000);
    }

    #[test]
    fn copy_buffer_to_texture_round_trip() {
        let cmd = BlitCommand::copy_buffer_to_texture(&CopyBufferToTextureInfo {
            buffer_handle: 0xAAAA_BBBB_CCCC_DDDD,
            buffer_offset: 256,
            bytes_per_row: 1024,
            texture_handle: 0x1111_2222_3333_4444,
            mip_level: 2,
            origin_x: 10,
            origin_y: 20,
            region_w: 128,
            region_h: 64,
            depth: 1,
            bytes_per_image: 1024 * 64,
        });
        assert_eq!(cmd.cmd, BlitCommandType::CopyBufferToTexture as u32);
        assert_eq!(cmd.src_handle, 0xAAAA_BBBB_CCCC_DDDD);
        assert_eq!(cmd.dst_handle, 0x1111_2222_3333_4444);
        assert_eq!(cmd.src_offset, 256);
        assert_eq!(cmd.bytes_per_row, 1024);
        assert_eq!(cmd.mip_level, 2);
        assert_eq!(cmd.origin_x, 10);
        assert_eq!(cmd.origin_y, 20);
        assert_eq!(cmd.region_w, 128);
        assert_eq!(cmd.region_h, 64);
        assert_eq!(cmd.depth, 1);
        assert_eq!(cmd.bytes_per_image, 1024 * 64);
    }

    #[test]
    fn generate_mipmaps_packs_texture_handle_only() {
        let cmd = BlitCommand::generate_mipmaps(0xCAFE_F00D_0BAD_BEEF);
        assert_eq!(cmd.cmd, BlitCommandType::GenerateMipmaps as u32);
        assert_eq!(cmd.dst_handle, 0xCAFE_F00D_0BAD_BEEF);
        assert_eq!(cmd.src_handle, 0);
        assert_eq!(cmd.mip_level, 0);
        assert_eq!(cmd.src_offset, 0);
        assert_eq!(cmd.dst_offset, 0);
        assert_eq!(cmd.bytes_per_row, 0);
        assert_eq!(cmd.origin_x, 0);
        assert_eq!(cmd.origin_y, 0);
        assert_eq!(cmd.region_w, 0);
        assert_eq!(cmd.region_h, 0);
        assert_eq!(cmd.byte_size, 0);
    }

    #[test]
    fn copy_texture_to_texture_full_mip_zeros_sub_rect_fields() {
        let cmd = BlitCommand::copy_texture_to_texture_full_mip(0xDEAD, 0xBEEF, 1, 512, 256);
        assert_eq!(cmd.cmd, BlitCommandType::CopyTextureToTexture as u32);
        assert_eq!(cmd.src_handle, 0xDEAD);
        assert_eq!(cmd.dst_handle, 0xBEEF);
        assert_eq!(cmd.mip_level, 1);
        assert_eq!(cmd.src_offset, 0);
        assert_eq!(cmd.bytes_per_row, 0);
        assert_eq!(cmd.origin_x, 0);
        assert_eq!(cmd.origin_y, 0);
        assert_eq!(cmd.region_w, 512);
        assert_eq!(cmd.region_h, 256);
        assert_eq!(cmd.dst_offset, 0);
    }

    #[test]
    fn set_blend_color_roundtrip() {
        let cmd = Command::set_blend_color(0.25, 0.5, 0.75, 1.0);
        assert_eq!(cmd.cmd, CommandType::SetBlendColor as u32);
        let to_u32 = |v: u64| u32::try_from(v).expect("packed f32 bits fit u32");
        assert_eq!(cmd.param_a, 0.25_f32.to_bits());
        assert_eq!(to_u32(cmd.param_b), 0.5_f32.to_bits());
        assert_eq!(to_u32(cmd.param_c), 0.75_f32.to_bits());
        assert_eq!(to_u32(cmd.param_d), 1.0_f32.to_bits());
    }

    #[test]
    fn set_depth_bias_roundtrip() {
        let cmd = Command::set_depth_bias(-1.5, 0.25);
        assert_eq!(cmd.cmd, CommandType::SetDepthBias as u32);
        assert_eq!(cmd.param_a, (-1.5_f32).to_bits());
        assert_eq!(
            u32::try_from(cmd.param_b).expect("packed f32 bits fit u32"),
            0.25_f32.to_bits(),
        );
        assert_eq!(cmd.param_c, 0);
        assert_eq!(cmd.param_d, 0);
    }

    #[test]
    fn copy_texture_to_texture_sub_rect_packs_dst_origin() {
        let cmd = BlitCommand::copy_texture_to_texture_sub_rect(&CopyTextureSubRectInfo {
            src_texture: 0xAAAA,
            dst_texture: 0xBBBB,
            mip_level: 2,
            src_origin_x: 16,
            src_origin_y: 32,
            dst_origin_x: 100,
            dst_origin_y: 200,
            region_w: 64,
            region_h: 48,
        });
        assert_eq!(cmd.cmd, BlitCommandType::CopyTextureToTexture as u32);
        assert_eq!(cmd.src_handle, 0xAAAA);
        assert_eq!(cmd.dst_handle, 0xBBBB);
        assert_eq!(cmd.mip_level, 2);
        assert_eq!(cmd.origin_x, 16);
        assert_eq!(cmd.origin_y, 32);
        assert_eq!(cmd.region_w, 64);
        assert_eq!(cmd.region_h, 48);
        // dst origin packed as (y << 32) | x — decoder splits the same way.
        assert_eq!(cmd.dst_offset & 0xFFFF_FFFF, 100);
        assert_eq!((cmd.dst_offset >> 32) & 0xFFFF_FFFF, 200);
    }

    #[test]
    fn draw_indexed_primitives_packs_index_count_and_type() {
        let cmd = Command::draw_indexed_primitives(
            PrimitiveType::Triangle,
            u32::MAX,
            IndexType::UInt32,
            0,
            0,
            0,
        );
        let (_, cnt, ty, _, _, _) = decode_draw_indexed(&cmd);
        assert_eq!(cnt, u32::MAX);
        assert_eq!(ty, IndexType::UInt32 as u32);
    }
}
