//! Field-wise wire codecs for encoder control operations.
//!
//! Metal handles are borrowed identities from the trusted in-process producer.
//! Allocation leases, query mailboxes and program ownership are decoded by the frame owner.

use mtld3d_shared::{
    MetalHandle,
    encoder_wire::{WireError, WireReader, WireWriter},
    mtl::{Swizzle, TextureCreateFlags, TextureUsage},
};

use crate::{
    encoder_data::{
        BindColorOp, BindDepthOp, BindDepthOpFlags, BlitSide, CarryDepthOp, ClearColorOp,
        ClearColorRectsOp, ClearDepthStencilOp, ClearDepthStencilRectsOp, ColorFillOp,
        ColorFillTarget, ColorRtBinding, DepthBinding, DepthTransfer, DestroyTextureOp,
        GenerateMipmapsOp, GenerateMipmapsOrderedOp, NoteColorReadOp, ResampledUpload,
        ResolveDepthSurfaceOp, ResolveDepthTextureOp, ResolveDynamicDepthOp, RetireColorOp,
        RetireDepthOp, RetiredColorTarget, RtBinding, SetDumpDrawOp, SetVertexSamplerOp,
        SetVertexTextureOp, SetViewportOp, StretchBlitOp, StretchKind, StretchSurfaceFlags,
        StretchSurfaceInfo, TextureInfo, UnbindExtraColorOp, UploadTextureOpFlags,
    },
    encoder_value::WireValue,
    render_scale::RenderScale,
    stretch_rect::StretchRegion,
};

macro_rules! fields_codec {
    ($value:ty { $($field:ident),+ $(,)? }) => {
        impl WireValue for $value {
            fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
                $(self.$field.write_wire(writer)?;)+
                Ok(())
            }
            fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
                Ok(Self { $($field: WireValue::read_wire(reader)?,)+ })
            }
        }
    };
}

macro_rules! flags_codec {
    ($($value:ty),+ $(,)?) => {$ (
        impl WireValue for $value {
            fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
                self.bits().write_wire(writer)
            }
            fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
                Self::from_bits(WireValue::read_wire(reader)?).ok_or(WireError::InvalidValue)
            }
        }
    )+};
}

impl<K> WireValue for MetalHandle<K> {
    const MIN_WIRE_BYTES: usize = 8;
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u64(self.raw())
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        let raw = reader.u64()?;
        if raw != 0 && !reader.has_trusted_addresses() {
            return Err(WireError::InvalidValue);
        }
        // SAFETY: zero is the null sentinel. For nonzero handles, the unsafe
        // reader constructor established the schema, object-kind and retained
        // lifetime contract, which nested record readers preserve.
        Ok(unsafe { Self::new(raw) })
    }
}

impl<A: WireValue, B: WireValue> WireValue for (A, B) {
    const MIN_WIRE_BYTES: usize = A::MIN_WIRE_BYTES + B::MIN_WIRE_BYTES;
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        self.0.write_wire(writer)?;
        self.1.write_wire(writer)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Ok((A::read_wire(reader)?, B::read_wire(reader)?))
    }
}

impl<A: WireValue, B: WireValue, C: WireValue, D: WireValue> WireValue for (A, B, C, D) {
    const MIN_WIRE_BYTES: usize =
        A::MIN_WIRE_BYTES + B::MIN_WIRE_BYTES + C::MIN_WIRE_BYTES + D::MIN_WIRE_BYTES;
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        self.0.write_wire(writer)?;
        self.1.write_wire(writer)?;
        self.2.write_wire(writer)?;
        self.3.write_wire(writer)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Ok((
            A::read_wire(reader)?,
            B::read_wire(reader)?,
            C::read_wire(reader)?,
            D::read_wire(reader)?,
        ))
    }
}

impl WireValue for RenderScale {
    const MIN_WIRE_BYTES: usize = 4;
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u32(self.percent())
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        let percent = reader.u32()?;
        if !(1..=100).contains(&percent) {
            return Err(WireError::InvalidValue);
        }
        Ok(Self::from_percent(percent))
    }
}

impl WireValue for Swizzle {
    const MIN_WIRE_BYTES: usize = 4;
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u32(*self as u32)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Self::from_repr(reader.u32()?).ok_or(WireError::InvalidValue)
    }
}

flags_codec!(
    TextureCreateFlags,
    TextureUsage,
    BindDepthOpFlags,
    UploadTextureOpFlags,
    StretchSurfaceFlags
);
fields_codec!(StretchRegion { x, y, w, h });
fields_codec!(StretchSurfaceInfo {
    kind,
    width,
    height,
    texture_size,
    scale,
    format,
    mip_level,
    slice,
    pool,
    flags,
    autogen_texture_id,
    msaa,
    msaa_srgb,
    sample_count
});
fields_codec!(ColorRtBinding {
    texture,
    msaa_texture,
    msaa_srgb_texture,
    sample_count,
    logical_size,
    size,
    format,
    has_alpha,
    scale,
    subresource
});
fields_codec!(BlitSide {
    handle,
    rect,
    dims,
    mip,
    slice,
    msaa,
    msaa_srgb,
    sample_count
});
fields_codec!(ResampledUpload {
    color_handle,
    format,
    logical,
    texture,
    bytes_per_row,
    msaa,
    msaa_srgb,
    sample_count
});
fields_codec!(ColorFillTarget {
    texture,
    logical_size,
    texture_size,
    format,
    scale,
    subresource,
    rect,
    rgba,
    msaa,
    msaa_srgb,
    sample_count,
    regenerate_mipmaps
});
fields_codec!(TextureInfo {
    texture_id,
    d3d_format,
    width,
    height,
    depth,
    levels,
    pixel_format,
    create_flags,
    swizzle,
    usage_flags
});
fields_codec!(RetiredColorTarget {
    base,
    srgb,
    msaa,
    msaa_srgb
});
fields_codec!(DepthTransfer {
    source,
    source_level,
    source_size,
    source_format,
    source_samples,
    destination,
    destination_size,
    destination_format
});
fields_codec!(SetViewportOp {
    x,
    y,
    width,
    height,
    min_z,
    max_z
});
fields_codec!(SetVertexSamplerOp { slot, state });
fields_codec!(SetVertexTextureOp { slot, id });
fields_codec!(BindDepthOp {
    binding,
    sample_count,
    flags
});
fields_codec!(BindColorOp { slot, info, scale });
fields_codec!(GenerateMipmapsOrderedOp { old_id });
fields_codec!(UnbindExtraColorOp { slot });
fields_codec!(DestroyTextureOp { tex_id });
fields_codec!(NoteColorReadOp { src });
fields_codec!(ResolveDepthSurfaceOp { transfer });
fields_codec!(StretchBlitOp {
    src_info,
    dst_info,
    src_region,
    dst_region,
    mip_level,
    render_quad,
    filter
});
fields_codec!(ColorFillOp { kind, fill });
fields_codec!(CarryDepthOp {
    prev_id,
    cur_id,
    mip_w,
    mip_h
});
fields_codec!(ClearColorOp {
    r_bits,
    g_bits,
    b_bits,
    a_bits,
    srgb_write
});
fields_codec!(ClearColorRectsOp {
    r_bits,
    g_bits,
    b_bits,
    a_bits,
    srgb_write,
    rects
});
fields_codec!(ClearDepthStencilRectsOp {
    depth,
    stencil,
    list
});
fields_codec!(ClearDepthStencilOp { depth, stencil });
fields_codec!(ResolveDynamicDepthOp { id, info });
fields_codec!(ResolveDepthTextureOp { id, w, h, format });
fields_codec!(RetireColorOp { retired });
fields_codec!(RetireDepthOp { depth });
fields_codec!(GenerateMipmapsOp { texture_id });
fields_codec!(SetDumpDrawOp { seq });

impl WireValue for DepthBinding {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        match self {
            Self::None => {
                writer.u8(0)?;
            }
            Self::Eager(handle, size, scale) => {
                writer.u8(1)?;
                handle.write_wire(writer)?;
                size.write_wire(writer)?;
                scale.write_wire(writer)?;
            }
            Self::Lazy(info, level, scale) => {
                writer.u8(2)?;
                info.write_wire(writer)?;
                level.write_wire(writer)?;
                scale.write_wire(writer)?;
            }
        }
        Ok(())
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::None),
            1 => Ok(Self::Eager(
                WireValue::read_wire(reader)?,
                WireValue::read_wire(reader)?,
                WireValue::read_wire(reader)?,
            )),
            2 => Ok(Self::Lazy(
                WireValue::read_wire(reader)?,
                WireValue::read_wire(reader)?,
                WireValue::read_wire(reader)?,
            )),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl WireValue for StretchKind {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        match self {
            Self::Texture(info) => {
                writer.u8(0)?;
                info.write_wire(writer)?;
            }
            Self::Backbuffer(handle) => {
                writer.u8(1)?;
                handle.write_wire(writer)?;
            }
            Self::DepthStencil(handle) => {
                writer.u8(2)?;
                handle.write_wire(writer)?;
            }
        }
        Ok(())
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::Texture(WireValue::read_wire(reader)?)),
            1 => Ok(Self::Backbuffer(WireValue::read_wire(reader)?)),
            2 => Ok(Self::DepthStencil(WireValue::read_wire(reader)?)),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl WireValue for RtBinding {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        match self {
            Self::Backbuffer {
                handle,
                msaa,
                msaa_srgb,
                sample_count,
                width,
                height,
            } => {
                writer.u8(0)?;
                handle.write_wire(writer)?;
                msaa.write_wire(writer)?;
                msaa_srgb.write_wire(writer)?;
                sample_count.write_wire(writer)?;
                width.write_wire(writer)?;
                height.write_wire(writer)?;
            }
            Self::StandaloneColor {
                handle,
                srgb,
                msaa,
                msaa_srgb,
                sample_count,
                format,
                has_alpha,
                width,
                height,
            } => {
                writer.u8(1)?;
                handle.write_wire(writer)?;
                srgb.write_wire(writer)?;
                msaa.write_wire(writer)?;
                msaa_srgb.write_wire(writer)?;
                sample_count.write_wire(writer)?;
                format.write_wire(writer)?;
                has_alpha.write_wire(writer)?;
                width.write_wire(writer)?;
                height.write_wire(writer)?;
            }
            Self::Texture {
                info,
                has_alpha,
                width,
                height,
                slice,
                level,
            } => {
                writer.u8(2)?;
                info.write_wire(writer)?;
                has_alpha.write_wire(writer)?;
                width.write_wire(writer)?;
                height.write_wire(writer)?;
                slice.write_wire(writer)?;
                level.write_wire(writer)?;
            }
        }
        Ok(())
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::Backbuffer {
                handle: WireValue::read_wire(reader)?,
                msaa: WireValue::read_wire(reader)?,
                msaa_srgb: WireValue::read_wire(reader)?,
                sample_count: WireValue::read_wire(reader)?,
                width: WireValue::read_wire(reader)?,
                height: WireValue::read_wire(reader)?,
            }),
            1 => Ok(Self::StandaloneColor {
                handle: WireValue::read_wire(reader)?,
                srgb: WireValue::read_wire(reader)?,
                msaa: WireValue::read_wire(reader)?,
                msaa_srgb: WireValue::read_wire(reader)?,
                sample_count: WireValue::read_wire(reader)?,
                format: WireValue::read_wire(reader)?,
                has_alpha: WireValue::read_wire(reader)?,
                width: WireValue::read_wire(reader)?,
                height: WireValue::read_wire(reader)?,
            }),
            2 => Ok(Self::Texture {
                info: WireValue::read_wire(reader)?,
                has_alpha: WireValue::read_wire(reader)?,
                width: WireValue::read_wire(reader)?,
                height: WireValue::read_wire(reader)?,
                slice: WireValue::read_wire(reader)?,
                level: WireValue::read_wire(reader)?,
            }),
            _ => Err(WireError::InvalidValue),
        }
    }
}

#[cfg(test)]
mod tests;
