//! Field-wise codecs for immutable encoder values.
//!
//! These implementations describe values, never Rust object representations.
//! Pointer-bearing snapshots and allocation leases use the frame codec instead.

use mtld3d_shared::{
    VertexAttrDesc,
    encoder_wire::{WireError, WireReader, WireWriter},
    mtl::{IndexType, PixelFormat, PrimitiveType, VertexFormat, VertexStepFunction},
};

use crate::{
    depth_stencil_state::{DepthStencilSnapshot, StencilFaceState},
    draw_data::{DepthScissorFlags, DepthStencilFlags, RenderStateSnapshot, StageBinding},
    dxso::{
        FfPsKey, FfStage, FfStageFlags, FfVsFlags, FfVsKey, VariantFlags, VariantKey,
        VsSamplerKinds,
    },
    pipeline_state::{
        ExtraColorAttachments, PipelineAttachFlags, PipelineRsBits, PipelineRsFlags, StreamLayout,
    },
};

#[cfg(test)]
mod tests;

pub trait WireValue: Sized {
    /// Minimum encoded size, used to reject impossible collection lengths before allocation.
    const MIN_WIRE_BYTES: usize = 1;

    /// Write explicit fields to the operation payload.
    ///
    /// # Errors
    ///
    /// Returns allocation, size or invalid-value errors from the wire writer.
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError>;

    /// Decode explicit fields from the current operation payload.
    ///
    /// # Errors
    ///
    /// Returns an error for truncated fields, invalid tags or impossible lengths.
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError>;
}

macro_rules! scalar_codec {
    ($($scalar:ident),+ $(,)?) => {$ (
        impl WireValue for $scalar {
            const MIN_WIRE_BYTES: usize = size_of::<Self>();

            #[inline]
            fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
                writer.$scalar(*self)
            }

            #[inline]
            fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
                reader.$scalar()
            }
        }
    )+};
}

scalar_codec!(u8, u16, u32, u64, i32, f32);

impl WireValue for bool {
    #[inline]
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u8(u8::from(*self))
    }

    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl<T: WireValue, const N: usize> WireValue for [T; N] {
    const MIN_WIRE_BYTES: usize = N * T::MIN_WIRE_BYTES;

    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        for value in self {
            value.write_wire(writer)?;
        }
        Ok(())
    }

    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        let mut values = [const { None }; N];
        for value in &mut values {
            *value = Some(T::read_wire(reader)?);
        }
        Ok(values.map(|value| value.expect("every array element was decoded")))
    }
}

impl<T: WireValue> WireValue for Option<T> {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        self.is_some().write_wire(writer)?;
        if let Some(value) = self {
            value.write_wire(writer)?;
        }
        Ok(())
    }

    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        if bool::read_wire(reader)? {
            T::read_wire(reader).map(Some)
        } else {
            Ok(None)
        }
    }
}

impl<T: WireValue> WireValue for Vec<T> {
    const MIN_WIRE_BYTES: usize = 4;

    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        // Nonempty collections of zero-byte values cannot be bounded by payload size.
        if T::MIN_WIRE_BYTES == 0 && !self.is_empty() {
            return Err(WireError::InvalidValue);
        }
        writer.u32(u32::try_from(self.len()).map_err(|_| WireError::TooLarge)?)?;
        for value in self {
            value.write_wire(writer)?;
        }
        Ok(())
    }

    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        let count = usize::try_from(reader.u32()?).map_err(|_| WireError::TooLarge)?;
        if count != 0 && T::MIN_WIRE_BYTES == 0 {
            return Err(WireError::InvalidValue);
        }
        if count
            .checked_mul(T::MIN_WIRE_BYTES)
            .ok_or(WireError::TooLarge)?
            > reader.remaining_len()
        {
            return Err(WireError::Truncated);
        }
        let mut values = Self::new();
        values
            .try_reserve(count)
            .map_err(|_| WireError::AllocationFailed)?;
        for _ in 0..count {
            values.push(T::read_wire(reader)?);
        }
        Ok(values)
    }
}

impl WireValue for String {
    const MIN_WIRE_BYTES: usize = 4;

    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u32(u32::try_from(self.len()).map_err(|_| WireError::TooLarge)?)?;
        writer.bytes(self.as_bytes())
    }

    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        let length = reader.u32()?;
        let bytes = reader.bytes(length)?;
        let text = std::str::from_utf8(bytes).map_err(|_| WireError::InvalidValue)?;
        let mut value = Self::new();
        value
            .try_reserve(text.len())
            .map_err(|_| WireError::AllocationFailed)?;
        value.push_str(text);
        Ok(value)
    }
}

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

flags_codec!(
    FfVsFlags,
    FfStageFlags,
    VariantFlags,
    PipelineRsFlags,
    PipelineAttachFlags,
    DepthScissorFlags,
    DepthStencilFlags
);

macro_rules! metal_enum_codec {
    ($($value:ty),+ $(,)?) => {$ (
        impl WireValue for $value {
            const MIN_WIRE_BYTES: usize = 4;

            fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
                writer.u32(*self as u32)
            }

            fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
                Self::from_repr(reader.u32()?).ok_or(WireError::InvalidValue)
            }
        }
    )+};
}

metal_enum_codec!(
    PixelFormat,
    VertexFormat,
    PrimitiveType,
    IndexType,
    VertexStepFunction
);

fields_codec!(FfVsKey {
    flags,
    input_tex_coord_count,
    tex_coord_count,
    light_active_mask,
    light_directional_mask,
    light_spot_mask,
    diffuse_source,
    ambient_source,
    specular_source,
    emissive_source,
    fog_mode,
    tci_modes,
    tci_coord_indices,
    tex_coord_dims,
    tt_flags,
    vertex_blend_count,
    declared_weights_count,
    clip_plane_count
});

fields_codec!(FfStage {
    color_op,
    color_arg1,
    color_arg2,
    alpha_op,
    alpha_arg1,
    alpha_arg2,
    flags
});

fields_codec!(FfPsKey {
    stages,
    specular_add,
    tt_projected_mask
});

fields_codec!(VariantKey {
    alpha_func,
    fog_mode,
    fog_table_mode,
    depth_sampler_mask,
    depth_fetch_mask,
    fetch4_mask,
    fetch4_alpha_mask,
    raw_depth_red_mask,
    volume_sampler_mask,
    cube_sampler_mask,
    tt_projected_mask,
    color_out_mask,
    sample_mask,
    flags
});

fields_codec!(VsSamplerKinds {
    volume_mask,
    cube_mask
});

fields_codec!(StencilFaceState {
    func,
    fail_op,
    depth_fail_op,
    pass_op
});

fields_codec!(DepthStencilSnapshot {
    depth_enable,
    depth_write,
    depth_func,
    stencil_enable,
    front,
    back,
    read_mask,
    write_mask
});

fields_codec!(PipelineRsBits {
    flags,
    src_blend,
    dst_blend,
    blend_op,
    src_blend_alpha,
    dst_blend_alpha,
    blend_op_alpha,
    color_write_mask,
    color_write_mask_ext
});

fields_codec!(ExtraColorAttachments {
    formats,
    present_mask,
    has_alpha_mask
});

fields_codec!(StreamLayout {
    stride,
    step,
    step_rate
});

fields_codec!(VertexAttrDesc {
    attr_index,
    buffer_index,
    format,
    offset
});

fields_codec!(StageBinding {
    texture_id,
    sampler_state
});

fields_codec!(RenderStateSnapshot {
    pipeline_rs,
    depth_scissor,
    depth_stencil_state,
    cull_mode,
    fill_mode,
    scissor_rect,
    blend_factor,
    depth_bias,
    slope_scale_depth_bias,
    stencil_ref,
    sample_mask
});
