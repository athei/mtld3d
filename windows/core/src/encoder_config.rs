//! Explicit encoding of resolved device configuration and GPU capabilities.
//!
//! Native construction reads every field from the API snapshot and never reapplies defaults.

use mtld3d_shared::{
    encoder_wire::{WireError, WireReader, WireWriter},
    mtl::{ColorSpacePolicy, DeviceCapsFlags, SoftwareCursorPolicy},
};

use crate::{
    config::{AdapterSpoof, CursorScale, Mtld3dConfig},
    encoder_value::WireValue,
    gpu_caps::GpuCaps,
};

#[cfg(test)]
mod tests;

impl WireValue for Mtld3dConfig {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        self.caps_all.write_wire(writer)?;
        self.main_thread_checker.write_wire(writer)?;
        self.expand_packed16.write_wire(writer)?;
        self.deny_float32_filtering.write_wire(writer)?;
        self.managed_memory.write_wire(writer)?;
        self.linear_align256.write_wire(writer)?;
        self.hdr_enable.write_wire(writer)?;
        self.color_space.write_wire(writer)?;
        self.cursor_scale.write_wire(writer)?;
        self.cursor_software.write_wire(writer)?;
        self.shader_cache_enable.write_wire(writer)?;
        self.shader_async_compile.write_wire(writer)?;
        self.log_dir.write_wire(writer)?;
        self.bytecode_dump_dir.write_wire(writer)?;
        self.skip_shaders.write_wire(writer)?;
        self.present_gate_file.write_wire(writer)?;
        self.query_flush_immediate.write_wire(writer)?;
        self.query_event_immediate.write_wire(writer)?;
        self.depth_alias_same_size.write_wire(writer)?;
        self.buffer_ignore_lock_bounds.write_wire(writer)?;
        self.vbib_retention_cap_bytes.write_wire(writer)?;
        self.vram_budget_cap_bytes.write_wire(writer)?;
        self.pagebox_pool_cap_bytes.write_wire(writer)?;
        self.present_max_fps.write_wire(writer)?;
        self.render_scale_percent.write_wire(writer)?;
        self.render_lod_bias.write_wire(writer)?;
        self.adapter_spoof.write_wire(writer)?;
        self.df_formats.write_wire(writer)?;
        Ok(())
    }

    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Ok(Self {
            caps_all: <bool>::read_wire(reader)?,
            main_thread_checker: <bool>::read_wire(reader)?,
            expand_packed16: <bool>::read_wire(reader)?,
            deny_float32_filtering: <bool>::read_wire(reader)?,
            managed_memory: <bool>::read_wire(reader)?,
            linear_align256: <bool>::read_wire(reader)?,
            hdr_enable: <bool>::read_wire(reader)?,
            color_space: <ColorSpacePolicy>::read_wire(reader)?,
            cursor_scale: <CursorScale>::read_wire(reader)?,
            cursor_software: <SoftwareCursorPolicy>::read_wire(reader)?,
            shader_cache_enable: <bool>::read_wire(reader)?,
            shader_async_compile: <bool>::read_wire(reader)?,
            log_dir: <String>::read_wire(reader)?,
            bytecode_dump_dir: <String>::read_wire(reader)?,
            skip_shaders: <Vec<u64>>::read_wire(reader)?,
            present_gate_file: <String>::read_wire(reader)?,
            query_flush_immediate: <bool>::read_wire(reader)?,
            query_event_immediate: <bool>::read_wire(reader)?,
            depth_alias_same_size: <bool>::read_wire(reader)?,
            buffer_ignore_lock_bounds: <bool>::read_wire(reader)?,
            vbib_retention_cap_bytes: <u64>::read_wire(reader)?,
            vram_budget_cap_bytes: <u64>::read_wire(reader)?,
            pagebox_pool_cap_bytes: <u64>::read_wire(reader)?,
            present_max_fps: <u32>::read_wire(reader)?,
            render_scale_percent: <u32>::read_wire(reader)?,
            render_lod_bias: <bool>::read_wire(reader)?,
            adapter_spoof: <AdapterSpoof>::read_wire(reader)?,
            df_formats: <bool>::read_wire(reader)?,
        })
    }
}

impl WireValue for AdapterSpoof {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u8(match self {
            Self::None => 0,
            Self::Nvidia => 1,
            Self::Amd => 2,
        })
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::None),
            1 => Ok(Self::Nvidia),
            2 => Ok(Self::Amd),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl WireValue for ColorSpacePolicy {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u8(match self {
            Self::Passthrough => 0,
            Self::Accurate => 1,
        })
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::Passthrough),
            1 => Ok(Self::Accurate),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl WireValue for SoftwareCursorPolicy {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u8(match self {
            Self::Auto => 0,
            Self::On => 1,
            Self::Off => 2,
        })
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::Auto),
            1 => Ok(Self::On),
            2 => Ok(Self::Off),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl WireValue for CursorScale {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        match self {
            Self::Auto => writer.u8(0),
            Self::Fixed(value) => {
                writer.u8(1)?;
                writer.u32(*value)
            }
        }
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::Auto),
            1 => Ok(Self::Fixed(reader.u32()?)),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl WireValue for GpuCaps {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        self.unified_memory.write_wire(writer)?;
        self.min_linear_texture_align.write_wire(writer)?;
        self.device_caps.bits().write_wire(writer)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Ok(Self {
            unified_memory: bool::read_wire(reader)?,
            min_linear_texture_align: u32::read_wire(reader)?,
            device_caps: DeviceCapsFlags::from_bits(u32::read_wire(reader)?)
                .ok_or(WireError::InvalidValue)?,
        })
    }
}
