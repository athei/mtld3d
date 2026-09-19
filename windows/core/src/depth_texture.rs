//! Packed dynamic depth pixels and their independent native transfer planes.

use mtld3d_types::{D3DFMT_D16, D3DFMT_D24S8, D3DFMT_D24X8};

/// A packed depth format whose normalized codes survive float32 storage.
pub enum PackedDepth {
    D16,
    D24X8,
    D24S8,
}

impl PackedDepth {
    /// Classify the three CPU-uploadable plain depth formats.
    #[must_use]
    pub const fn from_d3d(format: u32) -> Option<Self> {
        match format {
            D3DFMT_D16 => Some(Self::D16),
            D3DFMT_D24X8 => Some(Self::D24X8),
            D3DFMT_D24S8 => Some(Self::D24S8),
            _ => None,
        }
    }

    /// Packed bytes per logical texel.
    #[must_use]
    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            Self::D16 => 2,
            Self::D24X8 | Self::D24S8 => 4,
        }
    }

    /// Whether a second native plane carries stencil bytes.
    #[must_use]
    pub const fn has_stencil(&self) -> bool {
        matches!(self, Self::D24S8)
    }

    /// Decode one checked packed pixel to float32 depth and stencil.
    ///
    /// The caller supplies exactly `bytes_per_pixel()` initialized bytes.
    ///
    /// # Panics
    /// Panics if the input is shorter than the packed format.
    #[must_use]
    pub fn decode(&self, pixel: &[u8]) -> (f32, u8) {
        if matches!(self, Self::D16) {
            return (
                f32::from(u16::from_le_bytes([pixel[0], pixel[1]])) / 65_535.0,
                0,
            );
        }
        let (low, middle, high, stencil) = if self.has_stencil() {
            (pixel[1], pixel[2], pixel[3], pixel[0])
        } else {
            (pixel[0], pixel[1], pixel[2], 0)
        };
        let code = f32::from(high).mul_add(65_536.0, f32::from(u16::from_le_bytes([low, middle])));
        (code / 16_777_215.0, stencil)
    }

    /// Pack GPU-authored planes, rounding normalized depth to the nearest code.
    ///
    /// GPU-written D24X8 pixels zero their unused high byte. CPU-authored
    /// staging never passes through this conversion and retains that byte.
    /// NaN and negative values become zero; values above one saturate.
    ///
    /// # Panics
    /// Panics if the output is shorter than the packed format.
    pub fn encode(&self, depth: f32, stencil: u8, pixel: &mut [u8]) {
        let max = if matches!(self, Self::D16) {
            65_535.0
        } else {
            16_777_215.0
        };
        let depth = if depth.is_nan() {
            0.0
        } else {
            depth.clamp(0.0, 1.0)
        };
        // Adding 2^52 rounds a nonnegative binary64 value below 2^24 to
        // an integer in its mantissa, without an unchecked float-to-int cast.
        let rounded = (f64::from(depth) * max + 4_503_599_627_370_496.0).to_bits()
            - 4_503_599_627_370_496.0_f64.to_bits();
        let bytes = rounded.to_le_bytes();
        match self {
            Self::D16 => pixel[..2].copy_from_slice(&bytes[..2]),
            Self::D24X8 => pixel[..4].copy_from_slice(&[bytes[0], bytes[1], bytes[2], 0]),
            Self::D24S8 => pixel[..4].copy_from_slice(&[stencil, bytes[0], bytes[1], bytes[2]]),
        }
    }
}

/// Aligned rows for independently transferred depth and stencil planes.
pub struct PlaneLayout {
    pub width: usize,
    pub height: usize,
    pub depth_pitch: usize,
    pub stencil_pitch: usize,
}

impl PlaneLayout {
    /// Check the extent and align both native plane rows.
    #[must_use]
    pub fn new(width: u32, height: u32, alignment: u32) -> Option<Self> {
        let width = usize::try_from(width).ok()?;
        let height = usize::try_from(height).ok()?;
        let alignment = usize::try_from(alignment.max(4)).ok()?;
        if width == 0 || height == 0 || !alignment.is_power_of_two() {
            return None;
        }
        let align = |bytes: usize| {
            bytes
                .checked_add(alignment - 1)
                .map(|n| n & !(alignment - 1))
        };
        let depth_pitch = align(width.checked_mul(4)?)?;
        let stencil_pitch = align(width)?;
        depth_pitch.checked_mul(height)?;
        stencil_pitch.checked_mul(height)?;
        Some(Self {
            width,
            height,
            depth_pitch,
            stencil_pitch,
        })
    }

    /// Convert a checked packed rectangle into separately aligned native planes.
    ///
    /// A missing stencil plane is valid only for a depth-only format. Bounds
    /// are checked before any output is written, so failure cannot upload half
    /// of a combined pixel layout.
    pub fn unpack(
        &self,
        format: &PackedDepth,
        packed: &[u8],
        packed_pitch: usize,
        depth: &mut [u8],
        stencil: &mut [u8],
    ) -> bool {
        let Some(row_bytes) = self.width.checked_mul(format.bytes_per_pixel()) else {
            return false;
        };
        let Some(end) = (self.height - 1)
            .checked_mul(packed_pitch)
            .and_then(|n| n.checked_add(row_bytes))
        else {
            return false;
        };
        if packed_pitch < row_bytes
            || packed.len() < end
            || depth.len() < self.depth_pitch * self.height
            || (format.has_stencil() && stencil.len() < self.stencil_pitch * self.height)
        {
            return false;
        }
        for y in 0..self.height {
            for x in 0..self.width {
                let packed_offset = y * packed_pitch + x * format.bytes_per_pixel();
                let (value, mask) = format.decode(&packed[packed_offset..]);
                let depth_offset = y * self.depth_pitch + x * 4;
                depth[depth_offset..depth_offset + 4].copy_from_slice(&value.to_le_bytes());
                if format.has_stencil() {
                    stencil[y * self.stencil_pitch + x] = mask;
                }
            }
        }
        true
    }

    /// Pack complete GPU planes into the logical CPU level after readback.
    pub fn pack(
        &self,
        format: &PackedDepth,
        depth: &[u8],
        stencil: &[u8],
        packed: &mut [u8],
        packed_pitch: usize,
    ) -> bool {
        let Some(row_bytes) = self.width.checked_mul(format.bytes_per_pixel()) else {
            return false;
        };
        let Some(end) = (self.height - 1)
            .checked_mul(packed_pitch)
            .and_then(|n| n.checked_add(row_bytes))
        else {
            return false;
        };
        if packed_pitch < row_bytes
            || packed.len() < end
            || depth.len() < self.depth_pitch * self.height
            || (format.has_stencil() && stencil.len() < self.stencil_pitch * self.height)
        {
            return false;
        }
        for y in 0..self.height {
            for x in 0..self.width {
                let offset = y * self.depth_pitch + x * 4;
                let value = f32::from_le_bytes([
                    depth[offset],
                    depth[offset + 1],
                    depth[offset + 2],
                    depth[offset + 3],
                ]);
                let mask = if format.has_stencil() {
                    stencil[y * self.stencil_pitch + x]
                } else {
                    0
                };
                let packed_offset = y * packed_pitch + x * format.bytes_per_pixel();
                format.encode(value, mask, &mut packed[packed_offset..]);
            }
        }
        true
    }
}

#[cfg(test)]
mod tests;
