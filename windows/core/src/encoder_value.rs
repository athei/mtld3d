//! Field-wise codecs for immutable encoder values.
//!
//! These implementations describe values, never Rust object representations.
//! Pointer-bearing snapshots and allocation leases use the frame codec instead.

use mtld3d_shared::encoder_wire::{WireError, WireReader, WireWriter};

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

scalar_codec!(u32, u64);

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
        writer.u32(u32::try_from(self.len()).map_err(|_| WireError::TooLarge)?)?;
        for value in self {
            value.write_wire(writer)?;
        }
        Ok(())
    }

    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        let count = usize::try_from(reader.u32()?).map_err(|_| WireError::TooLarge)?;
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
