//! Aligned flat commands in an immutable retained frame arena.

/// The fixed prefix of every command, followed by its exact payload bytes.
///
/// `record_bytes` excludes alignment padding. Only the allocating runtime writes
/// records before admission seals the frame. The next record begins at the
/// following eight-byte boundary within the same retained command region.
#[repr(C)]
pub struct CommandHeader {
    pub opcode: u16,
    pub operand: u16,
    pub record_bytes: u32,
}

/// One contiguous command region, borrowed until the frame lease ends.
#[repr(C, align(8))]
pub struct CommandRegion {
    pub address: u64,
    pub used_bytes: u32,
    pub reserved: u32,
}

/// Command storage aligns the payload independently of the four-byte header alignment.
pub const COMMAND_ALIGNMENT: usize = 8;
pub const COMMAND_HEADER_BYTES: usize = 8;
pub const COMMAND_REGION_BYTES: usize = 16;

const _: () = {
    assert!(cfg!(target_endian = "little"));
    assert!(size_of::<CommandHeader>() == COMMAND_HEADER_BYTES);
    assert!(align_of::<CommandHeader>() == 4);
    assert!(COMMAND_ALIGNMENT.is_power_of_two());
    assert!(COMMAND_HEADER_BYTES.is_multiple_of(COMMAND_ALIGNMENT));
    assert!(size_of::<CommandRegion>() == COMMAND_REGION_BYTES);
    assert!(align_of::<CommandRegion>() == 8);
    assert!(core::mem::offset_of!(CommandRegion, address) == 0);
    assert!(core::mem::offset_of!(CommandRegion, used_bytes) == 8);
    assert!(core::mem::offset_of!(CommandRegion, reserved) == 12);
    assert!(core::mem::offset_of!(CommandHeader, opcode) == 0);
    assert!(core::mem::offset_of!(CommandHeader, operand) == 2);
    assert!(core::mem::offset_of!(CommandHeader, record_bytes) == 4);
};
