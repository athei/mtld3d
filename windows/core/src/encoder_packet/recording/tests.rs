use super::*;

#[test]
fn failed_rollover_keeps_committed_region_and_reuses_next_reservation() {
    let mut arena = ScratchArena::with_chunk_size(64);
    let mut commands = RecordedCommands::new();
    commands
        .push_fixed_record(&mut arena, 1, 0, 56, |bytes| {
            bytes.fill(1);
            Ok(())
        })
        .unwrap();
    let first = commands.regions[0].address;
    assert_eq!(commands.regions[0].used_bytes, 64);
    assert_eq!(
        commands.push_fixed_record(&mut arena, 2, 0, 16, |bytes| {
            bytes.fill(2);
            Err(WireError::InvalidValue)
        }),
        Err(WireError::InvalidValue)
    );
    assert_eq!(commands.regions.len(), 1);
    assert_eq!(commands.regions[0].address, first);
    assert_eq!(commands.regions[0].used_bytes, 64);
    assert_eq!(arena.bytes_used(), 64);
    commands
        .push_fixed_record(&mut arena, 3, 0, 16, |bytes| {
            bytes.fill(3);
            Ok(())
        })
        .unwrap();
    assert_eq!(commands.regions.len(), 2);
    assert_eq!(commands.regions[1].used_bytes, 24);
    assert_ne!(commands.regions[1].address, first);
    assert_eq!(arena.chunk_count(), 2);
}

#[test]
fn oversized_payload_and_two_cursors_reuse_one_pool_after_clear() {
    let mut arena = ScratchArena::with_chunk_size(64);
    let mut commands = RecordedCommands::new();
    commands
        .push_fixed_record(&mut arena, 1, 0, 0, |_| Ok(()))
        .unwrap();
    let first = commands.regions[0].address;
    let oversized = arena.alloc(&[7; 128]);
    let ordinary = arena.alloc(&[8; 16]);
    assert_ne!(oversized, first);
    assert_ne!(ordinary, first);
    commands
        .push_fixed_record(&mut arena, 2, 0, 0, |_| Ok(()))
        .unwrap();
    assert_eq!(commands.regions.len(), 1);
    assert_eq!(commands.regions[0].used_bytes, 16);
    assert_eq!(arena.oversized_chunk_count(), 1);
    let retained = arena.small_chunk_count();
    arena.clear();
    commands.clear();
    assert_eq!(arena.oversized_chunk_count(), 0);
    commands
        .push_fixed_record(&mut arena, 3, 0, 0, |_| Ok(()))
        .unwrap();
    arena.alloc(&[9; 16]);
    assert_eq!(commands.regions[0].address, first);
    assert_eq!(arena.small_chunk_count(), retained);
}

#[test]
fn cursor_borrows_typed_payloads_at_both_eight_byte_positions() {
    use super::super::replay::CommandCursor;
    use crate::encoder_records::{IdRecord, borrow, write};

    let mut arena = ScratchArena::with_chunk_size(64);
    let mut commands = RecordedCommands::new();
    commands
        .push_fixed_record(&mut arena, 1, 0, 0, |_| Ok(()))
        .unwrap();
    let ordinary = arena.alloc(&[9; 3]);
    assert_eq!(ordinary % 16, 0);
    commands
        .push_fixed_record(&mut arena, 2, 0, size_of::<IdRecord>(), |bytes| {
            write(
                bytes,
                IdRecord {
                    id: 0x1234_5678_9abc_def0,
                },
            )
        })
        .unwrap();
    commands
        .push_fixed_record(&mut arena, 3, 0, 0, |_| Ok(()))
        .unwrap();
    commands
        .push_fixed_record(&mut arena, 4, 0, size_of::<IdRecord>(), |bytes| {
            write(bytes, IdRecord { id: 17 })
        })
        .unwrap();
    // SAFETY: commands and arena retain the table and every immutable region through iteration.
    let mut cursor = unsafe { CommandCursor::new(commands.descriptor_bytes()) }.unwrap();
    for (opcode, id, payload_modulo) in [
        (1, None, 8),
        (2, Some(0x1234_5678_9abc_def0), 0),
        (3, None, 0),
        (4, Some(17), 8),
    ] {
        // SAFETY: the authentic command table and initialized regions remain unchanged above.
        let record = unsafe { cursor.next_record() }.unwrap().unwrap();
        assert_eq!(record.opcode, opcode);
        assert_eq!(record.payload.as_ptr() as usize % 16, payload_modulo);
        if let Some(id) = id {
            assert_eq!(borrow::<IdRecord>(record.payload).unwrap().id, id);
        } else {
            assert!(record.payload.is_empty());
        }
    }
    assert!(cursor.is_complete());
    // SAFETY: the same retained table remains valid when checking its end.
    assert!(unsafe { cursor.next_record() }.unwrap().is_none());
}
