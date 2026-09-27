use super::*;

#[test]
fn failed_rollover_keeps_committed_region_and_reuses_next_reservation() {
    let mut arena = ScratchArena::with_chunk_size(64);
    let mut commands = RecordedCommands::new();
    commands
        .push_fixed_record(&mut arena, 1, 0, 48, |bytes| {
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
    assert_eq!(commands.regions[1].used_bytes, 32);
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
    assert_eq!(commands.regions[0].used_bytes, 32);
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
