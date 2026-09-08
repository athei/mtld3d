use super::{
    LC_UUID, MACH_HEADER_64_SIZE, MACH_MAGIC_64, NCMDS, SIZEOFCMDS, format_uuid, mach_uuid,
};

const UUID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

#[test]
fn reads_uuid_after_an_unrelated_command() {
    let mut commands = command(1, 8);
    commands.extend(uuid_command());
    assert_eq!(mach_uuid(&image(2, &commands)), Some(UUID));
    assert_eq!(format_uuid(&UUID), "00010203-0405-0607-0809-0A0B0C0D0E0F");
}

#[test]
fn missing_uuid_is_absent() {
    assert_eq!(mach_uuid(&image(0, &[])), None);
    assert_eq!(mach_uuid(&image(1, &command(1, 8))), None);
}

#[test]
fn rejects_every_truncation_of_a_uuid_image() {
    let bytes = image(1, &uuid_command());
    for end in 0..bytes.len() {
        assert_eq!(mach_uuid(&bytes[..end]), None, "length {end}");
    }
}

#[test]
fn rejects_wrong_magic_and_command_extent() {
    let mut bytes = image(1, &uuid_command());
    set_word(&mut bytes, 0, 0xfeed_face);
    assert_eq!(mach_uuid(&bytes), None);
    set_word(&mut bytes, 0, MACH_MAGIC_64);
    set_word(&mut bytes, SIZEOFCMDS, u32::MAX);
    assert_eq!(mach_uuid(&bytes), None);
    set_word(&mut bytes, SIZEOFCMDS, 8);
    assert_eq!(mach_uuid(&bytes), None);
}

#[test]
fn rejects_inconsistent_command_counts() {
    let mut bytes = image(0, &uuid_command());
    assert_eq!(mach_uuid(&bytes), None);
    set_word(&mut bytes, NCMDS, 2);
    assert_eq!(mach_uuid(&bytes), None);
    set_word(&mut bytes, NCMDS, u32::MAX);
    assert_eq!(mach_uuid(&bytes), None);
}

#[test]
fn rejects_short_unaligned_and_oversized_commands() {
    for size in [0, 4, 7, 9, 16, 23, 25, 32, u32::MAX] {
        let mut bytes = image(1, &uuid_command());
        set_word(&mut bytes, MACH_HEADER_64_SIZE + 4, size);
        assert_eq!(mach_uuid(&bytes), None, "command size {size}");
    }
}

#[test]
fn cannot_read_uuid_bytes_outside_the_command() {
    let mut commands = command(LC_UUID, 8);
    commands.extend(command(1, 8));
    commands.extend(command(1, 8));
    assert_eq!(mach_uuid(&image(3, &commands)), None);
}

#[test]
fn validates_commands_after_the_uuid() {
    let mut commands = uuid_command();
    commands.extend(command(1, 0));
    assert_eq!(mach_uuid(&image(2, &commands)), None);
}

fn image(count: u32, commands: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0; MACH_HEADER_64_SIZE];
    set_word(&mut bytes, 0, MACH_MAGIC_64);
    set_word(&mut bytes, NCMDS, count);
    set_word(
        &mut bytes,
        SIZEOFCMDS,
        u32::try_from(commands.len()).unwrap(),
    );
    bytes.extend(commands);
    bytes
}

fn command(kind: u32, size: u32) -> Vec<u8> {
    let mut bytes = kind.to_ne_bytes().to_vec();
    bytes.extend(size.to_ne_bytes());
    bytes
}

fn uuid_command() -> Vec<u8> {
    let mut bytes = command(LC_UUID, 24);
    bytes.extend(UUID);
    bytes
}

fn set_word(bytes: &mut [u8], offset: usize, word: u32) {
    bytes[offset..offset + 4].copy_from_slice(&word.to_ne_bytes());
}
