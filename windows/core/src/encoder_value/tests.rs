use mtld3d_shared::encoder_wire::{FrameSlab, WireError, WireReader};

use super::WireValue;

fn round_trip<T: WireValue>(value: &T) -> T {
    let mut slab = FrameSlab::new();
    slab.push_record(1, |writer| value.write_wire(writer))
        .unwrap();
    let mut stream = WireReader::new(slab.as_bytes());
    let mut record = stream.next_record().unwrap().unwrap();
    let decoded = T::read_wire(&mut record.payload).unwrap();
    assert!(record.payload.is_empty());
    decoded
}

#[test]
fn options_use_a_bool_tag() {
    let value = vec![Some(1u32), None, Some(3)];
    assert_eq!(round_trip(&value), value);
    let invalid = [2u8];
    assert!(matches!(
        Option::<u32>::read_wire(&mut WireReader::new(&invalid)),
        Err(WireError::InvalidValue)
    ));
    assert!(matches!(
        bool::read_wire(&mut WireReader::new(&invalid)),
        Err(WireError::InvalidValue)
    ));
}

#[test]
fn collection_length_is_checked_before_allocating() {
    let truncated = [255u8; 4];
    assert!(matches!(
        Vec::<u64>::read_wire(&mut WireReader::new(&truncated)),
        Err(WireError::Truncated)
    ));
    let value = vec![11u32, 22, 33];
    assert_eq!(round_trip(&value), value);
}

#[test]
fn utf8_string_round_trip_and_invalid_bytes() {
    let value = String::from("shader/水");
    assert_eq!(round_trip(&value), value);
    let malformed = [1, 0, 0, 0, 255];
    assert!(matches!(
        String::read_wire(&mut WireReader::new(&malformed)),
        Err(WireError::InvalidValue)
    ));
}
