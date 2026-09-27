use mtld3d_shared::encoder_wire::{FrameSlab, WireError, WireReader};

use super::WireValue;
use crate::{
    depth_stencil_state::DepthStencilSnapshot,
    dxso::{FfStage, FfVsKey, VariantKey},
    ids::{BufferId, ProgramId, TextureId},
};

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
fn arrays_and_options_use_scalar_tags() {
    let value = [Some([1u16, 2, 3]), None, Some([4, 5, 6])];
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
    let zero_sized = [1u8, 0, 0, 0];
    assert!(matches!(
        Vec::<[u8; 0]>::read_wire(&mut WireReader::new(&zero_sized)),
        Err(WireError::InvalidValue)
    ));
}

#[test]
fn cache_identifiers_keep_their_original_identity() {
    let program = ProgramId::from_tokens(&[0xffff_0101, 0x0000_ffff]);
    assert_eq!(round_trip(&program), program);
    let texture = TextureId::new_unique();
    assert_eq!(round_trip(&texture), texture);
    let buffer = BufferId::new_unique();
    assert_eq!(round_trip(&buffer), buffer);
}

#[test]
fn shader_and_depth_snapshot_fields_round_trip() {
    let variant = VariantKey {
        alpha_func: 8,
        depth_sampler_mask: 0xaabb,
        cube_sampler_mask: 0x7788,
        ..VariantKey::default()
    };
    assert_eq!(round_trip(&variant), variant);
    let stage = FfStage {
        color_op: 4,
        alpha_arg2: 7,
        ..FfStage::default()
    };
    assert_eq!(round_trip(&stage), stage);
    let depth = DepthStencilSnapshot::inert();
    assert_eq!(round_trip(&depth), depth);
}

#[test]
fn invalid_metal_enum_and_flag_tags_are_rejected() {
    use mtld3d_shared::mtl::IndexType;

    use crate::dxso::FfStageFlags;

    assert!(matches!(
        IndexType::read_wire(&mut WireReader::new(&[255; 4])),
        Err(WireError::InvalidValue)
    ));
    assert!(matches!(
        FfStageFlags::read_wire(&mut WireReader::new(&[128])),
        Err(WireError::InvalidValue)
    ));
}

#[test]
fn truncated_fixed_function_vertex_key_never_decodes() {
    // The first field is a two-byte flags word, so even its prefix is invalid.
    assert!(matches!(
        FfVsKey::read_wire(&mut WireReader::new(&[0])),
        Err(WireError::Truncated)
    ));
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
