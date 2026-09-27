use mtld3d_shared::encoder_wire::{FrameSlab, LeaseCompletion};

use super::*;
use crate::{encoder_packet, guest_queries::QueryLeaseCache, ids::ProgramId, page_box::PageBox};

#[test]
fn release_footer_is_accepted_by_debug_decoder_without_dropping_owners() {
    let (mut frame, weak_query) = encoder_packet::tests::frame_with_leases();
    frame.flags.insert(FrameDataFlags::NO_PRESENT);
    frame.vbib_retentions.push(PendingVbibRetention {
        buffer_id: crate::ids::BufferId::new_unique(),
        page_box: PageBox::new_zeroed(4),
        last_submit_seq: 1,
    });
    let mut recorder = FrameRecorder::new();
    for op in core::mem::take(&mut frame.ops) {
        recorder.try_record(&mut frame.scratch, op).unwrap();
    }
    let mut metadata = FrameSlab::new();
    metadata
        .push_record(encoder_packet::FRAME_METADATA_TAG, |writer| {
            write_metadata_with_inventory(&mut frame, writer, &mut recorder, false)
        })
        .unwrap();
    assert_eq!(recorder.pages.len(), 3);
    assert_eq!(recorder.queries.len(), 1);
    assert_eq!(recorder.registrations, [41]);
    assert!(weak_query.upgrade().is_some());
    // SAFETY: frame and recorder retain authentic immutable producer allocations.
    let parsed = unsafe { encoder_packet::parse_metadata(metadata.as_bytes()) }.unwrap();
    assert!(!parsed.has_inventory);
    assert_eq!(parsed.frame.flags, FrameDataFlags::NO_PRESENT);
    assert!(parsed.inventory.ranges.is_empty());
    assert!(parsed.inventory.pages.is_empty());
    assert!(parsed.inventory.queries.is_empty());
    assert!(parsed.inventory.redirties.is_empty());
    assert!(parsed.inventory.replies_u64.is_empty());
    assert!(parsed.inventory.replies_bool.is_empty());
    assert!(parsed.inventory.readbacks.is_empty());
    assert!(parsed.inventory.command_spans.is_empty());
    assert!(parsed.inventory.backings.is_empty());
    assert_eq!(parsed.inventory.registrations, [41]);
    assert_eq!(parsed.retained.len(), 1);
    drop(parsed);

    let completion = LeaseCompletion::new();
    let mut queries = QueryLeaseCache::default();
    let tokens = [
        0xFFFF_0200,
        0x0200_0001,
        0x800F_0800,
        0xA0E4_0000,
        0x0000_FFFF,
    ];
    let mut resolutions = 0;
    // SAFETY: the typed recorder created every command and unique descriptor;
    // frame, recorder, metadata and completion outlive all native consumers.
    let decoded = unsafe {
        encoder_packet::decode_packet(
            metadata.as_bytes(),
            recorder.slab.descriptor_bytes(),
            std::ptr::from_ref(&completion) as u64,
            &mut queries,
            |registration| {
                assert_eq!(registration, 41);
                resolutions += 1;
                Ok((
                    ProgramId::from_tokens(&tokens),
                    crate::dxso::parse(&tokens).unwrap(),
                ))
            },
        )
    }
    .unwrap();
    assert_eq!(resolutions, 1);
    assert_eq!(decoded.ops.len(), 4);
    assert_eq!(decoded.vbib_retentions.len(), 1);
    assert!(!completion.is_complete());
    drop(decoded);
    assert!(completion.is_complete());
    drop(queries);
    drop(recorder);
    assert!(weak_query.upgrade().is_none());
}

#[test]
fn release_reader_consumes_debug_footer_and_keeps_only_registrations() {
    let backing = [0u8; 16];
    let mut slab = FrameSlab::new();
    slab.push_record(1, |writer| {
        writer.u32(1)?;
        writer.u64(backing.as_ptr() as u64)?;
        writer.u64(backing.len() as u64)?;
        vec![[1_u64; 7]].write_wire(writer)?;
        vec![[2_u64; 2]].write_wire(writer)?;
        vec![41_u64, 42].write_wire(writer)?;
        vec![[3_u64; 2]].write_wire(writer)?;
        vec![4_u64].write_wire(writer)?;
        vec![5_u64].write_wire(writer)?;
        vec![[6_u64; 2]].write_wire(writer)?;
        vec![[7_u64; 2]].write_wire(writer)?;
        vec![[8_u64; 2]].write_wire(writer)
    })
    .unwrap();
    let payload = &slab.as_bytes()[6..];
    // SAFETY: the only byte range names backing, retained unchanged through this test.
    let mut reader = unsafe { WireReader::new_trusted(payload) };
    assert_eq!(read_inventory_registrations(&mut reader).unwrap(), [41, 42]);
    assert!(reader.is_empty());
    for length in 0..payload.len() {
        // SAFETY: any complete range in this prefix names the same retained backing.
        let mut truncated = unsafe { WireReader::new_trusted(&payload[..length]) };
        assert!(read_inventory_registrations(&mut truncated).is_err());
    }
}
