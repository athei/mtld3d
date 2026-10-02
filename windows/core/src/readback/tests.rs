use mtld3d_types::{D3DFMT_A8B8G8R8, D3DFMT_A16B16G16R16F, D3DFMT_R8G8B8, D3DFMT_X1R5G5B5};

use super::*;

fn source() -> ReadbackSource {
    ReadbackSource {
        width: 64,
        height: 32,
        format: D3DFMT_X8R8G8B8,
    }
}

fn destination() -> ReadbackDestination {
    ReadbackDestination {
        width: 64,
        height: 32,
        format: D3DFMT_X8R8G8B8,
        bytes_per_row: 256,
        len: 8192,
    }
}

#[test]
fn a_matching_destination_is_accepted() {
    assert_eq!(reject_readback_dst(&source(), &destination()), None);
}

#[test]
fn a_destination_larger_than_its_rows_is_accepted() {
    // A page-rounded backing is the norm: the surface allocates whole pages
    // and the copy fills the first `height * bytes_per_row` of them.
    let dst = ReadbackDestination {
        len: 16384,
        ..destination()
    };
    assert_eq!(reject_readback_dst(&source(), &dst), None);
}

#[test]
fn an_empty_source_is_rejected() {
    for src in [
        ReadbackSource {
            width: 0,
            ..source()
        },
        ReadbackSource {
            height: 0,
            ..source()
        },
    ] {
        assert_eq!(
            reject_readback_dst(&src, &destination()),
            Some(ReadbackReject::EmptySource)
        );
    }
}

#[test]
fn a_differently_sized_destination_is_rejected() {
    for dst in [
        ReadbackDestination {
            width: 32,
            ..destination()
        },
        ReadbackDestination {
            height: 64,
            ..destination()
        },
    ] {
        assert_eq!(
            reject_readback_dst(&source(), &dst),
            Some(ReadbackReject::ExtentMismatch)
        );
    }
}

#[test]
fn a_destination_of_another_format_is_rejected() {
    // R8G8B8 shares the source's BGRA8 storage but not its row layout; the
    // others differ in storage too.
    for format in [
        D3DFMT_R8G8B8,
        D3DFMT_A16B16G16R16F,
        D3DFMT_A8B8G8R8,
        D3DFMT_X1R5G5B5,
    ] {
        let dst = ReadbackDestination {
            format,
            ..destination()
        };
        assert_eq!(
            reject_readback_dst(&source(), &dst),
            Some(ReadbackReject::FormatMismatch),
            "{format:#x}"
        );
    }
}

#[test]
fn the_x8_a8_pair_matches_either_way_round() {
    assert!(formats_match(D3DFMT_X8R8G8B8, D3DFMT_A8R8G8B8));
    assert!(formats_match(D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8));
    assert!(formats_match(D3DFMT_R8G8B8, D3DFMT_R8G8B8));
    assert!(!formats_match(D3DFMT_X8R8G8B8, D3DFMT_R8G8B8));
    assert!(!formats_match(D3DFMT_A8B8G8R8, D3DFMT_A8R8G8B8));
    let dst = ReadbackDestination {
        format: D3DFMT_A8R8G8B8,
        ..destination()
    };
    assert_eq!(reject_readback_dst(&source(), &dst), None);
}

#[test]
fn a_destination_shorter_than_the_copy_is_rejected() {
    let dst = ReadbackDestination {
        len: 8191,
        ..destination()
    };
    assert_eq!(
        reject_readback_dst(&source(), &dst),
        Some(ReadbackReject::DestinationTooSmall)
    );
}

#[test]
fn a_destination_with_no_row_stride_is_rejected() {
    let dst = ReadbackDestination {
        bytes_per_row: 0,
        ..destination()
    };
    assert_eq!(
        reject_readback_dst(&source(), &dst),
        Some(ReadbackReject::DestinationTooSmall)
    );
}

#[test]
fn every_reason_has_its_own_key_and_text() {
    let reasons = [
        ReadbackReject::NotSystemMemory,
        ReadbackReject::EmptySource,
        ReadbackReject::ExtentMismatch,
        ReadbackReject::FormatMismatch,
        ReadbackReject::DestinationTooSmall,
    ];
    for (i, a) in reasons.iter().enumerate() {
        for b in &reasons[i + 1..] {
            assert_ne!(a.key(), b.key(), "{a:?} and {b:?} share a log key");
            assert_ne!(a.as_str(), b.as_str(), "{a:?} and {b:?} share a message");
        }
    }
}
