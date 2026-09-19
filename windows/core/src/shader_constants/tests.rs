use super::rows_differ;

fn scalar(cur: &[[f32; 4]], new: &[[f32; 4]]) -> bool {
    cur.len() != new.len()
        || cur
            .as_flattened()
            .iter()
            .zip(new.as_flattened())
            .any(|(a, b)| a.to_bits() != b.to_bits())
}

#[test]
fn exact_bits_at_every_lane_and_register_boundary() {
    let patterns = [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0001,
        0x7fc0_0002,
        0x7f80_0001,
        0xffc0_0001,
    ];
    let words: Vec<_> = (0..1030)
        .map(|i| f32::from_bits(patterns[i % patterns.len()]))
        .collect();
    for offset in 0..4 {
        for len in 0..=256 {
            let rows = words[offset..offset + len * 4].as_chunks::<4>().0;
            let mut copy = rows.to_vec();
            assert!(!rows_differ(rows, &copy));
            for lane in 0..len * 4 {
                let old = copy[lane / 4][lane % 4].to_bits();
                copy[lane / 4][lane % 4] = f32::from_bits(old ^ 0x8000_0000);
                assert_eq!(rows_differ(rows, &copy), scalar(rows, &copy));
                copy[lane / 4][lane % 4] = f32::from_bits(old);
            }
        }
    }
}

#[test]
fn nan_payload_and_zero_sign_are_changes() {
    let a = [[f32::from_bits(0x7fc0_0001), 0.0, f32::INFINITY, 1.0]];
    let mut b = a;
    assert!(!rows_differ(&a, &b));
    b[0][0] = f32::from_bits(0x7fc0_0002);
    assert!(rows_differ(&a, &b));
    b = a;
    b[0][1] = -0.0;
    assert!(rows_differ(&a, &b));
}

#[test]
fn unequal_lengths_are_changes_even_with_equal_prefixes() {
    let rows = [[0.0; 4]; 2];
    assert!(rows_differ(&rows, &rows[..1]));
    assert!(rows_differ(&rows[..1], &rows));
    assert!(rows_differ(&[], &rows));
    assert!(!rows_differ(&[], &[]));
}
