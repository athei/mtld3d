use super::{
    INLINE_COPY_MAX, copy_row_bytes, copy_window, int_bool_rows, rows_differ, window_in_range,
    write_window,
};

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

/// Bit patterns a value-by-value float copy could alter: NaN payloads and signed zero.
const COPY_PATTERNS: [u32; 10] = [
    0x7fc0_0001,
    0x7f80_0001,
    0xffc0_0002,
    0xff80_0003,
    0x8000_0000,
    0,
    0x3f80_0000,
    0x7f80_0000,
    0x0000_0001,
    0xdead_beef,
];

/// The longest window the copy tests reach: past the inline limit and past 32 rows.
const COPY_TEST_ROWS: usize = 4 * INLINE_COPY_MAX + 8;

fn pattern_words(count: usize, salt: usize) -> Vec<u32> {
    (0..u32::try_from(count).expect("test word count fits u32"))
        .map(|i| COPY_PATTERNS[(i as usize + salt) % COPY_PATTERNS.len()] ^ (i << 8))
        .collect()
}

#[test]
fn copy_window_moves_every_row_count_bit_exactly() {
    let words: Vec<f32> = pattern_words(COPY_TEST_ROWS * 4 + 8, 0)
        .into_iter()
        .map(f32::from_bits)
        .collect();
    let guard = f32::from_bits(0x7fa5_a5a5);
    let bits = |rows: &[[f32; 4]]| -> Vec<u32> {
        rows.as_flattened().iter().map(|v| v.to_bits()).collect()
    };
    for offset in 0..4 {
        for len in 0..=COPY_TEST_ROWS {
            let src = words[offset..offset + len * 4].as_chunks::<4>().0;
            let mut dst = vec![[guard; 4]; len + 2];
            copy_window(&mut dst[1..=len], src);
            assert_eq!(
                bits(&dst[1..=len]),
                bits(src),
                "offset {offset}, {len} rows"
            );
            assert_eq!(bits(&dst[..1]), [guard.to_bits(); 4]);
            assert_eq!(bits(&dst[len + 1..]), [guard.to_bits(); 4]);
        }
    }
}

#[test]
fn copy_window_keeps_nan_payloads_and_signed_zero_in_every_row() {
    let specials = [
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xffc0_0002),
        -0.0,
    ];
    for len in 1..=COPY_TEST_ROWS {
        for row in 0..len {
            let mut src = vec![[0.0_f32; 4]; len];
            src[row] = specials;
            let mut dst = vec![[1.0_f32; 4]; len];
            copy_window(&mut dst, &src);
            assert!(
                !rows_differ(&dst, &src),
                "{len} rows, the special values in row {row}"
            );
        }
    }
}

#[test]
fn copy_window_moves_every_count_of_single_words() {
    let words: Vec<i32> = pattern_words(COPY_TEST_ROWS + 2, 3)
        .into_iter()
        .map(u32::cast_signed)
        .collect();
    for offset in 0..2 {
        for len in 0..=COPY_TEST_ROWS {
            let src = &words[offset..offset + len];
            let mut dst = vec![-1_i32; len + 2];
            copy_window(&mut dst[1..=len], src);
            assert_eq!(&dst[1..=len], src, "offset {offset}, {len} words");
            assert_eq!((dst[0], dst[len + 1]), (-1, -1));
        }
    }
}

#[test]
fn copy_row_bytes_takes_sources_and_destinations_at_every_byte_offset() {
    let bytes: Vec<u8> = pattern_words(COPY_TEST_ROWS * 4 + 8, 5)
        .into_iter()
        .flat_map(u32::to_ne_bytes)
        .collect();
    for src_offset in 0..16 {
        for dst_offset in [0, 1, 3, 8, 15] {
            for rows in 0..=COPY_TEST_ROWS {
                let len = rows * 16;
                let src = &bytes[src_offset..src_offset + len];
                let mut dst = vec![0xa5_u8; dst_offset + len + 16];
                copy_row_bytes(&mut dst[dst_offset..dst_offset + len], src);
                assert_eq!(
                    &dst[dst_offset..dst_offset + len],
                    src,
                    "source offset {src_offset}, destination offset {dst_offset}, {rows} rows"
                );
                assert!(dst[..dst_offset].iter().all(|&b| b == 0xa5));
                assert!(dst[dst_offset + len..].iter().all(|&b| b == 0xa5));
            }
        }
    }
}

#[test]
fn copy_row_bytes_copies_a_length_that_is_not_whole_rows() {
    let src: Vec<u8> = (0..=44).collect();
    for len in [1, 15, 17, 31, 33, 45] {
        let mut dst = vec![0xa5_u8; len + 1];
        copy_row_bytes(&mut dst[..len], &src[..len]);
        assert_eq!(&dst[..len], &src[..len]);
        assert_eq!(dst[len], 0xa5);
    }
}

#[test]
#[should_panic(expected = "source slice length")]
fn copy_window_rejects_unequal_lengths() {
    let mut dst = [[0.0_f32; 4]; 2];
    copy_window(&mut dst, &[[1.0; 4]; 3]);
}

#[test]
#[should_panic(expected = "source slice length")]
fn copy_window_rejects_unequal_lengths_past_the_inline_limit() {
    let mut dst = [[0.0_f32; 4]; 12];
    copy_window(&mut dst, &[[1.0; 4]; 20]);
}

#[test]
#[should_panic(expected = "source slice length")]
fn copy_row_bytes_rejects_unequal_lengths() {
    let mut dst = [0_u8; 32];
    copy_row_bytes(&mut dst, &[0; 16]);
}

#[test]
fn write_window_reports_a_change_only_when_a_stored_value_differs() {
    let mut rows = [[0_i32; 4]; 16];
    assert!(write_window(&mut rows, 3, &[[1, 2, 3, 4], [5, 6, 7, 8]]));
    assert_eq!((rows[3], rows[4]), ([1, 2, 3, 4], [5, 6, 7, 8]));
    assert_eq!((rows[2], rows[5]), ([0; 4], [0; 4]));
    let before = rows;
    assert!(!write_window(&mut rows, 3, &[[1, 2, 3, 4], [5, 6, 7, 8]]));
    assert!(!write_window(&mut rows, 4, &[[5, 6, 7, 8]]));
    assert_eq!(rows, before);
    assert!(write_window(&mut rows, 3, &[[1, 2, 3, 4], [5, 6, 7, 9]]));
    assert_eq!(rows[4], [5, 6, 7, 9]);

    let mut bools = [0_i32; 16];
    for len in 1..=16 {
        let ones = vec![1_i32; len];
        let zeros = vec![0_i32; len];
        for changed in 0..len {
            let mut data = zeros.clone();
            data[changed] = 1;
            assert!(
                write_window(&mut bools, 0, &data),
                "{len} values, {changed} set"
            );
            assert_eq!(&bools[..len], &data[..]);
            assert!(
                !write_window(&mut bools, 0, &data),
                "{len} values, {changed} repeated"
            );
            assert!(write_window(&mut bools, 0, &zeros));
        }
        assert!(write_window(&mut bools, 0, &ones));
        assert!(!write_window(&mut bools, 0, &ones));
        assert!(write_window(&mut bools, 0, &zeros));
    }
}

#[test]
fn write_window_clamps_to_the_file_and_ignores_a_window_past_it() {
    let mut file = [0_i32; 4];
    assert!(write_window(&mut file, 2, &[7, 8, 9, 10]));
    assert_eq!(file, [0, 0, 7, 8]);
    assert!(!write_window(&mut file, 2, &[7, 8, 11, 12]));
    assert!(!write_window(&mut file, 4, &[1]));
    assert!(!write_window(&mut file, u32::MAX, &[1, 2]));
    assert!(!write_window(&mut file, 1, &[]));
    assert_eq!(file, [0, 0, 7, 8]);
}

#[test]
fn float_windows_fit_up_to_the_end_of_the_file() {
    assert!(window_in_range(0, 256, 256));
    assert!(window_in_range(255, 1, 256));
    assert!(window_in_range(256, 0, 256));
    assert!(window_in_range(223, 1, 224));
    assert!(!window_in_range(255, 2, 256));
    assert!(!window_in_range(256, 1, 256));
    assert!(!window_in_range(257, 0, 256));
    assert!(!window_in_range(224, 1, 224));
    assert!(!window_in_range(u32::MAX, 1, 256));
    assert!(!window_in_range(1, u32::MAX, 256));
    assert!(!window_in_range(u32::MAX, u32::MAX, 256));
}

#[test]
fn int_bool_rows_refuse_a_start_past_the_file_and_clamp_the_count() {
    assert_eq!(int_bool_rows(0, 0, 16), Some(0));
    assert_eq!(int_bool_rows(0, 16, 16), Some(16));
    assert_eq!(int_bool_rows(0, 17, 16), Some(16));
    assert_eq!(int_bool_rows(15, 0, 16), Some(0));
    assert_eq!(int_bool_rows(15, 1, 16), Some(1));
    assert_eq!(int_bool_rows(15, u32::MAX, 16), Some(1));
    assert_eq!(int_bool_rows(16, 0, 16), None);
    assert_eq!(int_bool_rows(16, 1, 16), None);
    assert_eq!(int_bool_rows(u32::MAX, 1, 16), None);
}
