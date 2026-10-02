//! Shader-constant register windows, row comparison and short-window copies.
//!
//! The register windows the D3D9 constant calls accept, the bitwise comparison
//! of float rows, and the copy of short windows.

/// Whether two constant windows differ, including NaN payloads and signed zero.
///
/// Compare whole four-word rows so each row can use integer vector equality.
/// Array mapping preserves every bit without imposing SIMD alignment on callers.
#[must_use]
#[inline]
pub fn rows_differ(cur: &[[f32; 4]], new: &[[f32; 4]]) -> bool {
    cur.len() != new.len()
        || cur
            .iter()
            .zip(new)
            .any(|(a, b)| a.map(f32::to_bits) != b.map(f32::to_bits))
}

/// The longest window, in elements, that [`copy_window`] copies without a call.
///
/// A shader-constant upload is mostly one to four rows, and for a window that
/// short the call to the C `memcpy` costs more than the copy. A window up to
/// this length is copied as two blocks of constant size, which the compiler
/// expands to inline moves; a longer one goes to `copy_from_slice` and so to
/// `memcpy`.
pub const INLINE_COPY_MAX: usize = 8;

/// Copy `src` into `dst`, without a call for windows up to [`INLINE_COPY_MAX`] elements.
///
/// Every block is copied as memory, never value by value, so float rows keep
/// NaN payloads and signed zero, and neither slice needs more than its
/// element's own alignment.
///
/// # Panics
/// Panics when the two slices differ in length, as `copy_from_slice` does.
#[inline]
pub fn copy_window<T: Copy>(dst: &mut [T], src: &[T]) {
    const {
        assert!(
            INLINE_COPY_MAX == 8,
            "the block sizes below cover windows of one to eight elements"
        );
    }
    let len = dst.len();
    if len != src.len() {
        dst.copy_from_slice(src);
        return;
    }
    match len {
        1 => copy_ends::<T, 1>(dst, src),
        2..=4 => copy_ends::<T, 2>(dst, src),
        5..=INLINE_COPY_MAX => copy_ends::<T, 4>(dst, src),
        _ => dst.copy_from_slice(src),
    }
}

/// Copy `data` into `file` starting at element `start`, clamping to the file's length.
///
/// Returns whether any element changed; a window that repeats the stored
/// values is left alone, so the caller can drop the redundant write. Compares
/// by value, which for the integer and boolean register files is by bits;
/// float rows go through [`rows_differ`].
#[inline]
pub fn write_window<T: Copy + PartialEq>(file: &mut [T], start: u32, data: &[T]) -> bool {
    let start = start as usize;
    let end = start.saturating_add(data.len()).min(file.len());
    let (Some(dst), Some(src)) = (
        file.get_mut(start..end),
        data.get(..end.saturating_sub(start)),
    ) else {
        return false;
    };
    let changed = dst.iter().zip(src).any(|(cur, new)| cur != new);
    if changed {
        store_window(dst, src);
    }
    changed
}

/// [`copy_window`], out of line, for a setter to store a changed window into its mirror.
///
/// With the copy's blocks inlined after the comparison, a write that repeats
/// the bound values carried their stack spills; with the call it runs the
/// comparison alone.
///
/// # Panics
/// Panics when the two slices differ in length, as `copy_from_slice` does.
#[inline(never)]
pub fn store_window<T: Copy>(dst: &mut [T], src: &[T]) {
    copy_window(dst, src);
}

/// Copy a payload of whole 16-byte constant rows, through [`copy_window`].
///
/// A length that is not a multiple of 16 takes `copy_from_slice`.
///
/// # Panics
/// Panics when the two slices differ in length, as `copy_from_slice` does.
#[inline]
pub fn copy_row_bytes(dst: &mut [u8], src: &[u8]) {
    let (src_rows, src_rest) = src.as_chunks::<16>();
    if !src_rest.is_empty() || dst.len() != src.len() {
        dst.copy_from_slice(src);
        return;
    }
    copy_window(dst.as_chunks_mut::<16>().0, src_rows);
}

/// Copy the first and the last `K` elements of `src` into the same places of `dst`.
///
/// Covers a window of `K` to `2 * K` elements: the two blocks overlap when it
/// is shorter than `2 * K`, and each is a copy of constant size. A window
/// shorter than `K` has neither block and is left untouched.
#[inline]
fn copy_ends<T: Copy, const K: usize>(dst: &mut [T], src: &[T]) {
    if let (Some(to), Some(from)) = (dst.first_chunk_mut::<K>(), src.first_chunk::<K>()) {
        *to = *from;
    }
    if let (Some(to), Some(from)) = (dst.last_chunk_mut::<K>(), src.last_chunk::<K>()) {
        *to = *from;
    }
}

/// Whether a `[start, start + count)` constant-register window fits a `limit`-row file.
///
/// The window of the float `Set`/`Get` shader-constant calls, which D3D9
/// refuses whole rather than clamping. The sum is widened to `u64` so a start
/// near `u32::MAX` (a signed `-1`, or a `start++` probe sweep past the file)
/// cannot wrap back into range. A zero count fits when its start is at or
/// before the end of the file.
#[must_use]
#[inline]
pub fn window_in_range(start: u32, count: u32, limit: usize) -> bool {
    u64::from(start) + u64::from(count) <= limit as u64
}

/// How many registers an integer or boolean constant call moves, or `None` to refuse it.
///
/// D3D9 refuses a start at or past the `rows`-deep file whatever the count,
/// and clamps a count that runs past the file to the registers that remain. A
/// zero count moves nothing and is accepted.
#[must_use]
#[inline]
pub fn int_bool_rows(start: u32, count: u32, rows: usize) -> Option<usize> {
    let remaining = rows.checked_sub(start as usize).filter(|&n| n > 0)?;
    Some((count as usize).min(remaining))
}

#[cfg(test)]
mod tests;
