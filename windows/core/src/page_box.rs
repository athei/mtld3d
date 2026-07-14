//! Page-aligned, page-sized heap backing for dynamic VB/IB data.
//!
//! Metal's `newBufferWithBytesNoCopy:length:options:deallocator:` requires
//! both the backing pointer and the length to be page-aligned. 16 KiB
//! covers both Apple Silicon (16 KiB pages) and x86 macOS (4 KiB pages,
//! so a 16 KiB multiple is also 4 KiB-aligned).
//!
//! `logical_len` is the unrounded length the game sees through Lock; the
//! raw `len` is the rounded-up page multiple that Metal sees. Everything
//! past `logical_len` is padding — game writes never reach it, GPU reads
//! stay within the vertex/index stride × count the draw specifies.

use std::{
    alloc::{self, Layout},
    ptr::NonNull,
};

/// Apple Silicon page size.
///
/// Metal's `newBufferWithBytesNoCopy` demands the backing be page-aligned +
/// page-sized; 16 KiB satisfies both `ASi` and x86 macOS.
pub const PAGE_SIZE: usize = 16 * 1024;

/// RAII wrapper around a `std::alloc::alloc`-ed page-aligned byte region.
///
/// Sized to the next page multiple of `logical_len`; both the raw pointer
/// and the reported length are safe to hand to Metal via
/// `newBufferWithBytesNoCopy:`.
pub struct PageBox {
    ptr: NonNull<u8>,
    /// Rounded-up page multiple.
    ///
    /// What `len()` reports and what the `MTLBuffer` wraps.
    len: usize,
    /// Original request from the caller. Game's visible buffer length.
    logical_len: usize,
    /// Layout used for `alloc` — stored so `Drop` can match `dealloc`.
    layout: Layout,
}

// SAFETY: PageBox owns a heap allocation with no interior sharing. Transferring
// ownership across threads is sound — callers must externally synchronize to
// prevent simultaneous reads/writes, the same contract as `Box<[u8]>`.
unsafe impl Send for PageBox {}
// SAFETY: same as Send above — `&PageBox` only exposes the raw pointer, so
// `Sync` requires the same caller-synchronized contract.
unsafe impl Sync for PageBox {}

impl PageBox {
    /// Allocate `logical_len` bytes rounded up to the next page multiple.
    ///
    /// Contents are uninitialized. For any caller that returns the
    /// pointer to the game, the game is expected to write every byte it
    /// later reads.
    #[must_use]
    pub fn new_uninit(logical_len: usize) -> Self {
        let (len, layout) = Self::layout_for(logical_len);
        // SAFETY: `layout_for` returns a non-zero-size, page-aligned Layout.
        let ptr = unsafe { alloc::alloc(layout) };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self {
            ptr,
            len,
            logical_len,
            layout,
        }
    }

    /// Fallible counterpart of `new_uninit`.
    ///
    /// Returns `None` when the underlying allocator returns null instead of
    /// panicking via `handle_alloc_error`. Used by VB/IB Lock-rename so the
    /// caller can attempt mid-frame retention drain + retry before giving
    /// up. Non-rename call sites (`CreateVertexBuffer` / `IndexBuffer` at
    /// device-construction time) keep the panicking `new_uninit` because
    /// there's no recovery available there.
    #[must_use]
    pub fn try_new_uninit(logical_len: usize) -> Option<Self> {
        let (len, layout) = Self::layout_for(logical_len);
        // SAFETY: `layout_for` returns a non-zero-size, page-aligned Layout.
        let ptr = unsafe { alloc::alloc(layout) };
        NonNull::new(ptr).map(|ptr| Self {
            ptr,
            len,
            logical_len,
            layout,
        })
    }

    /// Same as `new_uninit` but zero-initializes the full padded region.
    ///
    /// Used by VB/IB creation so a first-draw-before-Lock sees defined
    /// bytes. Costs one `bzero` per buffer create — negligible compared
    /// to the alternative of every rename paying for the same zero init.
    #[must_use]
    pub fn new_zeroed(logical_len: usize) -> Self {
        let (len, layout) = Self::layout_for(logical_len);
        // SAFETY: `layout_for` returns a non-zero-size, page-aligned Layout.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self {
            ptr,
            len,
            logical_len,
            layout,
        }
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub const fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Borrow the full padded region as a slice.
    ///
    /// Matches `Box<[u8]>`'s deref semantic: callers that consume `&[u8]`
    /// see exactly the same byte range.
    #[must_use]
    pub const fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is non-null, valid for `len` bytes (we allocated
        // it that way), and the lifetime is tied to `&self`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutable counterpart of `as_slice`.
    ///
    /// Caller takes the unique-borrow guarantee from `&mut self`.
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is non-null, valid for `len` bytes, and the
        // unique borrow on `&mut self` keeps the slice exclusive.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Padded length (page multiple).
    ///
    /// This is what the Metal `MTLBuffer` wrapper sees.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Original `logical_len` the caller requested.
    ///
    /// What the game sees through Lock.
    #[must_use]
    pub const fn logical_len(&self) -> usize {
        self.logical_len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.logical_len == 0
    }

    fn layout_for(logical_len: usize) -> (usize, Layout) {
        // A zero-length PageBox is still allocated at one page so the
        // returned pointer is non-null and the MTLBuffer wrap doesn't
        // choke on length=0.
        let padded = logical_len
            .max(1)
            .div_ceil(PAGE_SIZE)
            .checked_mul(PAGE_SIZE)
            .expect("PageBox length overflow");
        let layout = Layout::from_size_align(padded, PAGE_SIZE).expect("valid page-aligned layout");
        (padded, layout)
    }
}

impl Drop for PageBox {
    fn drop(&mut self) {
        // SAFETY: same layout used to alloc; pointer came from that
        // allocator; nothing else owns this allocation.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::{PAGE_SIZE, PageBox};

    #[test]
    fn uninit_alloc_is_page_aligned_and_page_sized() {
        let pb = PageBox::new_uninit(1);
        assert_eq!(pb.as_ptr() as usize % PAGE_SIZE, 0);
        assert_eq!(pb.len(), PAGE_SIZE);
        assert_eq!(pb.logical_len(), 1);
    }

    #[test]
    fn length_rounds_up_to_page_multiple() {
        let pb = PageBox::new_uninit(PAGE_SIZE + 1);
        assert_eq!(pb.len(), 2 * PAGE_SIZE);
        assert_eq!(pb.logical_len(), PAGE_SIZE + 1);
    }

    #[test]
    fn exact_page_multiple_is_not_rounded_further() {
        let pb = PageBox::new_uninit(3 * PAGE_SIZE);
        assert_eq!(pb.len(), 3 * PAGE_SIZE);
    }

    #[test]
    fn zero_length_still_allocates_one_page() {
        let pb = PageBox::new_uninit(0);
        assert_eq!(pb.len(), PAGE_SIZE);
        assert_eq!(pb.logical_len(), 0);
    }

    #[test]
    fn writable_across_full_logical_len() {
        let logical = 5000usize;
        let mut pb = PageBox::new_uninit(logical);
        let ptr = pb.as_mut_ptr();
        for i in 0..logical {
            let byte = u8::try_from(i & 0xff).expect("masked to 0xFF fits u8");
            // SAFETY: `ptr + i` stays within the just-allocated `logical`-byte slab.
            let dst = unsafe { ptr.add(i) };
            // SAFETY: same slab; `u8` writes are always aligned.
            unsafe { dst.write(byte) };
        }
        let rp = pb.as_ptr();
        for i in 0..logical {
            // SAFETY: `rp + i` stays within the just-written `logical`-byte slab.
            let src = unsafe { rp.add(i) };
            // SAFETY: same slab; `u8` reads are always aligned.
            let v = unsafe { src.read() };
            let expected = u8::try_from(i & 0xff).expect("masked to 0xFF fits u8");
            assert_eq!(v, expected);
        }
    }

    #[test]
    fn multiple_allocs_do_not_alias() {
        let a = PageBox::new_uninit(PAGE_SIZE);
        let b = PageBox::new_uninit(PAGE_SIZE);
        assert_ne!(a.as_ptr(), b.as_ptr());
    }

    #[test]
    fn zeroed_init_is_actually_zero() {
        let pb = PageBox::new_zeroed(100);
        let p = pb.as_ptr();
        for i in 0..100 {
            // SAFETY: `p + i` stays within the just-allocated 100-byte zeroed slab.
            let byte_ptr = unsafe { p.add(i) };
            // SAFETY: same slab; `byte_ptr` is well-aligned for `u8`.
            let byte = unsafe { byte_ptr.read() };
            assert_eq!(byte, 0);
        }
    }

    #[test]
    fn drop_does_not_panic() {
        for _ in 0..16 {
            drop(PageBox::new_uninit(64 * 1024));
            drop(PageBox::new_zeroed(8 * 1024));
        }
    }
}
