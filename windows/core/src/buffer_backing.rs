//! CPU backing of a VB/IB, with the live-bytes gauge and the post-upload release.
//!
//! A `Staged` buffer (see [`crate::buffer_rename::classify_map_mode`]) owns two
//! allocations: a `Private` device buffer the GPU reads, and this `PageBox`,
//! which is pure CPU staging the application writes through `Lock`. The staging
//! is what `Unlock` snapshots the dirtied range out of; the GPU never touches
//! it.
//!
//! For one class of buffer the staging is dead weight the moment its upload is
//! queued: `D3DPOOL_DEFAULT` plus `D3DUSAGE_WRITEONLY` and no
//! `D3DUSAGE_DYNAMIC`, the static geometry a title fills once at load. D3D9
//! promises that class no readback, and inside a large-address-aware i386
//! process every byte of it sits in the same 4 GiB the title needs for its own
//! world data. [`may_release_backing`] names the class, and [`BufferBacking`]
//! tracks what releasing it costs.
//!
//! Releasing loses the copy the whole buffer's contents can be rebuilt from, so
//! the state a backing is in decides what a later upload may carry:
//!
//! - [`BackingState::Mirrors`]: the backing holds every byte the device buffer
//!   holds. Any upload out of it is correct, including a whole-buffer one, and
//!   this is the only state a release may happen in.
//! - [`BackingState::Released`]: no backing at all; the device buffer holds the
//!   only copy.
//! - [`BackingState::Partial`]: re-created after a release, so only the ranges
//!   written since are real bytes and the rest is zero. An upload wider than
//!   what the application announced would push those zeros over device bytes
//!   nobody rewrote, so the caller must not widen one here.
//!
//! [`BufferBacking::note_upload`] returns a `Partial` backing to `Mirrors` once
//! an upload covers the whole buffer, because the device buffer is then a copy
//! of the backing whatever the backing held.
//!
//! One path reads a released buffer's bytes back anyway: the indexed
//! triangle-fan rewrite needs the application's indices, and Metal has no fan
//! primitive to hand them to. [`BufferBacking::adopt_device_copy`] takes the
//! copy that path reads off the GPU and makes it the backing again, in
//! `Mirrors` because it came from the device buffer itself. That read costs a
//! GPU stall, so the backing it installs is pinned and no later upload
//! releases it: a buffer pays the stall once however many fans it draws.
//!
//! The gauge is three process-wide counters keyed by [`BackingClass`], one add
//! per allocation and one subtract per release. The address-space watch reports
//! them split, so a 32-bit title's log says which class of buffer holds the
//! space rather than leaving it to a guess.

use core::sync::atomic::{AtomicU64, Ordering};

use mtld3d_types::{
    D3DPOOL_DEFAULT, D3DUSAGE_DYNAMIC, D3DUSAGE_SOFTWAREPROCESSING, D3DUSAGE_WRITEONLY,
};

use crate::page_box::PageBox;

/// Live backing bytes of `D3DPOOL_DEFAULT` `D3DUSAGE_WRITEONLY` static buffers.
static WRITE_ONLY_STATIC_BYTES: AtomicU64 = AtomicU64::new(0);
/// Live backing bytes of `D3DUSAGE_DYNAMIC` buffers.
static DYNAMIC_BYTES: AtomicU64 = AtomicU64::new(0);
/// Live backing bytes of every buffer neither of the other two counters claims.
static OTHER_BYTES: AtomicU64 = AtomicU64::new(0);

/// Which gauge counter a buffer's CPU backing is charged to.
///
/// The split is the one the release decision turns on, so
/// `WriteOnlyStatic` reads as the bytes the release can still reclaim and
/// the other two read as the bytes it structurally cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackingClass {
    /// `D3DPOOL_DEFAULT`, `D3DUSAGE_WRITEONLY`, neither dynamic nor software-processed.
    WriteOnlyStatic,
    /// `D3DUSAGE_DYNAMIC`, in any pool: the per-frame batcher shape.
    Dynamic,
    /// Everything else: the lockable pools, and default-pool buffers a title may read.
    Other,
}

/// How much of the device buffer's contents the CPU backing still holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackingState {
    /// The backing holds every byte the device buffer holds.
    Mirrors,
    /// The backing is gone; the device buffer holds the only copy.
    Released,
    /// Re-created after a release: only the ranges written since it came back are real.
    Partial,
}

/// Live CPU-backing bytes per [`BackingClass`], as the gauge reports them.
pub struct BackingBytes {
    /// Bytes held by buffers a release can reclaim.
    pub write_only_static: u64,
    /// Bytes held by `D3DUSAGE_DYNAMIC` buffers.
    pub dynamic: u64,
    /// Bytes held by every other buffer.
    pub other: u64,
}

impl BackingBytes {
    /// The three counters added up.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.write_only_static + self.dynamic + self.other
    }
}

/// Whether a buffer's CPU backing may be released once its upload is queued.
///
/// `D3DUSAGE_WRITEONLY` in `D3DPOOL_DEFAULT` is the class D3D9 documents as
/// never read back, and without `D3DUSAGE_DYNAMIC` it is also `Staged`, so
/// the GPU reads a device buffer rather than this memory. `ProcessVertices`
/// is the one path that reads a vertex buffer's bytes back on the CPU
/// regardless of what the usage promised, and `D3DUSAGE_SOFTWAREPROCESSING`
/// is the usage a title declares for a buffer it feeds to it, so that
/// declaration keeps the backing.
#[must_use]
pub const fn may_release_backing(usage: u32, pool: u32) -> bool {
    pool == D3DPOOL_DEFAULT
        && usage & D3DUSAGE_WRITEONLY != 0
        && usage & (D3DUSAGE_DYNAMIC | D3DUSAGE_SOFTWAREPROCESSING) == 0
}

/// Pick the gauge counter a buffer's backing is charged to.
#[must_use]
pub const fn classify_backing(usage: u32, pool: u32) -> BackingClass {
    if may_release_backing(usage, pool) {
        BackingClass::WriteOnlyStatic
    } else if usage & D3DUSAGE_DYNAMIC != 0 {
        BackingClass::Dynamic
    } else {
        BackingClass::Other
    }
}

/// Live CPU-backing bytes, split by [`BackingClass`].
#[must_use]
pub fn live_backing_bytes() -> BackingBytes {
    BackingBytes {
        write_only_static: WRITE_ONLY_STATIC_BYTES.load(Ordering::Relaxed),
        dynamic: DYNAMIC_BYTES.load(Ordering::Relaxed),
        other: OTHER_BYTES.load(Ordering::Relaxed),
    }
}

/// A VB/IB's CPU backing plus the record of what it still holds.
///
/// Owns the gauge accounting: the padded byte count is charged to the
/// buffer's class while a `PageBox` is present and returned the moment it
/// is not, including when the whole buffer is destroyed.
pub struct BufferBacking {
    page_box: Option<PageBox>,
    /// Padded length of the backing, kept across a release.
    ///
    /// Draw snapshots carry it as the buffer's length long after the
    /// bytes are gone, so it cannot live on the `PageBox` alone.
    padded_len: usize,
    logical_len: u32,
    class: BackingClass,
    state: BackingState,
    /// Whether the backing is held for the rest of the buffer's life.
    ///
    /// Set by [`BufferBacking::adopt_device_copy`], the one path that has
    /// paid a GPU stall to get these bytes back, and never cleared.
    pinned: bool,
}

impl BufferBacking {
    /// Adopt a freshly allocated backing for a buffer of `logical_len` bytes.
    #[must_use]
    pub fn new(page_box: PageBox, logical_len: u32, class: BackingClass) -> Self {
        let padded_len = page_box.len();
        charge(class, padded_len);
        Self {
            page_box: Some(page_box),
            padded_len,
            logical_len,
            class,
            state: BackingState::Mirrors,
            pinned: false,
        }
    }

    /// How much of the device buffer's contents this backing still holds.
    #[must_use]
    pub const fn state(&self) -> BackingState {
        self.state
    }

    /// Whether the backing has been released and holds no bytes at all.
    #[must_use]
    pub const fn is_released(&self) -> bool {
        matches!(self.state, BackingState::Released)
    }

    /// Whether an upload out of this backing may widen past what a `Lock` announced.
    ///
    /// True only in [`BackingState::Mirrors`], where every byte outside the
    /// announced window is the byte the device buffer already holds.
    #[must_use]
    pub const fn may_widen_upload(&self) -> bool {
        matches!(self.state, BackingState::Mirrors)
    }

    /// The backing's address as the wire carries it; `0` once released.
    #[must_use]
    pub fn ptr(&self) -> u64 {
        self.page_box.as_ref().map_or(0, |b| b.as_ptr() as u64)
    }

    /// The backing's padded length, which outlives the bytes themselves.
    #[must_use]
    pub const fn padded_len(&self) -> u64 {
        self.padded_len as u64
    }

    /// The backing's full padded region; empty once released.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self.page_box.as_ref().map_or(&[], PageBox::as_slice)
    }

    /// Mutable counterpart of [`Self::as_slice`].
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.page_box
            .as_mut()
            .map_or_else(<&mut [u8]>::default, PageBox::as_mut_slice)
    }

    /// Read pointer to `offset` bytes into the backing, or `None` once released.
    #[must_use]
    pub fn read_ptr_at(&self, offset: usize) -> Option<*const u8> {
        let page_box = self.page_box.as_ref()?;
        if offset > page_box.len() {
            return None;
        }
        // SAFETY: `offset <= len`, so the result is inside the allocation or
        // one past its end, which is what pointer arithmetic allows.
        Some(unsafe { page_box.as_ptr().add(offset) })
    }

    /// Write pointer to `offset` bytes into the backing, or `None` once released.
    pub fn write_ptr_at(&mut self, offset: usize) -> Option<*mut u8> {
        let page_box = self.page_box.as_mut()?;
        if offset > page_box.len() {
            return None;
        }
        // SAFETY: `offset <= len`, so the result is inside the allocation or
        // one past its end, which is what pointer arithmetic allows.
        Some(unsafe { page_box.as_mut_ptr().add(offset) })
    }

    /// Hand the backing out and mark the buffer as holding no CPU copy.
    ///
    /// Returns `None` when there was nothing left to hand out. Serves both
    /// the address-space release and the buffer's own destruction, which
    /// pass the `PageBox` on to the retention pipeline.
    pub fn release(&mut self) -> Option<PageBox> {
        let page_box = self.page_box.take()?;
        discharge(self.class, self.padded_len);
        self.state = BackingState::Released;
        Some(page_box)
    }

    /// Whether the backing is held for the rest of the buffer's life.
    ///
    /// A pinned backing has been read back off the GPU once already, so
    /// releasing it again would only buy another stall.
    #[must_use]
    pub const fn is_pinned(&self) -> bool {
        self.pinned
    }

    /// Adopt a copy of the device buffer's contents, read back off the GPU.
    ///
    /// The bytes come from the device buffer, so the backing mirrors it
    /// again rather than holding only what is written next. Pins the
    /// backing: the read that produced these bytes is a GPU stall, and a
    /// buffer pays it once.
    pub fn adopt_device_copy(&mut self, page_box: PageBox) {
        if self.page_box.is_some() {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "adopt_device_copy: the buffer already holds a backing, dropping the read-back copy");
            return;
        }
        self.padded_len = page_box.len();
        charge(self.class, self.padded_len);
        self.page_box = Some(page_box);
        self.state = BackingState::Mirrors;
        self.pinned = true;
    }

    /// Give a released buffer a backing again, holding only what is written next.
    pub fn restore(&mut self, page_box: PageBox) {
        self.padded_len = page_box.len();
        charge(self.class, self.padded_len);
        self.page_box = Some(page_box);
        self.state = BackingState::Partial;
    }

    /// Swap in a fresh backing for a rename, handing the old one back.
    ///
    /// The state is unchanged: a rename replaces one whole-buffer copy with
    /// another, and the caller decides whether to preserve the contents.
    /// `None` means the buffer had already released its backing, which the
    /// rename path never reaches.
    pub fn replace(&mut self, page_box: PageBox) -> Option<PageBox> {
        let old = self.page_box.take()?;
        discharge(self.class, self.padded_len);
        self.padded_len = page_box.len();
        charge(self.class, self.padded_len);
        self.page_box = Some(page_box);
        Some(old)
    }

    /// Record that `[min, max)` of the backing has been queued for upload.
    ///
    /// A range covering the whole buffer makes the device buffer a copy of
    /// the backing whatever the backing held, so a `Partial` backing
    /// mirrors the device again from here on.
    pub const fn note_upload(&mut self, min: u32, max: u32) {
        if matches!(self.state, BackingState::Partial) && min == 0 && max >= self.logical_len {
            self.state = BackingState::Mirrors;
        }
    }
}

impl Drop for BufferBacking {
    fn drop(&mut self) {
        if self.page_box.is_some() {
            discharge(self.class, self.padded_len);
        }
    }
}

/// Add `bytes` to `class`'s gauge counter.
fn charge(class: BackingClass, bytes: usize) {
    counter(class).fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Return `bytes` to `class`'s gauge counter.
fn discharge(class: BackingClass, bytes: usize) {
    counter(class).fetch_sub(bytes as u64, Ordering::Relaxed);
}

/// The counter a class is charged to.
fn counter(class: BackingClass) -> &'static AtomicU64 {
    match class {
        BackingClass::WriteOnlyStatic => &WRITE_ONLY_STATIC_BYTES,
        BackingClass::Dynamic => &DYNAMIC_BYTES,
        BackingClass::Other => &OTHER_BYTES,
    }
}

#[cfg(test)]
mod tests;
