//! A device's arena of upload snapshots: copies of the box a partial texture upload carries.
//!
//! A partial upload normally reads the level's staging pages in place, from
//! when it is scheduled until the GPU has copied them and the encoder has seen
//! that copy, replays included. A level the game rewrites in part while such an
//! upload is queued would have to move its staging to fresh pages, carrying the
//! whole level, on every lock. Once a level has needed that once, its partial
//! uploads copy their box here instead when they are scheduled, so they no
//! longer read the staging and the next lock writes it in place.
//!
//! The arena is a short list of chunks. Snapshots are appended to the active
//! chunk at the next aligned offset; a chunk that cannot fit a request goes to
//! the back of the queue of full chunks, and the chunk at the front of that
//! queue starts over from offset zero once nothing reads it any more, which
//! [`PageBox::has_readers`] answers: it stays raised until every upload that
//! read a snapshot in it has been copied by the GPU and acknowledged by the
//! encoder, including an upload the encoder replays after an aborted submit.
//! A sequence-stamped ring cannot use the submit sequence for that: an
//! aborted submission's sequence retires while its uploads still wait to be
//! replayed from the same bytes. A front chunk still read means a new chunk,
//! up to [`MAX_CHUNKS`]; past that the upload reads the staging as it would
//! without a snapshot, and the lock rule covers the next write.
//!
//! Its unix-side twin, the encoder's upload ring for inline draw data, keeps
//! its chunks in native memory. These bytes are PE pages instead, because the
//! texture-upload record, its page lease and its replay already carry PE pages
//! across the boundary: a snapshot is the same kind of source as the staging
//! it was copied from, at an offset into a shared chunk.

use std::{collections::VecDeque, sync::Arc};

use crate::page_box::{PageBox, PageBoxRead};

/// Bytes of one arena chunk.
///
/// About eight times the largest snapshot a sprite-atlas workload takes
/// (129x112 texels at four bytes, 57 KiB of rows), so a chunk holds a run of
/// small boxes, and below snmalloc's 1 MiB local-cache cutoff
/// ([`crate::page_box::bypasses_local_cache`]), so allocating a chunk stays a
/// buddy operation rather than a commit.
pub const CHUNK_BYTES: usize = 256 * 1024;

/// The largest snapshot the arena takes: a whole chunk.
///
/// A snapshot of a box never costs more than the whole-level copy a rename
/// of its level would, so any box that fits a chunk is worth taking. One that
/// does not fit the active chunk's tail moves the arena to the next chunk,
/// leaving the tail unused; large snapshots are rare, since only a level a
/// partial lock had to rename takes any, and [`MAX_CHUNKS`] bounds the arena
/// whatever they waste.
pub const SNAPSHOT_MAX_BYTES: usize = CHUNK_BYTES;

/// The most chunks the arena holds, 16 MiB of the 32-bit address space.
///
/// The busiest frame measured snapshots about 1.6 MiB, and a chunk comes back
/// once its frame's uploads are acknowledged, three to four frames later, so
/// the cap leaves room for a workload several times heavier before an upload
/// falls back to reading the staging.
pub const MAX_CHUNKS: usize = 64;

/// Offset alignment of every snapshot: the largest D3D9 texel or compressed block.
pub const SNAPSHOT_ALIGN: usize = 16;

/// Where a snapshot landed: a read of its chunk and the byte offset into it.
pub struct UploadSnapshot {
    /// A published read of the chunk, raised before the snapshot leaves the arena.
    pub read: PageBoxRead,
    /// Byte offset of the snapshot's first row in the chunk.
    pub offset: u32,
}

/// The active chunk with its bump cursor, and the full chunks oldest first.
#[derive(Default)]
pub struct UploadSnapshots {
    active: Option<Arc<PageBox>>,
    /// First byte of the active chunk no snapshot handed out since its start covers.
    cursor: usize,
    full: VecDeque<Arc<PageBox>>,
}

impl UploadSnapshots {
    /// Copy a snapshot of `len` bytes into the arena through `fill`.
    ///
    /// `fill` writes the snapshot into the slice it is given and answers
    /// whether it did; a `false` leaves the slice unclaimed. `None` when the
    /// request is empty or over [`SNAPSHOT_MAX_BYTES`], when every chunk is
    /// still read and the arena holds [`MAX_CHUNKS`], when a new chunk could
    /// not be allocated, or when `fill` declined. The caller then uploads from
    /// the staging; nothing here waits.
    pub fn write(
        &mut self,
        len: usize,
        fill: impl FnOnce(&mut [u8]) -> bool,
    ) -> Option<UploadSnapshot> {
        if len == 0 || len > SNAPSHOT_MAX_BYTES {
            return None;
        }
        let start = self.room_for(len)?;
        let offset = u32::try_from(start).ok()?;
        let chunk = self.active.as_ref()?;
        let base = chunk.as_ptr().cast_mut().wrapping_add(start);
        // SAFETY: `room_for` placed `[start, start + len)` inside the chunk at
        // or past the cursor, so no snapshot handed out since the chunk started
        // covers it, and a chunk starts over only once no upload reads it. The
        // chunk was allocated zeroed, so every byte is initialized, and no
        // other reference to these bytes exists while the slice lives.
        let destination = unsafe { core::slice::from_raw_parts_mut(base, len) };
        if !fill(destination) {
            return None;
        }
        let read = PageBoxRead::new(Arc::clone(chunk));
        self.cursor = start + len;
        Some(UploadSnapshot { read, offset })
    }

    /// Padded bytes of every chunk the arena holds.
    #[must_use]
    pub fn held_bytes(&self) -> u64 {
        self.active
            .iter()
            .chain(self.full.iter())
            .map(|chunk| chunk.len() as u64)
            .sum()
    }

    /// Chunks the arena holds, active and full.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.full.len() + usize::from(self.active.is_some())
    }

    /// The offset in the active chunk a request of `len` bytes goes to, moving chunks if needed.
    fn room_for(&mut self, len: usize) -> Option<usize> {
        let start = self.cursor.next_multiple_of(SNAPSHOT_ALIGN);
        if self.active.is_some() && start + len <= CHUNK_BYTES {
            return Some(start);
        }
        if let Some(chunk) = self.active.take() {
            self.full.push_back(chunk);
        }
        if self.full.front().is_some_and(|chunk| !chunk.has_readers()) {
            self.active = self.full.pop_front();
        } else if self.chunk_count() < MAX_CHUNKS {
            self.active = Some(Arc::new(PageBox::try_new_zeroed(CHUNK_BYTES)?));
        } else {
            return None;
        }
        self.cursor = 0;
        Some(0)
    }
}

#[cfg(test)]
mod tests;
