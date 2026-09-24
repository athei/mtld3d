//! Per-device GPU buffers the submit path reuses once the submissions that used them retired.
//!
//! A submission stamps every reuse with its sequence and the command buffer it
//! encodes into, the upload or the render one, since each advances its own
//! PE-side retirement counter from its completion handler. A region is written
//! again only after the counter of every command buffer that read it has
//! reached the stamped sequence: an earlier write would change bytes the GPU
//! has still to read, which Apple Silicon usually hides and the Intel and
//! paravirtual devices do not.

use core::{ffi::c_void, ptr::NonNull};
use std::sync::atomic::{AtomicU64, Ordering};

use mtld3d_shared::SubmitFrameParams;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResource, MTLResourceOptions};

/// Capacity of the first upload-ring chunk.
const FIRST_CHUNK_BYTES: usize = 64 * 1024;
/// Largest capacity the ring grows a chunk to for the running high-water.
///
/// A single payload larger than this still gets a chunk of its own size.
const MAX_CHUNK_BYTES: usize = 16 * 1024 * 1024;
/// Retired chunks the ring keeps beside the active one for the next frames.
const SPARE_CHUNKS: usize = 2;

/// The command buffer an encode goes into, and the counters that retire it.
pub struct SubmitStamp {
    /// The submission's sequence, or `u64::MAX` for one without counters.
    seq: u64,
    /// The upload command buffer rather than the render one.
    upload: bool,
    /// Address of the PE-side render retirement counter, 0 when absent.
    draw_counter: u64,
    /// Address of the PE-side upload retirement counter, 0 when absent.
    upload_counter: u64,
}

impl SubmitStamp {
    /// The render command buffer of the submission `params` describes.
    ///
    /// A submission without a sequence or a render counter (a frame stamped
    /// before its counters were wired) registers no completion handler, so
    /// nothing it uses ever retires: it gets `u64::MAX`, which no counter
    /// reaches, and `persistent` reports false so the caller keeps its
    /// buffers out of the device's pools.
    #[must_use]
    pub const fn new(params: &SubmitFrameParams) -> Self {
        let persistent = params.submit_seq != 0 && params.coherent_seq_ptr != 0;
        Self {
            seq: if persistent {
                params.submit_seq
            } else {
                u64::MAX
            },
            upload: false,
            draw_counter: params.coherent_seq_ptr,
            upload_counter: params.upload_coherent_seq_ptr,
        }
    }

    /// The same submission's upload command buffer.
    #[must_use]
    pub const fn upload(&self) -> Self {
        Self {
            seq: self.seq,
            upload: true,
            draw_counter: self.draw_counter,
            upload_counter: self.upload_counter,
        }
    }

    /// A render-buffer stamp for `seq` against two counters the caller owns.
    #[cfg(test)]
    #[must_use]
    pub fn for_counters(seq: u64, draw: &AtomicU64, upload: &AtomicU64) -> Self {
        Self {
            seq,
            upload: false,
            draw_counter: core::ptr::from_ref(draw) as u64,
            upload_counter: core::ptr::from_ref(upload) as u64,
        }
    }

    /// Whether what this submission uses can retire, so the device's pools may keep it.
    #[must_use]
    pub const fn persistent(&self) -> bool {
        self.seq != u64::MAX
    }

    /// The newest sequence the render command buffers have retired.
    fn draw_retired(&self) -> u64 {
        load_counter(self.draw_counter)
    }

    /// The newest sequence the upload command buffers have retired.
    fn upload_retired(&self) -> u64 {
        load_counter(self.upload_counter)
    }
}

/// Read a PE-side retirement counter; 0 for an absent one.
fn load_counter(address: u64) -> u64 {
    if address == 0 {
        return 0;
    }
    // SAFETY: SubmitFrame carries stable PE `AtomicU64` addresses that stay
    // live through the call, and every stamp lives only inside that call.
    let counter = unsafe { &*(address as *const AtomicU64) };
    counter.load(Ordering::Acquire)
}

/// The newest submission whose command buffers use a resource, per retirement counter.
#[derive(Default)]
pub struct LastUse {
    draw: u64,
    upload: u64,
}

impl LastUse {
    /// Note a use by `stamp`'s command buffer.
    pub const fn record(&mut self, stamp: &SubmitStamp) {
        if stamp.upload {
            self.upload = stamp.seq;
        } else {
            self.draw = stamp.seq;
        }
    }

    /// Whether every command buffer that used the resource has finished on the GPU.
    #[must_use]
    pub fn retired(&self, stamp: &SubmitStamp) -> bool {
        (self.draw == 0 || stamp.draw_retired() >= self.draw)
            && (self.upload == 0 || stamp.upload_retired() >= self.upload)
    }

    /// Whether `stamp`'s command buffer may write the resource now.
    ///
    /// Either every earlier use retired, or the only unretired use is this
    /// same command buffer, whose encoders Metal's hazard tracking orders.
    #[must_use]
    pub fn writable_by(&self, stamp: &SubmitStamp) -> bool {
        let (own, other, own_retired, other_retired) = if stamp.upload {
            (self.upload, self.draw, stamp.upload_retired(), stamp.draw_retired())
        } else {
            (self.draw, self.upload, stamp.draw_retired(), stamp.upload_retired())
        };
        (other == 0 || other_retired >= other)
            && (own == 0 || own == stamp.seq || own_retired >= own)
    }
}

/// Where the ring put one payload.
pub struct RingSlice<'a> {
    pub buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub offset: usize,
}

/// Shared-storage chunks that inline draw data rides to the GPU.
///
/// Metal has no inline index draw and caps `setVertexBytes` at
/// `SET_BYTES_MAX`, so those payloads need a buffer. Each is copied at the
/// next aligned offset of the active chunk; a full chunk is set aside and a
/// retired spare, or a new chunk, takes over. Chunks grow to the largest
/// submission's total, so a steady workload settles on one chunk per frame
/// in flight. The ring only appends into a chunk until it moves on, so a
/// chunk still read by an earlier submission is safe to append to; only
/// starting a chunk again from offset 0 waits for its retirement.
#[derive(Default)]
pub struct UploadRing {
    active: Option<Chunk>,
    spare: Vec<Chunk>,
    chunk_bytes: usize,
    /// Bytes this submission wrote, alignment included.
    written: usize,
}

impl UploadRing {
    /// Copy `bytes` into the ring for a read by `stamp`'s command buffer.
    ///
    /// `align` is the offset alignment the binding needs, a power of two.
    /// `None` only when a new chunk had to be allocated and Metal refused.
    pub fn write(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        stamp: &SubmitStamp,
        bytes: &[u8],
        align: usize,
    ) -> Option<RingSlice<'_>> {
        let offset = self.reserve(device, stamp, bytes.len(), align)?;
        let chunk = self.active.as_mut()?;
        // SAFETY: `reserve` placed `offset..offset + len` inside the chunk's
        // shared storage, a region no queued command buffer reads: it lies
        // past every earlier write since the chunk was last started, and a
        // chunk is started again only after its readers retired.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                chunk.contents.as_ptr().cast::<u8>().add(offset),
                bytes.len(),
            );
        }
        chunk.last_use.record(stamp);
        Some(RingSlice {
            buffer: &chunk.buffer,
            offset,
        })
    }

    /// Close a submission: grow to its total, and drop spares it outgrew or no longer needs.
    pub fn end_submission(&mut self, stamp: &SubmitStamp) {
        let high_water = self.written.next_power_of_two().min(MAX_CHUNK_BYTES);
        self.chunk_bytes = self.chunk_bytes.max(high_water);
        self.written = 0;
        let chunk_bytes = self.chunk_bytes;
        let mut kept = 0;
        self.spare.retain(|chunk| {
            if !chunk.last_use.retired(stamp) {
                return true;
            }
            if chunk.capacity < chunk_bytes || kept == SPARE_CHUNKS {
                return false;
            }
            kept += 1;
            true
        });
    }

    /// Chunks the ring holds, the active one included.
    #[cfg(test)]
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.spare.len() + usize::from(self.active.is_some())
    }

    /// Find room for `len` bytes and return its offset in the active chunk.
    fn reserve(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        stamp: &SubmitStamp,
        len: usize,
        align: usize,
    ) -> Option<usize> {
        if let Some(chunk) = self.active.as_mut() {
            let offset = chunk.cursor.next_multiple_of(align);
            if offset.checked_add(len).is_some_and(|end| end <= chunk.capacity) {
                self.written += offset + len - chunk.cursor;
                chunk.cursor = offset + len;
                return Some(offset);
            }
        }
        self.written += len;
        let need = len
            .max(self.chunk_bytes)
            .max(self.written.min(MAX_CHUNK_BYTES))
            .max(FIRST_CHUNK_BYTES);
        let reusable = self
            .spare
            .iter()
            .position(|chunk| chunk.capacity >= need && chunk.last_use.retired(stamp));
        let mut next = match reusable {
            Some(index) => self.spare.swap_remove(index),
            None => Chunk::new(device, need.next_power_of_two())?,
        };
        next.cursor = len;
        next.last_use = LastUse::default();
        if let Some(full) = self.active.replace(next) {
            self.spare.push(full);
        }
        Some(0)
    }
}

/// One shared-storage buffer of the ring and where its next payload goes.
struct Chunk {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    contents: NonNull<c_void>,
    capacity: usize,
    cursor: usize,
    last_use: LastUse,
}

// SAFETY: a chunk lives on its device's record behind a mutex, so one
// submitting thread at a time touches it. Retaining and releasing an
// `MTLBuffer` is thread-safe, and the contents pointer names shared storage
// that stays mapped for the buffer's life on any thread.
unsafe impl Send for Chunk {}

impl Chunk {
    fn new(device: &ProtocolObject<dyn MTLDevice>, capacity: usize) -> Option<Self> {
        let buffer =
            device.newBufferWithLength_options(capacity, MTLResourceOptions::StorageModeShared)?;
        buffer.setLabel(Some(&NSString::from_str("mtld3d-upload-ring")));
        let contents = buffer.contents();
        Some(Self {
            buffer,
            contents,
            capacity,
            cursor: 0,
            last_use: LastUse::default(),
        })
    }
}

#[cfg(test)]
mod tests;
