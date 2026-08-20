//! Vertex stream + index buffer bindings owned by `DeviceInner`.
//!
//! COM `AddRef`/`Release` pairing for `SetStreamSource` / `SetIndices` lives
//! here so the ref-count invariant can't leak elsewhere in the crate. Every
//! stream is a rendered source: stream `n` binds at Metal vertex buffer slot
//! `n` when the bound declaration reads from it.

use mtld3d_core::streams::STREAM_FREQ_DEFAULT;
use mtld3d_types::MAX_STREAMS;

use crate::{
    com_ref::{Bound, CachedComPtr},
    index_buffer::Direct3DIndexBuffer9,
    vertex_buffer::Direct3DVertexBuffer9,
};

/// One vertex stream: a `Bound` buffer slot plus its offset, stride and frequency.
struct StreamSlot {
    vb: CachedComPtr<Direct3DVertexBuffer9, Bound>,
    offset: u32,
    stride: u32,
    /// Raw `SetStreamSourceFreq` word, flags included.
    freq: u32,
}

impl StreamSlot {
    const fn new() -> Self {
        Self {
            vb: CachedComPtr::null(),
            offset: 0,
            stride: 0,
            freq: STREAM_FREQ_DEFAULT,
        }
    }
}

pub struct BoundBuffers {
    /// Vertex streams, indexed by D3D9 stream number.
    ///
    /// Uses the `Bound` ownership marker — swaps bump the wrapper's
    /// `private_refcount` inline. A caller that binds a stream, releases its
    /// own reference, then reads the stream back relies on the binding keeping
    /// the buffer alive, which the marker provides.
    streams: [StreamSlot; MAX_STREAMS as usize],
    /// Indexed-draw source slot. Same `Bound` semantics.
    index_buffer: CachedComPtr<Direct3DIndexBuffer9, Bound>,
}

impl BoundBuffers {
    pub const fn new() -> Self {
        Self {
            streams: [const { StreamSlot::new() }; MAX_STREAMS as usize],
            index_buffer: CachedComPtr::null(),
        }
    }

    pub const fn index_buffer(&self) -> *mut Direct3DIndexBuffer9 {
        self.index_buffer.raw()
    }

    /// `DrawPrimitiveUP` / `DrawIndexedPrimitiveUP` reset stream source 0 to `(NULL, 0, 0)`.
    ///
    /// That reset happens on success — unlike `SetStreamSource(0, NULL, …)`,
    /// which retains the prior offset/stride per the D3D9 spec. Drops the
    /// previously-bound VB's private reference. The stream's frequency is
    /// untouched.
    pub fn reset_stream0(&mut self) {
        self.restore_stream(0, core::ptr::null_mut(), 0, 0);
    }

    /// Bind `new` at `stream` (0..MAX) with COM `AddRef`/`Release`.
    ///
    /// Pass null to clear. The caller (the `SetStreamSource` thunk) must keep
    /// `stream` below [`MAX_STREAMS`].
    pub fn set_stream(
        &mut self,
        stream: usize,
        new: *mut Direct3DVertexBuffer9,
        offset: u32,
        stride: u32,
    ) {
        let slot = &mut self.streams[stream];
        // SAFETY: `new` came from the IDirect3DDevice9 vtable layer; the
        // SetStreamSource thunk guarantees it is null or *mut Direct3DVertexBuffer9.
        slot.vb = unsafe { CachedComPtr::adopt(new) };
        // D3D9 retains the previous offset/stride when the stream source is set
        // to NULL: `GetStreamSource` after `SetStreamSource(n, NULL, 0, 0)`
        // reports the last non-null stride, not 0. Only a non-null bind updates
        // them.
        if !new.is_null() {
            slot.offset = offset;
            slot.stride = stride;
        }
    }

    /// Write `(vb, offset, stride)` to `stream` unconditionally.
    ///
    /// The state-block restore path and the UP-draw reset, which replace the
    /// whole binding rather than applying the NULL-bind retention quirk of
    /// [`Self::set_stream`].
    pub fn restore_stream(
        &mut self,
        stream: usize,
        vb: *mut Direct3DVertexBuffer9,
        offset: u32,
        stride: u32,
    ) {
        let slot = &mut self.streams[stream];
        // SAFETY: `vb` is null or a live `Direct3DVertexBuffer9` held by the
        // caller (a state block's own reference, or null for the UP reset).
        slot.vb = unsafe { CachedComPtr::adopt(vb) };
        slot.offset = offset;
        slot.stride = stride;
    }

    /// The vertex buffer bound at `stream` (raw pointer; null if unbound).
    pub const fn stream_vertex_buffer(&self, stream: usize) -> *mut Direct3DVertexBuffer9 {
        self.streams[stream].vb.raw()
    }

    /// The offset bound at `stream`.
    pub const fn stream_offset(&self, stream: usize) -> u32 {
        self.streams[stream].offset
    }

    /// The stride bound at `stream`.
    pub const fn stream_stride(&self, stream: usize) -> u32 {
        self.streams[stream].stride
    }

    /// The raw `SetStreamSourceFreq` word of `stream`.
    pub const fn stream_freq(&self, stream: usize) -> u32 {
        self.streams[stream].freq
    }

    /// Store a validated `SetStreamSourceFreq` word for `stream`.
    pub const fn set_stream_freq(&mut self, stream: usize, setting: u32) {
        self.streams[stream].freq = setting;
    }

    /// Bit `s` set: a vertex buffer is bound at stream `s`.
    pub fn bound_mask(&self) -> u16 {
        self.streams
            .iter()
            .enumerate()
            .filter(|(_, slot)| !slot.vb.raw().is_null())
            .fold(0u16, |m, (s, _)| m | (1 << s))
    }

    /// Bind `new` as the indexed-draw source with COM `AddRef`/`Release`.
    pub fn replace_index_buffer(&mut self, new: *mut Direct3DIndexBuffer9) {
        // SAFETY: `new` came from the IDirect3DDevice9 vtable layer; the
        // SetIndices thunk guarantees it is null or *mut Direct3DIndexBuffer9.
        self.index_buffer = unsafe { CachedComPtr::adopt(new) };
    }

    /// Release and null every buffer slot and reset every frequency.
    ///
    /// Used from the device release and `Reset` paths.
    pub fn teardown(&mut self) {
        self.index_buffer = CachedComPtr::null();
        for slot in &mut self.streams {
            *slot = StreamSlot::new();
        }
    }
}
