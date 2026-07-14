//! Stream-0 vertex buffer + index buffer bindings owned by `DeviceInner`.
//!
//! COM `AddRef`/`Release` pairing for `SetStreamSource` / `SetIndices` lives
//! here so the ref-count invariant can't leak elsewhere in the crate.

use crate::{
    com_ref::{Bound, CachedComPtr},
    index_buffer::Direct3DIndexBuffer9,
    vertex_buffer::Direct3DVertexBuffer9,
};

/// Maximum vertex streams we accept (matches `caps.max_streams`).
///
/// Stream 0 is the only stream we render; streams 1..MAX exist solely so
/// `SetStreamSource` / `GetStreamSource` round-trip (the single-stream
/// architecture never fetches the higher streams). A caller that binds a
/// higher stream, releases its own reference, then reads the stream back
/// relies on the binding keeping the buffer alive — the `Bound` marker
/// provides exactly that.
pub const MAX_STREAMS: usize = 16;

/// One higher-stream (1..MAX) binding: a `Bound` buffer slot plus its offset/stride.
struct StreamSlot {
    vb: CachedComPtr<Direct3DVertexBuffer9, Bound>,
    offset: u32,
    stride: u32,
}

impl StreamSlot {
    const fn new() -> Self {
        Self {
            vb: CachedComPtr::null(),
            offset: 0,
            stride: 0,
        }
    }
}

pub struct BoundBuffers {
    /// Stream-0 vertex buffer slot.
    ///
    /// Uses the `Bound` ownership marker — swaps bump the wrapper's
    /// `private_refcount` inline.
    vertex_buffer: CachedComPtr<Direct3DVertexBuffer9, Bound>,
    vb_offset: u32,
    vb_stride: u32,
    /// Indexed-draw source slot. Same `Bound` semantics.
    index_buffer: CachedComPtr<Direct3DIndexBuffer9, Bound>,
    /// Higher streams (1..MAX), indexed by `stream - 1`.
    ///
    /// Stored for get/set round-trip only; never rendered.
    extra_streams: [StreamSlot; MAX_STREAMS - 1],
}

impl BoundBuffers {
    pub const fn new() -> Self {
        Self {
            vertex_buffer: CachedComPtr::null(),
            vb_offset: 0,
            vb_stride: 0,
            index_buffer: CachedComPtr::null(),
            extra_streams: [const { StreamSlot::new() }; MAX_STREAMS - 1],
        }
    }

    pub const fn vertex_buffer(&self) -> *mut Direct3DVertexBuffer9 {
        self.vertex_buffer.raw()
    }

    pub const fn vb_offset(&self) -> u32 {
        self.vb_offset
    }

    pub const fn vb_stride(&self) -> u32 {
        self.vb_stride
    }

    pub const fn index_buffer(&self) -> *mut Direct3DIndexBuffer9 {
        self.index_buffer.raw()
    }

    /// Bind `new` at stream 0 with COM `AddRef`/`Release`. Pass null to clear.
    pub fn replace_vertex_buffer(
        &mut self,
        new: *mut Direct3DVertexBuffer9,
        offset: u32,
        stride: u32,
    ) {
        // SAFETY: `new` came from the IDirect3DDevice9 vtable layer; the
        // SetStreamSource thunk guarantees it is null or *mut Direct3DVertexBuffer9.
        self.vertex_buffer = unsafe { CachedComPtr::adopt(new) };
        // D3D9 retains the previous offset/stride when the stream source is set
        // to NULL: `GetStreamSource` after `SetStreamSource(0, NULL, 0, 0)`
        // reports the last non-null stride, not 0. Only a non-null bind updates
        // them.
        if !new.is_null() {
            self.vb_offset = offset;
            self.vb_stride = stride;
        }
    }

    /// `DrawPrimitiveUP` / `DrawIndexedPrimitiveUP` reset stream source 0 to `(NULL, 0, 0)`.
    ///
    /// That reset happens on success — unlike `SetStreamSource(0, NULL, …)`,
    /// which retains the prior offset/stride per the D3D9 spec. Drops the
    /// previously-bound VB's private reference.
    pub fn reset_stream0(&mut self) {
        // SAFETY: a null pointer is a valid `CachedComPtr::adopt` input; the
        // assignment drops the prior `Bound` reference.
        self.vertex_buffer = unsafe { CachedComPtr::adopt(core::ptr::null_mut()) };
        self.vb_offset = 0;
        self.vb_stride = 0;
    }

    /// Bind `new` at `stream` (0..MAX) with COM `AddRef`/`Release`.
    ///
    /// Stream 0 is the rendered source; higher streams are stored for
    /// round-trip only. Pass null to clear. The caller (the `SetStreamSource`
    /// thunk) must keep `stream` below [`MAX_STREAMS`].
    pub fn set_stream(
        &mut self,
        stream: usize,
        new: *mut Direct3DVertexBuffer9,
        offset: u32,
        stride: u32,
    ) {
        if stream == 0 {
            self.replace_vertex_buffer(new, offset, stride);
            return;
        }
        let slot = &mut self.extra_streams[stream - 1];
        // SAFETY: `new` came from the IDirect3DDevice9 vtable layer; the
        // SetStreamSource thunk guarantees it is null or *mut Direct3DVertexBuffer9.
        slot.vb = unsafe { CachedComPtr::adopt(new) };
        // Same NULL-bind offset/stride retention quirk as stream 0.
        if !new.is_null() {
            slot.offset = offset;
            slot.stride = stride;
        }
    }

    /// The vertex buffer bound at `stream` (raw pointer; null if unbound).
    pub const fn stream_vertex_buffer(&self, stream: usize) -> *mut Direct3DVertexBuffer9 {
        if stream == 0 {
            self.vertex_buffer.raw()
        } else {
            self.extra_streams[stream - 1].vb.raw()
        }
    }

    /// The offset bound at `stream`.
    pub const fn stream_offset(&self, stream: usize) -> u32 {
        if stream == 0 {
            self.vb_offset
        } else {
            self.extra_streams[stream - 1].offset
        }
    }

    /// The stride bound at `stream`.
    pub const fn stream_stride(&self, stream: usize) -> u32 {
        if stream == 0 {
            self.vb_stride
        } else {
            self.extra_streams[stream - 1].stride
        }
    }

    /// Bind `new` as the indexed-draw source with COM `AddRef`/`Release`.
    pub fn replace_index_buffer(&mut self, new: *mut Direct3DIndexBuffer9) {
        // SAFETY: `new` came from the IDirect3DDevice9 vtable layer; the
        // SetIndices thunk guarantees it is null or *mut Direct3DIndexBuffer9.
        self.index_buffer = unsafe { CachedComPtr::adopt(new) };
    }

    /// Release and null every buffer slot. Used from the device release path.
    pub fn teardown(&mut self) {
        self.vertex_buffer = CachedComPtr::null();
        self.index_buffer = CachedComPtr::null();
        self.vb_offset = 0;
        self.vb_stride = 0;
        for slot in &mut self.extra_streams {
            *slot = StreamSlot::new();
        }
    }
}
