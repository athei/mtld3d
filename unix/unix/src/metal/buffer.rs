//! Generic `MTLBuffer` create/destroy at the Metal layer.
//!
//! `create_buffer` wraps the caller-provided memory region directly via
//! `newBufferWithBytesNoCopy:length:options:deallocator:`. The PE side
//! (per-buffer `PageBox` for VB/IB, per-mip staging Box for textures) owns
//! the allocation and guarantees the backing pointer lives in the 32-bit
//! PE's addressable range; Metal's own allocator would otherwise return a
//! high-64-bit address the 32-bit PE can't dereference.

use core::{ffi::c_void, ptr::NonNull};

use log::warn;
use mtld3d_shared::{
    BufferCreateDesc,
    mtl::{BufferKind, StorageMode},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResource, MTLResourceOptions};

use crate::LOG_TARGET;

/// Wrap `backing_ptr` (caller-owned, page-aligned, PE-heap) in an `MTLBuffer`.
///
/// Caller retains ownership of the memory — the deallocator block is nil, so
/// `destroy_buffer` just releases the Metal wrapper and leaves the PE
/// allocation alone.
///
/// One descriptor → one `MTLBuffer`. The batched handler iterates this
/// per element; same call shape used by load-phase warmup batches and
/// one-off lazy creates (mid-frame DISCARD renames, per-upload staging).
pub fn create_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    desc: &BufferCreateDesc,
) -> Option<u64> {
    if desc.length == 0 {
        warn!(target: LOG_TARGET, "reject create_buffer: length=0");
        return None;
    }
    let length = usize::try_from(desc.length).expect("buffer length fits host address space");

    // `Staged` VB/IB device buffer: Metal allocates `StorageModePrivate`
    // storage (no caller backing). It is written only by the
    // staging-upload blit and read by draws — never CPU-mapped — so
    // `newBufferWithBytesNoCopy` doesn't apply.
    if matches!(desc.kind, BufferKind::VbIbDevice) {
        let Some(buffer) =
            device.newBufferWithLength_options(length, MTLResourceOptions::StorageModePrivate)
        else {
            warn!(
                target: LOG_TARGET,
                "create_buffer: newBufferWithLength(Private) returned nil (id={:#x} len={})",
                desc.id, desc.length
            );
            return None;
        };
        let label =
            objc2_foundation::NSString::from_str(&format!("mtld3d-vbib-dev-{:#x}", desc.id));
        buffer.setLabel(Some(&label));
        return Some(Retained::into_raw(buffer) as u64);
    }

    if desc.backing_ptr == 0 {
        warn!(target: LOG_TARGET, "reject create_buffer: backing_ptr=0");
        return None;
    }
    let options = match desc.storage_mode {
        StorageMode::Shared => MTLResourceOptions::StorageModeShared,
        StorageMode::Managed => MTLResourceOptions::StorageModeManaged,
        StorageMode::Private | StorageMode::Memoryless => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "create_buffer: storage_mode={:?} incompatible with \
                 newBufferWithBytesNoCopy → Shared",
                desc.storage_mode
            );
            MTLResourceOptions::StorageModeShared
        }
    };
    let Some(pe_ptr) = NonNull::new(desc.backing_ptr as *mut c_void) else {
        warn!(
            target: LOG_TARGET,
            "create_buffer: backing_ptr null (kind={:?} id={:#x})",
            desc.kind, desc.id
        );
        return None;
    };

    // SAFETY: `pe_ptr` is PE-side `PageBox`-owned and lives ≥ `MTLBuffer`
    // lifetime; `length` matches its allocation; deallocator is `None` so
    // Metal will not free the PE allocation when the buffer is released.
    let Some(buffer) = (unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(pe_ptr, length, options, None)
    }) else {
        warn!(
            target: LOG_TARGET,
            "create_buffer: newBufferWithBytesNoCopy returned nil \
             (kind={:?} id={:#x} backing_ptr={:#x} len={} storage={:?})",
            desc.kind, desc.id, desc.backing_ptr, desc.length, desc.storage_mode
        );
        return None;
    };
    let prefix = match desc.kind {
        BufferKind::VbIb => "vbib",
        BufferKind::TexStaging => "tex-staging",
        BufferKind::Visibility => "vis",
        BufferKind::Repack => "repack",
        // Handled by the early-return Private path above; this arm only
        // satisfies match exhaustiveness.
        BufferKind::VbIbDevice => "vbib-dev",
    };
    let label = objc2_foundation::NSString::from_str(&format!("mtld3d-{prefix}-{:#x}", desc.id));
    buffer.setLabel(Some(&label));
    Some(Retained::into_raw(buffer) as u64)
}

pub fn destroy_buffer(buffer_handle: u64) {
    if buffer_handle == 0 {
        return;
    }
    // SAFETY: `buffer_handle` is the raw `u64` of a previously-retained
    // `MTLBuffer` (created via `into_raw` in `create_buffer`); adopting it
    // back into a `Retained` and dropping releases the canonical retain.
    unsafe {
        drop(Retained::from_raw(
            buffer_handle as *mut ProtocolObject<dyn MTLBuffer>,
        ));
    }
}
