//! Unit tests for the process-pinned Metal device.
//!
//! The pin is what makes the process-wide Metal caches sound: they hold
//! pipeline states, textures and buffers built on one `MTLDevice`, and Metal
//! rejects an encode that mixes objects of two. Every Mac these tests run on
//! has one GPU, so the assertions pin the invariant rather than reproduce the
//! dual-GPU divergence, which needs graphics switching to observe.

use objc2_metal::MTLDevice;

use super::{DeviceCaps, create_command_queue, default_device_info};
use crate::metal::handle::{IntoRetained, ReleaseRetain};

/// Drops the retains one `DeviceCaps` handed out, as the destroy thunk does.
///
/// The caller must be done with both handles, and no copy of either may be
/// used afterwards.
fn release(caps: &DeviceCaps) {
    // SAFETY: this stands in for `destroy_command_queue`. The queue handle
    // carries the only retain on a queue nothing else names.
    unsafe { caps.queue_handle.release_retain() };
    // SAFETY: the device handle carries the retain `create_command_queue`
    // took for this D3D device, not the pin's own.
    unsafe { caps.device_handle.release_retain() };
}

/// Two `create_command_queue` calls hand out one device, and the caps agree with it.
///
/// The second call must name the same `id<MTLDevice>` as the first, and
/// `default_device_info`, which answers the PE side's `GetDeviceInfo`, must
/// report that device's registry id rather than a second resolution's. The
/// third call proves the pin keeps a retain of its own: the device is still
/// live after both handed-out retains are dropped.
#[test]
fn create_command_queue_hands_out_the_pinned_device() {
    let Some(first) = create_command_queue() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let second = create_command_queue().expect("a second queue on the pinned device");
    assert_eq!(
        first.device_handle.raw(),
        second.device_handle.raw(),
        "every D3D device gets the same MTLDevice"
    );
    assert_ne!(
        first.queue_handle.raw(),
        second.queue_handle.raw(),
        "each D3D device gets its own MTLCommandQueue"
    );

    let (_, registry_id, _) = default_device_info().expect("caps for the pinned device");
    let device = first
        .device_handle
        .into_retained()
        .expect("the handed-out device handle is live");
    assert_eq!(
        device.registryID(),
        registry_id,
        "the caps answer describes the device the queues were made on"
    );
    drop(device);

    release(&first);
    release(&second);

    let third = create_command_queue().expect("a queue after both devices were destroyed");
    let device = third
        .device_handle
        .into_retained()
        .expect("the pinned device outlives the handles it was handed out through");
    assert_eq!(device.registryID(), registry_id);
    drop(device);
    release(&third);
}
