//! Shared `D3DPOOL_DEFAULT` textures: the D3DKMT side of a `pSharedHandle`.
//!
//! An extended device's `CreateTexture` opens a texture a Direct3D 11 device
//! of the same process shares through a handle (`mtld3d_core::shared_texture`
//! has the rules and the record). This module makes the win32u calls behind
//! it:
//!
//! - [`open`] reads the record of the D3DKMT resource a handle names and has
//!   the unix side wrap the Metal texture the record's service name leads to.
//! - [`release`] gives back what a shared texture holds outside itself.
//!
//! Nothing synchronises the two runtimes' command queues beyond what Metal
//! does for a texture both use: a consumer sees what the producer had
//! submitted before the consumer's own commands were submitted. A producer
//! that needs a frame complete before it is read waits on its own event
//! query, which is what an application written for legacy shared handles
//! does.

use core::ffi::c_void;
use std::sync::LazyLock;

use log::{info, warn};
use mtld3d_core::{
    ids::TextureId,
    shared_texture::{RECORD_LEN, SERVICE_NAME_LEN, SharedTextureRecord, global_handle},
};
use mtld3d_shared::{
    MetalHandle, SHARED_TEXTURE_NAME_LEN, SharedTextureParams,
    mtl::{PixelFormat, SharedTextureOp, TextureUsage},
    mtl_handle::MTLDeviceKind,
    record_handle::DeviceRecordHandle,
};
use mtld3d_types::LUID;

use crate::{D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, LOG_TARGET, unix_call::unix_call};

const _: () = assert!(SHARED_TEXTURE_NAME_LEN >= SERVICE_NAME_LEN);

#[link(name = "gdi32")]
unsafe extern "system" {
    fn D3DKMTOpenAdapterFromLuid(desc: *mut D3dkmtOpenAdapterFromLuid) -> u32;
    fn D3DKMTCreateDevice(desc: *mut D3dkmtCreateDevice) -> u32;
    fn D3DKMTQueryResourceInfo(desc: *mut D3dkmtQueryResourceInfo) -> u32;
    fn D3DKMTOpenResource2(desc: *mut D3dkmtOpenResource) -> u32;
    fn D3DKMTDestroyAllocation(desc: *const D3dkmtDestroyAllocation) -> u32;
}

/// The texture a create that opens a share asks for, and the device it asks on.
pub struct ShareRequest {
    pub device_handle: MetalHandle<MTLDeviceKind>,
    pub record_handle: DeviceRecordHandle,
    pub texture_id: TextureId,
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    pub usage: TextureUsage,
}

/// Open the texture behind `slot`, the value the application's handle slot holds.
///
/// On success the texture's storage is parked on the unix side under the
/// request's texture id.
///
/// # Errors
///
/// The HRESULT of the create: `D3DERR_INVALIDCALL` for a value that names no
/// shared texture of the requested shape, `D3DERR_NOTAVAILABLE` for a resource
/// another runtime made or a Wine without D3DKMT resources.
pub fn open(request: &ShareRequest, slot: usize) -> Result<(), i32> {
    let tex_id = request.texture_id.raw();
    let (width, height) = (request.width, request.height);
    let Some(global) = global_handle(slot) else {
        warn!(target: LOG_TARGET,
            "CreateTexture: shared handle {slot:#x} is no D3DKMT global handle (D3D9 opens \
             neither an NT handle nor a pointer) → INVALIDCALL");
        return Err(D3DERR_INVALIDCALL);
    };
    let Some(device) = kmt_device() else {
        return Err(D3DERR_NOTAVAILABLE);
    };
    let record = query_record(device, global)?;
    if !record.describes_texture(width, height) {
        let (found_width, found_height) = record.extent();
        warn!(target: LOG_TARGET,
            "CreateTexture: shared handle {global:#x} is a dimension-{} {found_width}x\
             {found_height} array={} samples={} DXGI format {} resource, not the {width}x{height} \
             2D texture the create describes → INVALIDCALL",
            record.dimension(), record.array_size(), record.sample_count(), record.format());
        return Err(D3DERR_INVALIDCALL);
    }
    if record.keyed_mutex() != 0 {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "CreateTexture: shared handle {global:#x} carries a keyed mutex ({:#x}), which D3D9 \
             has no call to acquire; the texture is used without it",
            record.keyed_mutex()
        );
    }
    let mut name = [0; SHARED_TEXTURE_NAME_LEN];
    name[..SERVICE_NAME_LEN].copy_from_slice(record.service_name());
    if !open_metal_texture(request, name) {
        warn!(target: LOG_TARGET,
            "CreateTexture: the Metal texture behind shared handle {global:#x} could not be \
             opened (mtld3d::unix says why) → INVALIDCALL");
        return Err(D3DERR_INVALIDCALL);
    }
    info!(target: LOG_TARGET,
        "CreateTexture: opened shared handle {global:#x} as texture {tex_id:#x} \
         ({width}x{height}, DXGI format {}, bind {:#x}, misc {:#x}, mips {})",
        record.format(), record.bind_flags(), record.misc_flags(), record.mip_levels());
    Ok(())
}

/// Give back what a shared texture holds outside the texture itself.
///
/// Called once, at the final release of a texture whose device still lives:
/// the storage the encoder never adopted. A device that is gone took its
/// record, and with it whatever was parked, along.
pub fn release(
    device_handle: MetalHandle<MTLDeviceKind>,
    record_handle: DeviceRecordHandle,
    tex_id: u64,
) {
    let mut params = SharedTextureParams {
        device_handle,
        record_handle,
        tex_id,
        op: SharedTextureOp::Forget,
        width: 0,
        height: 0,
        pixel_format: PixelFormat::Bgra8Unorm,
        usage: TextureUsage::empty(),
        pad0: 0,
        name: [0; SHARED_TEXTURE_NAME_LEN],
    };
    let status = unix_call(&mut params);
    if status != 0 {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "shared textures: the unix side did not release the storage parked for texture \
             {tex_id:#x} (status {status:#x})"
        );
    }
}

/// Win32 `D3DKMT_OPENADAPTERFROMLUID`.
#[repr(C)]
struct D3dkmtOpenAdapterFromLuid {
    adapter_luid: LUID,
    adapter: u32,
}

/// Win32 `D3DKMT_CREATEDEVICE`; the leading union is pointer-sized.
#[repr(C)]
struct D3dkmtCreateDevice {
    adapter: usize,
    flags: u32,
    device: u32,
    command_buffer: *mut c_void,
    command_buffer_size: u32,
    allocation_list: *mut c_void,
    allocation_list_size: u32,
    patch_location_list: *mut c_void,
    patch_location_list_size: u32,
}

/// Win32 `D3DKMT_QUERYRESOURCEINFO`.
#[repr(C)]
struct D3dkmtQueryResourceInfo {
    device: u32,
    global_share: u32,
    private_runtime_data: *mut c_void,
    private_runtime_data_size: u32,
    total_private_driver_data_size: u32,
    resource_private_driver_data_size: u32,
    num_allocations: u32,
}

/// Win32 `D3DDDI_OPENALLOCATIONINFO2`.
#[repr(C)]
struct D3dddiOpenAllocationInfo2 {
    allocation: u32,
    private_driver_data: *const c_void,
    private_driver_data_size: u32,
    gpu_virtual_address: u64,
    reserved: [usize; 6],
}

/// Win32 `D3DKMT_OPENRESOURCE`.
#[repr(C)]
struct D3dkmtOpenResource {
    device: u32,
    global_share: u32,
    num_allocations: u32,
    open_allocation_info: *mut D3dddiOpenAllocationInfo2,
    private_runtime_data: *mut c_void,
    private_runtime_data_size: u32,
    resource_private_driver_data: *mut c_void,
    resource_private_driver_data_size: u32,
    total_private_driver_data_buffer: *mut c_void,
    total_private_driver_data_buffer_size: u32,
    resource: u32,
}

/// Win32 `D3DKMT_DESTROYALLOCATION`.
#[repr(C)]
struct D3dkmtDestroyAllocation {
    device: u32,
    resource: u32,
    allocation_list: *const u32,
    allocation_count: u32,
}

/// The process's D3DKMT device, `None` when win32u gives none.
///
/// Every D3DKMT resource call names a device, and win32u only checks that
/// the handle is one: resources are looked up by their own handles, whatever
/// device asks.
fn kmt_device() -> Option<u32> {
    /// The D3DKMT device handle, 0 when none could be opened.
    ///
    /// The resource is process-wide and latched once: a D3DKMT device is a
    /// handle win32u gives the process, not state of a D3D9 device, and one
    /// serves every shared open of every device. It and its adapter stay
    /// open for the life of the process.
    static DEVICE: LazyLock<u32> = LazyLock::new(open_kmt_device);
    (*DEVICE != 0).then_some(*DEVICE)
}

/// Open a D3DKMT adapter and a device on it; 0 when either fails, logged.
fn open_kmt_device() -> u32 {
    let adapter = match crate::direct3d9::open_primary_kmt_adapter() {
        Ok(adapter) if adapter.handle != 0 => adapter.handle,
        opened => {
            // Any adapter serves, since the device is only a name: one opened
            // by LUID is granted for every LUID where the display driver
            // hands none out for the primary display.
            let mut by_luid = D3dkmtOpenAdapterFromLuid {
                adapter_luid: LUID {
                    low_part: 0,
                    high_part: 0,
                },
                adapter: 0,
            };
            // SAFETY: gdi32 export; `by_luid` is a live struct of the
            // documented layout.
            let luid_status = unsafe { D3DKMTOpenAdapterFromLuid(&raw mut by_luid) };
            if luid_status != 0 || by_luid.adapter == 0 {
                warn!(target: LOG_TARGET,
                    "shared textures: no D3DKMT adapter opens (by display name: status {:#x}, \
                     by LUID: status {luid_status:#x}); no shared handle can be opened",
                    opened.err().unwrap_or(0));
                return 0;
            }
            by_luid.adapter
        }
    };
    let mut create = D3dkmtCreateDevice {
        adapter: usize::try_from(adapter).expect("u32 fits usize"),
        flags: 0,
        device: 0,
        command_buffer: core::ptr::null_mut(),
        command_buffer_size: 0,
        allocation_list: core::ptr::null_mut(),
        allocation_list_size: 0,
        patch_location_list: core::ptr::null_mut(),
        patch_location_list_size: 0,
    };
    // SAFETY: gdi32 export; `create` names the adapter just opened.
    let status = unsafe { D3DKMTCreateDevice(&raw mut create) };
    if status != 0 || create.device == 0 {
        warn!(target: LOG_TARGET,
            "shared textures: D3DKMTCreateDevice failed (status {status:#x}); no shared handle \
             can be opened");
        return 0;
    }
    info!(target: LOG_TARGET,
        "shared textures: D3DKMT device {:#x} on adapter {adapter:#x}", create.device);
    create.device
}

/// Read the record of the shared resource `global` names.
///
/// `Err` is the HRESULT of the create that asked.
fn query_record(device: u32, global: u32) -> Result<SharedTextureRecord, i32> {
    let mut bytes = [0u8; RECORD_LEN];
    let record_len = u32::try_from(RECORD_LEN).expect("the record length fits u32");
    let mut query = D3dkmtQueryResourceInfo {
        device,
        global_share: global,
        private_runtime_data: bytes.as_mut_ptr().cast(),
        private_runtime_data_size: record_len,
        total_private_driver_data_size: 0,
        resource_private_driver_data_size: 0,
        num_allocations: 0,
    };
    // SAFETY: gdi32 export; `query` is a live struct of the documented layout
    // whose runtime-data pointer names `bytes`, live for the call.
    let status = unsafe { D3DKMTQueryResourceInfo(&raw mut query) };
    if status != 0 {
        warn!(target: LOG_TARGET,
            "CreateTexture: shared handle {global:#x} names no shared resource (query status \
             {status:#x}) → INVALIDCALL");
        return Err(D3DERR_INVALIDCALL);
    }
    if query.private_runtime_data_size != record_len {
        warn!(target: LOG_TARGET,
            "CreateTexture: shared handle {global:#x} carries {} bytes of runtime data, not the \
             {record_len} of a shared texture record (another Direct3D runtime made it) → \
             NOTAVAILABLE",
            query.private_runtime_data_size);
        return Err(D3DERR_NOTAVAILABLE);
    }
    let mut allocation = D3dddiOpenAllocationInfo2 {
        allocation: 0,
        private_driver_data: core::ptr::null(),
        private_driver_data_size: 0,
        gpu_virtual_address: 0,
        reserved: [0; 6],
    };
    let mut open = D3dkmtOpenResource {
        device,
        global_share: global,
        num_allocations: 1,
        open_allocation_info: &raw mut allocation,
        private_runtime_data: bytes.as_mut_ptr().cast(),
        private_runtime_data_size: record_len,
        resource_private_driver_data: core::ptr::null_mut(),
        resource_private_driver_data_size: 0,
        total_private_driver_data_buffer: core::ptr::null_mut(),
        total_private_driver_data_buffer_size: 0,
        resource: 0,
    };
    // SAFETY: gdi32 export; every pointer of `open` names a live local.
    let status = unsafe { D3DKMTOpenResource2(&raw mut open) };
    if status != 0 {
        warn!(target: LOG_TARGET,
            "CreateTexture: shared handle {global:#x} does not open (status {status:#x}) → \
             INVALIDCALL");
        return Err(D3DERR_INVALIDCALL);
    }
    // The record is all the open is for: the texture's storage is kept alive
    // by the Metal texture, not by this D3DKMT reference.
    destroy_resource(device, open.resource);
    Ok(SharedTextureRecord::from_bytes(&bytes))
}

fn destroy_resource(device: u32, resource: u32) {
    if resource == 0 {
        return;
    }
    let destroy = D3dkmtDestroyAllocation {
        device,
        resource,
        allocation_list: core::ptr::null(),
        allocation_count: 0,
    };
    // SAFETY: gdi32 export; `destroy` names a resource this module opened on
    // `device`.
    let status = unsafe { D3DKMTDestroyAllocation(&raw const destroy) };
    if status != 0 {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "shared textures: D3DKMTDestroyAllocation({resource:#x}) failed (status \
             {status:#x}); the resource stays until the process ends"
        );
    }
}

/// Have the unix side open the Metal texture registered under `name`; whether it did.
fn open_metal_texture(request: &ShareRequest, name: [u8; SHARED_TEXTURE_NAME_LEN]) -> bool {
    let mut params = SharedTextureParams {
        device_handle: request.device_handle,
        record_handle: request.record_handle,
        tex_id: request.texture_id.raw(),
        op: SharedTextureOp::Open,
        width: request.width,
        height: request.height,
        pixel_format: request.pixel_format,
        usage: request.usage,
        pad0: 0,
        name,
    };
    unix_call(&mut params) == 0
}
