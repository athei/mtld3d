//! Shared textures: one `MTLTexture` storage two Direct3D runtimes of a process use.
//!
//! An extended D3D9 device shares a `D3DPOOL_DEFAULT` texture with a Direct3D
//! 11 device through the `pSharedHandle` argument of `CreateTexture`. On an
//! arm64 Wine the Direct3D 11 runtime is DXMT, which makes a shared texture
//! with `newSharedTextureWithDescriptor:` and names it to other runtimes and
//! processes by the Mach port of its `MTLSharedTextureHandle`, registered with
//! the bootstrap server under a unique service name. The name travels in the
//! private runtime data of the D3DKMT resource whose global handle the
//! application passes around (`mtld3d_core::shared_texture`). This module is
//! the Metal half of that protocol, in both directions:
//!
//! - [`open`] takes the service name of a texture the other runtime
//!   allocated, looks the port up and wraps the same storage in an
//!   `MTLTexture` of this runtime.
//! - [`create`] allocates the storage here and registers its port, for a
//!   texture the other runtime imports with `OpenSharedResource`.
//!
//! Either way the texture exists before `CreateTexture` returns, because the
//! call has to answer with a handle or a failure, while every other texture
//! of this layer is created by the encoder when its creation record arrives.
//! So the texture waits in the device record's [`ParkedTextures`] under its
//! mtld3d texture id, and the encoder adopts it there when a descriptor
//! carries `TextureCreateFlags::SHARED`.
//!
//! `initWithMachPort:` and `createMachPort` are not in Metal's headers: they
//! are how a shared texture handle crosses to a runtime that shares no
//! `NSXPCConnection` with this one, and the selectors the other runtime uses.
//! [`MachPortHandle`] declares them, and both are probed with
//! `respondsToSelector:` before use, so a macOS without them refuses the
//! share instead of raising.

use std::{
    ffi::{CStr, c_char, c_int},
    sync::{Mutex, PoisonError},
    time::{SystemTime, UNIX_EPOCH},
};

use libc::{kern_return_t, mach_port_t};
use mtld3d_core::shared_texture::{SERVICE_NAME_LEN, service_name};
use mtld3d_shared::{
    MetalHandle, SHARED_TEXTURE_NAME_LEN, SharedTextureParams,
    mtl::{PixelFormat, SharedTextureOp, TextureUsage},
    mtl_handle::MTLTextureKind,
};
use objc2::{
    AnyThread, ClassType, extern_methods,
    rc::{Allocated, Retained},
    runtime::{NSObjectProtocol, ProtocolObject, Sel},
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLDevice, MTLPixelFormat, MTLResource, MTLSharedTextureHandle, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use super::{
    handle::{IntoRetained, ReleaseRetain},
    texture::mtl_pixel_format,
};
use crate::LOG_TARGET;

/// `TASK_BOOTSTRAP_PORT` of `<mach/task_special_ports.h>`.
const TASK_BOOTSTRAP_PORT: c_int = 4;

const _: () = assert!(SHARED_TEXTURE_NAME_LEN >= SERVICE_NAME_LEN);

unsafe extern "C" {
    /// This task's own port name, the value `mach_task_self()` expands to.
    static mach_task_self_: mach_port_t;
    fn task_get_special_port(
        task: mach_port_t,
        which: c_int,
        port: *mut mach_port_t,
    ) -> kern_return_t;
    fn mach_port_deallocate(task: mach_port_t, name: mach_port_t) -> kern_return_t;
    fn bootstrap_look_up(
        bootstrap: mach_port_t,
        name: *const c_char,
        port: *mut mach_port_t,
    ) -> kern_return_t;
    /// Register `port` under `name`; exported by libSystem, absent from the SDK headers.
    fn bootstrap_register2(
        bootstrap: mach_port_t,
        name: *const c_char,
        port: mach_port_t,
        flags: c_int,
    ) -> kern_return_t;
}

/// The shared textures of one device that its encoder has not adopted yet.
///
/// The thunk that fills it runs on the API thread inside `CreateTexture`,
/// before the texture's creation record is submitted, and the encoder drains
/// it by the texture id both sides know. Each entry owns one retain;
/// [`Self::take`] hands it to the encoder, [`Self::forget`] releases it for a
/// texture released before any frame carried its creation, and dropping the
/// record releases whatever is left.
pub struct ParkedTextures {
    parked: Mutex<Vec<(u64, MetalHandle<MTLTextureKind>)>>,
}

impl ParkedTextures {
    pub const fn new() -> Self {
        Self {
            parked: Mutex::new(Vec::new()),
        }
    }

    /// Hand the texture parked under `tex_id` to the caller, with its retain.
    pub fn take(&self, tex_id: u64) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
        let handle = self.remove(tex_id)?;
        // SAFETY: the handle holds the retain `park` moved into it and has
        // just left the table, so the retain moves on into the `Retained`.
        unsafe { Retained::from_raw(handle.raw() as *mut ProtocolObject<dyn MTLTexture>) }
    }

    /// Release the texture parked under `tex_id`, if the encoder never took it.
    pub fn forget(&self, tex_id: u64) {
        if let Some(handle) = self.remove(tex_id) {
            // SAFETY: the handle holds the retain `park` moved into it and
            // has just left the table, so this is that retain's one release.
            unsafe { handle.release_retain() };
        }
    }

    fn park(&self, tex_id: u64, texture: Retained<ProtocolObject<dyn MTLTexture>>) {
        // SAFETY: the raw value is the retained `id<MTLTexture>` the
        // `Retained` owned, which the handle owns from here on.
        let handle = unsafe { MetalHandle::new(Retained::into_raw(texture) as u64) };
        // Texture ids are unique in the process, so an id is parked once; a
        // second park under one id would be a caller's error and replaces
        // the first rather than leaving two entries a take cannot tell apart.
        let replaced = self.remove(tex_id);
        self.lock().push((tex_id, handle));
        if let Some(replaced) = replaced {
            log::error!(target: LOG_TARGET,
                "shared texture {tex_id:#x}: parked twice; the first texture is released");
            // SAFETY: as in `forget`.
            unsafe { replaced.release_retain() };
        }
    }

    fn remove(&self, tex_id: u64) -> Option<MetalHandle<MTLTextureKind>> {
        let mut parked = self.lock();
        let index = parked.iter().position(|(id, _)| *id == tex_id)?;
        Some(parked.swap_remove(index).1)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(u64, MetalHandle<MTLTextureKind>)>> {
        self.parked.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Drop for ParkedTextures {
    fn drop(&mut self) {
        for (_, handle) in self.lock().drain(..) {
            // SAFETY: as in `forget`; the table is going away with its retains.
            unsafe { handle.release_retain() };
        }
    }
}

/// Serve one `SharedTexture` thunk for the device whose record owns `parked`.
///
/// `false` is the thunk's failure: the create or open made no texture, and
/// its reason is already in the log.
pub fn serve(parked: &ParkedTextures, params: &mut SharedTextureParams) -> bool {
    if params.op == SharedTextureOp::Forget {
        parked.forget(params.tex_id);
        return true;
    }
    let Some(device) = params.device_handle.into_retained() else {
        log::error!(target: LOG_TARGET,
            "shared texture {:#x}: the thunk names no Metal device", params.tex_id);
        return false;
    };
    let request = ShareRequest::of(params);
    if params.op == SharedTextureOp::Open {
        let mut name = [0; SERVICE_NAME_LEN];
        name.copy_from_slice(&params.name[..SERVICE_NAME_LEN]);
        return open(&device, parked, &request, &name);
    }
    let Some(name) = create(&device, parked, &request) else {
        return false;
    };
    params.name = [0; SHARED_TEXTURE_NAME_LEN];
    params.name[..SERVICE_NAME_LEN].copy_from_slice(&name);
    true
}

/// The Mach-port methods of `MTLSharedTextureHandle`, which Metal's headers leave out.
///
/// # Safety
///
/// Implemented for `MTLSharedTextureHandle` alone, whose instances answer
/// these selectors with these signatures where they answer them at all.
unsafe trait MachPortHandle: ClassType + Sized {
    extern_methods!(
        /// The handle of the shared texture `port` names, `None` when it names none.
        ///
        /// # Safety
        ///
        /// The class responds to `initWithMachPort:`, and `port` is a send
        /// right of this task.
        #[unsafe(method(initWithMachPort:))]
        #[unsafe(method_family = init)]
        unsafe fn init_with_mach_port(
            this: Allocated<Self>,
            port: mach_port_t,
        ) -> Option<Retained<Self>>;

        /// A send right that names the handle's texture to any task that receives it.
        ///
        /// # Safety
        ///
        /// The handle responds to `createMachPort`.
        #[unsafe(method(createMachPort))]
        #[unsafe(method_family = none)]
        unsafe fn create_mach_port(&self) -> mach_port_t;
    );
}

// SAFETY: the trait's contract names this class.
unsafe impl MachPortHandle for MTLSharedTextureHandle {}

/// The texture a create or an open is asked for.
struct ShareRequest {
    tex_id: u64,
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
    /// The usage of the D3D9 texture, which an opened texture has to allow.
    usage: TextureUsage,
}

impl ShareRequest {
    const fn of(params: &SharedTextureParams) -> Self {
        Self {
            tex_id: params.tex_id,
            width: params.width,
            height: params.height,
            pixel_format: params.pixel_format,
            usage: params.usage,
        }
    }
}

/// Allocate a shared texture, register its port and park it under the request's id.
///
/// Returns the NUL-terminated service name the port is registered under.
fn create(
    device: &ProtocolObject<dyn MTLDevice>,
    parked: &ParkedTextures,
    request: &ShareRequest,
) -> Option<[u8; SERVICE_NAME_LEN]> {
    let ShareRequest {
        tex_id,
        width,
        height,
        pixel_format,
        ..
    } = *request;
    // SAFETY: objc2 typed binding; class-method constructor returning a fresh
    // autoreleased descriptor.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            mtl_pixel_format(pixel_format),
            usize::try_from(width).expect("u32 fits usize"),
            usize::try_from(height).expect("u32 fits usize"),
            false,
        )
    };
    // A render target that is also sampled, whatever the D3D9 usage: that is
    // how the record describes the texture to the runtime that opens it.
    desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget);
    desc.setStorageMode(MTLStorageMode::Private);
    let Some(texture) = device.newSharedTextureWithDescriptor(&desc) else {
        log::error!(target: LOG_TARGET,
            "shared texture {tex_id:#x}: newSharedTextureWithDescriptor returned nil for \
             {width}x{height} {pixel_format:?}");
        return None;
    };
    texture.setLabel(Some(&NSString::from_str(&format!(
        "mtld3d-shared-{tex_id:#x}"
    ))));
    let Some(handle) = texture.newSharedTextureHandle() else {
        log::error!(target: LOG_TARGET,
            "shared texture {tex_id:#x}: newSharedTextureHandle returned nil");
        return None;
    };
    if !handle.respondsToSelector(Sel::register(c"createMachPort")) {
        log::error!(target: LOG_TARGET,
            "shared texture {tex_id:#x}: this macOS's MTLSharedTextureHandle has no \
             createMachPort, so the texture cannot be named to another runtime");
        return None;
    }
    // SAFETY: the handle responds to the selector, probed above.
    let port = unsafe { handle.create_mach_port() };
    if port == 0 {
        log::error!(target: LOG_TARGET, "shared texture {tex_id:#x}: createMachPort returned 0");
        return None;
    }
    let ticks = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| u64::try_from(since.as_micros()).unwrap_or(0));
    let name = service_name(std::process::id(), tex_id, ticks);
    let Some(bootstrap) = bootstrap_port() else {
        log::error!(target: LOG_TARGET, "shared texture {tex_id:#x}: no bootstrap port");
        deallocate(port);
        return None;
    };
    // SAFETY: `name` is NUL-terminated (`service_name` guarantees it) and
    // both ports are live send rights of this task.
    let kr = unsafe { bootstrap_register2(bootstrap, name.as_ptr().cast(), port, 0) };
    deallocate(bootstrap);
    // The bootstrap server holds a right of its own from here on, which is
    // what keeps the name resolving; this task's copy is not needed again.
    deallocate(port);
    if kr != 0 {
        log::error!(target: LOG_TARGET,
            "shared texture {tex_id:#x}: bootstrap_register2 failed ({kr:#x})");
        return None;
    }
    log::info!(target: LOG_TARGET,
        "shared texture {tex_id:#x}: created {width}x{height} {pixel_format:?} as {:?} \
         (registry id {:#x})",
        name_str(&name), device.registryID());
    parked.park(tex_id, texture);
    Some(name)
}

/// Wrap the shared texture registered under `name` and park it under the request's id.
///
/// Refuses a texture whose shape is not the 2D, single-level, single-sample
/// texture of the extent and format the D3D9 create described, since every
/// later use of the D3D9 texture is sized from those arguments, and one whose
/// usage does not allow what the D3D9 usage asks of it.
fn open(
    device: &ProtocolObject<dyn MTLDevice>,
    parked: &ParkedTextures,
    request: &ShareRequest,
    name: &[u8; SERVICE_NAME_LEN],
) -> bool {
    let ShareRequest {
        tex_id,
        width,
        height,
        pixel_format,
        usage,
    } = *request;
    let Ok(service) = CStr::from_bytes_until_nul(name) else {
        log::error!(target: LOG_TARGET, "shared texture {tex_id:#x}: unterminated service name");
        return false;
    };
    let Some(bootstrap) = bootstrap_port() else {
        log::error!(target: LOG_TARGET, "shared texture {tex_id:#x}: no bootstrap port");
        return false;
    };
    let mut port: mach_port_t = 0;
    // SAFETY: `service` is NUL-terminated and `port` a writable out slot.
    let kr = unsafe { bootstrap_look_up(bootstrap, service.as_ptr(), &raw mut port) };
    deallocate(bootstrap);
    if kr != 0 || port == 0 {
        log::error!(target: LOG_TARGET,
            "shared texture {tex_id:#x}: bootstrap_look_up({service:?}) failed ({kr:#x})");
        return false;
    }
    if !MTLSharedTextureHandle::class().responds_to(Sel::register(c"initWithMachPort:")) {
        log::error!(target: LOG_TARGET,
            "shared texture {tex_id:#x}: this macOS's MTLSharedTextureHandle has no \
             initWithMachPort:, so a texture of another runtime cannot be opened");
        deallocate(port);
        return false;
    }
    let allocated = MTLSharedTextureHandle::alloc();
    // SAFETY: instances respond to the selector, probed above, and `port` is
    // the send right the lookup returned.
    let handle = unsafe { MTLSharedTextureHandle::init_with_mach_port(allocated, port) };
    let texture = handle
        .as_deref()
        .and_then(|handle| device.newSharedTextureWithHandle(handle));
    drop(handle);
    // The send right the lookup returned is left alone from here on. Whether
    // the handle took it over is not documented, and releasing a right the
    // handle already released could drop a reference that belongs to another
    // user of the same port name. Kept, it costs one user reference on a port
    // this task holds anyway.
    let Some(texture) = texture else {
        log::error!(target: LOG_TARGET,
            "shared texture {tex_id:#x}: no texture behind {service:?} on device registry id \
             {:#x} (another GPU, or the port is not a shared texture's)",
            device.registryID());
        return false;
    };
    let found = (
        texture.width(),
        texture.height(),
        texture.pixelFormat(),
        texture.textureType(),
        texture.mipmapLevelCount(),
        texture.sampleCount(),
    );
    let wanted: (usize, usize, MTLPixelFormat, MTLTextureType, usize, usize) = (
        usize::try_from(width).expect("u32 fits usize"),
        usize::try_from(height).expect("u32 fits usize"),
        mtl_pixel_format(pixel_format),
        MTLTextureType::Type2D,
        1,
        1,
    );
    let mut needed = MTLTextureUsage::ShaderRead;
    if usage.contains(TextureUsage::RENDER_TARGET) {
        needed |= MTLTextureUsage::RenderTarget;
    }
    if found != wanted || !texture.usage().contains(needed) {
        log::error!(target: LOG_TARGET,
            "shared texture {tex_id:#x}: {service:?} is {}x{} {:?} {:?} levels={} samples={} \
             usage={:#x}, not the {width}x{height} {pixel_format:?} 2D texture of usage {:#x} \
             the create describes",
            found.0, found.1, found.2, found.3, found.4, found.5, texture.usage().0, needed.0);
        return false;
    }
    log::info!(target: LOG_TARGET,
        "shared texture {tex_id:#x}: opened {service:?} as {width}x{height} {pixel_format:?} \
         usage={:#x} storage={} (registry id {:#x})",
        texture.usage().0, texture.storageMode().0, device.registryID());
    parked.park(tex_id, texture);
    true
}

/// The task's bootstrap port, a send right the caller deallocates.
fn bootstrap_port() -> Option<mach_port_t> {
    let mut port: mach_port_t = 0;
    // SAFETY: a word libSystem sets once at start and never writes again.
    let task = unsafe { mach_task_self_ };
    // SAFETY: Mach call on this task with a writable out slot.
    let kr = unsafe { task_get_special_port(task, TASK_BOOTSTRAP_PORT, &raw mut port) };
    (kr == 0 && port != 0).then_some(port)
}

fn deallocate(port: mach_port_t) {
    // SAFETY: a word libSystem sets once at start and never writes again.
    let task = unsafe { mach_task_self_ };
    // SAFETY: `port` is a send right this module received and owns.
    let kr = unsafe { mach_port_deallocate(task, port) };
    if kr != 0 {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "shared texture: mach_port_deallocate failed ({kr:#x}); one port right stays held"
        );
    }
}

fn name_str(name: &[u8; SERVICE_NAME_LEN]) -> &str {
    CStr::from_bytes_until_nul(name)
        .ok()
        .and_then(|name| name.to_str().ok())
        .unwrap_or("?")
}

#[cfg(test)]
mod tests;
