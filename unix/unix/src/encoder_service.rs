//! Device-local native encoder ownership and lifecycle handlers.
//!
//! The API lock serializes access to the opaque allocation. Native workers own their
//! messages and complete before destruction returns to the PE allocation owner.

use std::{path::PathBuf, sync::Arc};

use mtld3d_core::{
    config::Mtld3dConfig, encoder_value::WireValue, gpu_caps::GpuCaps,
    shader_prewarm::PrewarmHandle,
};
use mtld3d_shared::{
    InPtrMut, MetalHandle,
    encoder_protocol::{EncoderControl, EncoderSubmitMode},
    encoder_runtime::{CONFIG_RECORD, CreateEncoderParams, DestroyEncoderParams},
    encoder_wire::{WireError, WireReader},
    mtl_handle::MTLDeviceKind,
};
use mtld3d_types::{D3D_OK, D3DERR_INVALIDCALL, E_OUTOFMEMORY};

use crate::{
    LOG_TARGET,
    encoder::{EncoderThread, FrameData},
    shader_prewarm,
    shader_programs::ProgramRegistry,
};

#[cfg(test)]
mod tests;

/// Native allocation belonging to exactly one PE device.
pub struct EncoderService {
    pub encoder: EncoderThread,
    pub programs: Arc<ProgramRegistry>,
    prewarm: PrewarmHandle,
    failure_ptr: u64,
    calibration: CalibrationWorker,
}

impl EncoderService {
    /// Start native workers without publishing a partially initialized handle.
    ///
    /// # Errors
    ///
    /// Returns the OS error when an encoder or submit worker cannot start.
    pub fn new(
        device: MetalHandle<MTLDeviceKind>,
        config: Mtld3dConfig,
        caps: GpuCaps,
        cache_path: Option<PathBuf>,
        source_clock_ptr: u64,
    ) -> std::io::Result<Self> {
        let calibration = CalibrationWorker::start();
        let clocks = EncoderClocks {
            native: Arc::clone(&calibration.clock),
            #[cfg(perf_tracking)]
            source: source_clock_ptr,
        };
        #[cfg(not(perf_tracking))]
        let _ = source_clock_ptr;
        let config = Arc::new(config);
        let (mut prewarm, receiver) =
            shader_prewarm::spawn(device, config.shader_cache_enable, cache_path.clone());
        match EncoderThread::spawn(caps, Arc::clone(&config), receiver, cache_path, clocks) {
            Ok(encoder) => Ok(Self {
                encoder,
                programs: Arc::new(ProgramRegistry::new()),
                prewarm,
                failure_ptr: 0,
                calibration,
            }),
            Err(error) => {
                prewarm.cancel_and_join();
                Err(error)
            }
        }
    }

    /// Borrow the sole device-owned runtime during a serialized API call.
    ///
    /// # Safety
    ///
    /// `runtime` must be a nonzero handle returned by creation, still owned by the
    /// caller's device. Its API lock must exclude destruction for the borrow's lifetime.
    pub const unsafe fn from_handle<'a>(runtime: u64) -> &'a Self {
        // SAFETY: the caller guarantees the live service and exclusive teardown ownership.
        unsafe { &*(runtime as *const Self) }
    }
}

impl Drop for EncoderService {
    fn drop(&mut self) {
        self.prewarm.cancel_and_join();
        self.calibration.join();
        self.encoder.shutdown();
    }
}

/// Device-owned calibration uses this native linkage unit's clock domain.
///
/// Startup never waits for the calibration sleep; shutdown joins the worker so
/// neither device teardown nor library unloading leaves native code running.
pub struct EncoderClocks {
    pub native: Arc<mtld3d_shared::clock_calibration::ClockCalibration>,
    #[cfg(perf_tracking)]
    pub source: u64,
}

struct CalibrationWorker {
    clock: Arc<mtld3d_shared::clock_calibration::ClockCalibration>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl CalibrationWorker {
    fn start() -> Self {
        Self::spawn(mtld3d_shared::tsc::tsc_hz)
    }

    fn spawn(work: impl FnOnce() -> u64 + Send + 'static) -> Self {
        let clock = Arc::new(mtld3d_shared::clock_calibration::ClockCalibration::new());
        let published = Arc::clone(&clock);
        let result = std::thread::Builder::new()
            .name("mtld3d-native-tsc-warmup".into())
            .spawn(move || {
                struct PublishFailure(Arc<mtld3d_shared::clock_calibration::ClockCalibration>);
                impl Drop for PublishFailure {
                    fn drop(&mut self) {
                        if matches!(self.0.get(), Ok(None)) {
                            // SAFETY: this worker is the mailbox's only publisher.
                            unsafe {
                                self.0.publish_failed();
                            }
                        }
                    }
                }
                let publication = PublishFailure(published);
                let hz = work();
                // SAFETY: this worker is the mailbox's only publisher.
                unsafe {
                    publication.0.publish_ready(hz);
                }
            });
        let handle = match result {
            Ok(handle) => Some(handle),
            Err(error) => {
                // SAFETY: a failed spawn never created a competing publisher.
                unsafe {
                    clock.publish_failed();
                }
                log::error!(target: LOG_TARGET, "perf-invalid: native calibration worker could not start: {error}; rendering continues");
                None
            }
        };
        Self { clock, handle }
    }

    fn join(&mut self) {
        if let Some(handle) = self.handle.take()
            && handle.join().is_err()
        {
            log::error!(target: LOG_TARGET, "perf-invalid: native clock calibration worker panicked; rendering continues");
        }
    }
}

impl Drop for CalibrationWorker {
    fn drop(&mut self) {
        self.join();
    }
}

fn decode_settings(bytes: &[u8]) -> Result<(Mtld3dConfig, GpuCaps, Option<PathBuf>), WireError> {
    let mut outer = WireReader::new(bytes);
    let mut record = outer.next_record()?.ok_or(WireError::Truncated)?;
    if record.tag != CONFIG_RECORD || !outer.is_empty() {
        return Err(WireError::InvalidValue);
    }
    let config = Mtld3dConfig::read_wire(&mut record.payload)?;
    let caps = GpuCaps::read_wire(&mut record.payload)?;
    let path = Option::<String>::read_wire(&mut record.payload)?.map(PathBuf::from);
    if !record.payload.is_empty() {
        return Err(WireError::InvalidValue);
    }
    Ok((config, caps, path))
}

pub extern "C" fn create_handler(args: *mut core::ffi::c_void) -> i32 {
    // SAFETY: the dispatcher supplies the matching exclusive parameter record.
    let Some(mut params) = (unsafe { InPtrMut::<CreateEncoderParams>::opt(args) }) else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "encoder: null creation parameters");
        return D3DERR_INVALIDCALL;
    };
    params.runtime = 0;
    params.result = D3DERR_INVALIDCALL;
    if params.device.is_null()
        || params.failure_ptr == 0
        || params.failure_ptr % 4 != 0
        || (cfg!(perf_tracking)
            && (params.source_clock_ptr == 0 || params.source_clock_ptr % 8 != 0))
        || params.config_ptr == 0
        || params.config_len == 0
        || params
            .config_ptr
            .checked_add(u64::from(params.config_len))
            .is_none()
    {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "encoder: invalid creation input");
        return params.result;
    }
    // SAFETY: creation retains the immutable config bytes for this synchronous call;
    // the nonempty range was checked above and contains only byte-aligned values.
    let bytes = unsafe {
        std::slice::from_raw_parts(params.config_ptr as *const u8, params.config_len as usize)
    };
    let (config, caps, path) = match decode_settings(bytes) {
        Ok(settings) => settings,
        Err(error) => {
            log::error!(target: LOG_TARGET, "encoder: invalid resolved configuration: {error:?}");
            return params.result;
        }
    };
    match EncoderService::new(params.device, config, caps, path, params.source_clock_ptr) {
        Ok(mut service) => {
            service.failure_ptr = params.failure_ptr;
            params.runtime = Box::into_raw(Box::new(service)) as u64;
            params.result = D3D_OK;
        }
        Err(error) => {
            log::error!(target: LOG_TARGET, "encoder: native worker startup failed: {error}");
            params.result = E_OUTOFMEMORY;
        }
    }
    params.result
}

pub extern "C" fn destroy_handler(args: *mut core::ffi::c_void) -> i32 {
    // SAFETY: the dispatcher supplies the matching exclusive parameter record.
    let Some(mut params) = (unsafe { InPtrMut::<DestroyEncoderParams>::opt(args) }) else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "encoder: null destruction parameters");
        return D3DERR_INVALIDCALL;
    };
    if params.runtime == 0 {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "encoder: null runtime destruction");
        return D3DERR_INVALIDCALL;
    }
    let runtime = std::mem::take(&mut params.runtime);
    // SAFETY: this is the sole device owner's creation allocation; its API lock
    // excludes concurrent calls and the zeroed parameter prevents reuse by this call.
    drop(unsafe { Box::from_raw(runtime as *mut EncoderService) });
    D3D_OK
}

/// Borrowed packet addresses held by the native queue until replay completes.
pub struct EncodedFrame {
    metadata_ptr: u64,
    operations_ptr: u64,
    metadata_len: u32,
    operations_len: u32,
    completion: u64,
    pub mode: EncoderSubmitMode,
    pub failure_ptr: u64,
    programs: Arc<ProgramRegistry>,
}

impl EncodedFrame {
    /// Publish native admission/replay failure before acknowledging the frame.
    pub fn report_failure(&self) {
        // SAFETY: admission retains the device mailbox until encoder shutdown.
        unsafe {
            publish_failure(self.failure_ptr);
        }
    }

    /// Decode only on the native encoder thread while the PE packet lease remains live.
    ///
    /// # Safety
    ///
    /// Creation must come from an admitted request whose packet owner retains its buffers
    /// and completion mailbox until decoder rejection or final replay completion.
    pub unsafe fn decode(
        &self,
        queries: &mut mtld3d_core::guest_queries::QueryLeaseCache,
    ) -> Result<FrameData, WireError> {
        // SAFETY: the admitted packet contract retains this immutable byte range.
        let metadata = unsafe {
            std::slice::from_raw_parts(self.metadata_ptr as *const u8, self.metadata_len as usize)
        };
        // SAFETY: the admitted packet contract retains this immutable byte range.
        let operations = unsafe {
            std::slice::from_raw_parts(
                self.operations_ptr as *const u8,
                self.operations_len as usize,
            )
        };
        // SAFETY: both ranges and every lease descriptor remain retained by the PE packet.
        unsafe {
            mtld3d_core::encoder_packet::decode_packet(
                metadata,
                operations,
                self.completion,
                queries,
                |registration| {
                    self.programs
                        .take(registration)
                        .ok_or(WireError::InvalidValue)
                },
            )
        }
    }
}

pub extern "C" fn submit_handler(args: *mut core::ffi::c_void) -> i32 {
    use mtld3d_shared::encoder_runtime::SubmitEncoderFrameParams;
    // SAFETY: the dispatcher supplies this request's exclusive typed record.
    let Some(mut params) = (unsafe { InPtrMut::<SubmitEncoderFrameParams>::opt(args) }) else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "encoder: null submission parameters");
        return D3DERR_INVALIDCALL;
    };
    params.admitted = 0;
    let Ok(mode) = EncoderSubmitMode::try_from(params.mode) else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "encoder: invalid submit mode");
        return D3DERR_INVALIDCALL;
    };
    if params.runtime == 0
        || params.metadata_ptr == 0
        || params.metadata_len == 0
        || params.operations_ptr == 0
        || params.completion == 0
        || params.completion % 8 != 0
        || params
            .metadata_ptr
            .checked_add(u64::from(params.metadata_len))
            .is_none()
        || params
            .operations_ptr
            .checked_add(u64::from(params.operations_len))
            .is_none()
    {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "encoder: invalid packet addresses or mode");
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: the device API lock retains and serializes this live service handle.
    let service = unsafe { EncoderService::from_handle(params.runtime) };
    let frame = EncodedFrame {
        metadata_ptr: params.metadata_ptr,
        operations_ptr: params.operations_ptr,
        metadata_len: params.metadata_len,
        operations_len: params.operations_len,
        completion: params.completion,
        mode,
        failure_ptr: service.failure_ptr,
        programs: Arc::clone(&service.programs),
    };
    let (done, done_rx) = if mode == EncoderSubmitMode::Queue {
        (None, None)
    } else {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        (Some(sender), Some(receiver))
    };
    if service.encoder.send_encoded(frame, done).is_err() {
        log::error!(target: LOG_TARGET, "encoder: packet admission failed");
        return D3DERR_INVALIDCALL;
    }
    params.admitted = 1;
    let Some(done_rx) = done_rx else {
        return D3D_OK;
    };
    done_rx.recv().unwrap_or_else(|_| {
        log::error!(target: LOG_TARGET, "encoder: packet acknowledgment disconnected after admission");
        D3DERR_INVALIDCALL
    })
}

pub extern "C" fn control_handler(args: *mut core::ffi::c_void) -> i32 {
    use mtld3d_shared::encoder_runtime::EncoderControlParams;
    // SAFETY: the dispatcher supplies this request's exclusive typed record.
    let Some(params) = (unsafe { InPtrMut::<EncoderControlParams>::opt(args) }) else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "encoder: null control parameters");
        return D3DERR_INVALIDCALL;
    };
    let Ok(command) = EncoderControl::try_from(params.command) else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "encoder: invalid control command");
        return D3DERR_INVALIDCALL;
    };
    if params.runtime == 0 {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "encoder: invalid control request");
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: the device API lock retains and serializes this live service handle.
    let service = unsafe { EncoderService::from_handle(params.runtime) };
    let result = match command {
        EncoderControl::DrainRetention => service.encoder.drain_retired_now(),
        EncoderControl::IntakeVisibility => service.encoder.intake_visibility_for(params.argument),
        EncoderControl::Reset => {
            if params.textures_len != 0
                && (params.textures_ptr == 0
                    || params.textures_ptr % 8 != 0
                    || params
                        .textures_ptr
                        .checked_add(u64::from(params.textures_len) * 8)
                        .is_none())
            {
                mtld3d_shared::log_once_warn!(target: LOG_TARGET, "encoder: invalid reset texture list");
                return D3DERR_INVALIDCALL;
            }
            let textures = if params.textures_len == 0 {
                Vec::new()
            } else {
                // SAFETY: the API call retains the validated texture handle list until return.
                unsafe {
                    std::slice::from_raw_parts(
                        params.textures_ptr as *const u64,
                        params.textures_len as usize,
                    )
                }
                .to_vec()
            };
            service.encoder.reset(textures)
        }
    };
    match result {
        Ok(()) => D3D_OK,
        Err(error) => {
            log::error!(target: LOG_TARGET, "encoder: control failed: {error}");
            D3DERR_INVALIDCALL
        }
    }
}

/// Record a failed native replay or submit in the device-owned mailbox.
///
/// # Safety
///
/// A nonzero address is an aligned `AtomicU32` retained through this store.
pub unsafe fn publish_failure(address: u64) {
    if address != 0 {
        // SAFETY: device creation retains this aligned mailbox through native shutdown.
        unsafe { &*(address as *const std::sync::atomic::AtomicU32) }
            .store(1, std::sync::atomic::Ordering::Release);
    }
}
