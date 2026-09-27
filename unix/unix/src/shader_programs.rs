//! Device-local ownership of synchronously parsed shader programs.
//!
//! Creation does not wait for the encoder or prewarm workers. Ordered frame operations adopt
//! registrations later, and dropping the registry releases every unadopted program.

use std::sync::Mutex;

use mtld3d_core::{
    dxso::DxsoProgram,
    ids::ProgramId,
    shader_create::{ShaderCreateError, parse_shader},
};
use mtld3d_shared::{
    InPtrMut, OutPtr, log_once_warn,
    shader_create::{
        CancelShaderProgramParams, CreateShaderProgramParams, ShaderInputSemantic, ShaderStage,
        ShaderUsage,
    },
    slice_from_caller,
};
use mtld3d_types::{D3D_OK, D3DERR_INVALIDCALL};
use rustc_hash::FxHashMap;

use crate::{LOG_TARGET, encoder_service::EncoderService};

/// Native parsed programs awaiting ordered adoption by one device's encoder.
pub struct ProgramRegistry {
    pending: Mutex<PendingPrograms>,
}

impl ProgramRegistry {
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(PendingPrograms {
                next_registration: 1,
                programs: FxHashMap::default(),
            }),
        }
    }

    /// Validate and parse a synchronous creation request, then publish its registration.
    ///
    /// # Safety
    ///
    /// Non-null token and semantic addresses must identify live caller-owned buffers for this
    /// call. The token buffer contains `token_count` initialized u32 values, and the semantic
    /// buffer permits `semantic_capacity` exclusive writes. Both buffers must be disjoint from
    /// each other and from `params`. The caller keeps them unchanged until this call returns.
    pub unsafe fn parse(&self, params: &mut CreateShaderProgramParams) -> i32 {
        params.result = D3DERR_INVALIDCALL;
        params.program_id = 0;
        params.registration = 0;
        params.max_const_used = 0;
        params.semantic_count = 0;
        params.usage = ShaderUsage::empty();
        params.color_out_mask = 0;
        if params.token_count == 0
            || !valid_range::<u32>(params.tokens_ptr, params.token_count)
            || (params.semantic_capacity != 0
                && !valid_range::<ShaderInputSemantic>(
                    params.semantics_ptr,
                    params.semantic_capacity,
                ))
        {
            log_once_warn!(target: LOG_TARGET, "reject shader creation: invalid buffer range");
            return params.result;
        }
        // SAFETY: the caller guarantees readable initialized tokens for this call; the range
        // checks above reject null, overflow, misalignment and oversized slices.
        let tokens = unsafe {
            slice_from_caller(params.tokens_ptr as *const u32, params.token_count as usize)
        };
        let shader = match parse_shader(&params.stage, &tokens) {
            Ok(shader) => shader,
            Err(error) => {
                report_creation_error(&params.stage, &error);
                return params.result;
            }
        };
        let Ok(semantic_count) = u32::try_from(shader.input_semantics.len()) else {
            log_once_warn!(target: LOG_TARGET, "reject shader creation: semantic count overflow");
            return params.result;
        };
        if semantic_count > params.semantic_capacity {
            log_once_warn!(target: LOG_TARGET, "reject shader creation: semantic output too small");
            return params.result;
        }
        let registration = {
            let Ok(mut pending) = self.pending.lock() else {
                log_once_warn!(target: LOG_TARGET, "reject shader creation: registry poisoned");
                return params.result;
            };
            let registration = pending.next_registration;
            let Some(next) = registration.checked_add(1) else {
                log_once_warn!(target: LOG_TARGET, "reject shader creation: registrations exhausted");
                return params.result;
            };
            pending.next_registration = next;
            pending
                .programs
                .insert(registration, (shader.id, shader.program));
            registration
        };
        for (index, semantic) in shader.input_semantics.into_iter().enumerate() {
            let out = (params.semantics_ptr as *mut ShaderInputSemantic).wrapping_add(index);
            // SAFETY: capacity and alignment were checked before publication. The caller grants
            // exclusive writable access to this buffer, and index is less than its capacity.
            unsafe { OutPtr::write_opt(out, semantic) };
        }
        params.program_id = shader.id.raw();
        params.registration = registration;
        params.max_const_used = shader.max_const_used;
        params.semantic_count = semantic_count;
        params.usage = shader.usage;
        params.color_out_mask = shader.color_out_mask;
        params.result = D3D_OK;
        D3D_OK
    }

    /// Transfer one pending program to its encoder exactly once.
    pub fn take(&self, registration: u64) -> Option<(ProgramId, DxsoProgram)> {
        let Ok(mut pending) = self.pending.lock() else {
            log_once_warn!(target: LOG_TARGET, "cannot adopt shader: registry poisoned");
            return None;
        };
        let program = pending.programs.remove(&registration);
        if program.is_none() {
            log_once_warn!(target: LOG_TARGET, "cannot adopt unknown shader registration");
        }
        program
    }

    /// Release an unadopted program belonging to a rejected frame.
    pub fn cancel(&self, registration: u64) {
        drop(self.take(registration));
    }
}

pub extern "C" fn create_handler(args: *mut core::ffi::c_void) -> i32 {
    // SAFETY: the dispatcher supplies the exclusive matching shader creation record.
    let Some(mut params) = (unsafe { InPtrMut::<CreateShaderProgramParams>::opt(args) }) else {
        log_once_warn!(target: LOG_TARGET, "shader creation: null parameters");
        return D3DERR_INVALIDCALL;
    };
    if params.runtime == 0 {
        log_once_warn!(target: LOG_TARGET, "shader creation: null native device");
        params.result = D3DERR_INVALIDCALL;
        params.registration = 0;
        return params.result;
    }
    // SAFETY: the PE device owns this runtime and its API lock excludes destruction.
    let service = unsafe { EncoderService::from_handle(params.runtime) };
    // SAFETY: the synchronous caller owns both disjoint buffers for this call and supplies
    // their exact capacities; parsing checks alignment and arithmetic before dereferencing.
    unsafe { service.programs.parse(&mut params) }
}

pub extern "C" fn cancel_handler(args: *mut core::ffi::c_void) -> i32 {
    // SAFETY: the dispatcher supplies the exclusive matching cancellation record.
    let Some(params) = (unsafe { InPtrMut::<CancelShaderProgramParams>::opt(args) }) else {
        log_once_warn!(target: LOG_TARGET, "shader cancellation: null parameters");
        return D3DERR_INVALIDCALL;
    };
    if params.runtime == 0 || params.registration == 0 {
        log_once_warn!(target: LOG_TARGET, "shader cancellation: null device or registration");
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: the PE device owns this runtime and its API lock excludes destruction.
    let service = unsafe { EncoderService::from_handle(params.runtime) };
    service.programs.cancel(params.registration);
    D3D_OK
}

struct PendingPrograms {
    next_registration: u64,
    programs: FxHashMap<u64, (ProgramId, DxsoProgram)>,
}

fn report_creation_error(stage: &ShaderStage, error: &ShaderCreateError) {
    let operation = match stage {
        ShaderStage::Vertex => "CreateVertexShader",
        ShaderStage::Pixel => "CreatePixelShader",
    };
    match error {
        ShaderCreateError::Parse(error) => {
            log::warn!(target: LOG_TARGET, "{operation} parse failed: {error:?}");
        }
        ShaderCreateError::WrongStage => {
            log_once_warn!(target: LOG_TARGET, "reject {operation}: wrong shader stage -> INVALIDCALL");
        }
        ShaderCreateError::InvalidPixelInput => {
            log::warn!(target: LOG_TARGET, "reject {operation}: POSITION0 on a pixel-shader input register -> INVALIDCALL");
        }
        ShaderCreateError::ConstantRegisterLimit => {
            log::warn!(target: LOG_TARGET, "reject {operation}: constant register out of range -> INVALIDCALL");
        }
    }
}

fn valid_range<T>(address: u64, count: u32) -> bool {
    let Ok(address) = usize::try_from(address) else {
        return false;
    };
    let Some(bytes) = (count as usize).checked_mul(size_of::<T>()) else {
        return false;
    };
    address != 0
        && address.is_multiple_of(align_of::<T>())
        && isize::try_from(bytes).is_ok()
        && address.checked_add(bytes).is_some()
}

#[cfg(test)]
mod tests;
