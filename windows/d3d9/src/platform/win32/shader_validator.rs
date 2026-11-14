//! `IDirect3DShaderValidator9` — the object the `Direct3DShaderValidatorCreate9` export returns.
//!
//! An undocumented validation interface used by the conformance suite and
//! shader tools (`fxc`); games never use it. Mirrors Wine's `d3d9_main.c`: a
//! stateless `'static` singleton with constant refcounts (`AddRef` → 2,
//! `Release` → 1, never freed) whose `Begin`/`Instruction`/`End` accept
//! everything (`S_OK`). The contract here is "be a valid, callable object" —
//! the suite otherwise calls through a NULL pointer (the import would resolve
//! to a Wine stub returning NULL) and faults.

use core::ffi::c_void;

use mtld3d_types::{Guid, IDirect3DShaderValidator9Vtbl};

use super::{D3D_OK, E_NOINTERFACE, null_out};

static SHADER_VALIDATOR_VTBL: IDirect3DShaderValidator9Vtbl = IDirect3DShaderValidator9Vtbl {
    query_interface: sv_query_interface,
    add_ref: sv_add_ref,
    release: sv_release,
    begin: sv_begin,
    instruction: sv_instruction,
    end: sv_end,
};

/// `repr(C)` so the leading `vtbl` pointer matches the COM ABI the caller dereferences.
#[repr(C)]
struct ShaderValidator {
    vtbl: *const IDirect3DShaderValidator9Vtbl,
}

// SAFETY: a stateless stub — the only field is a `'static` vtbl pointer and the
// object carries no mutable state (refcounts are constants), so sharing the
// singleton across threads is sound.
unsafe impl Sync for ShaderValidator {}

static SHADER_VALIDATOR: ShaderValidator = ShaderValidator {
    vtbl: &raw const SHADER_VALIDATOR_VTBL,
};

/// Backs the `Direct3DShaderValidatorCreate9` export.
///
/// Returns the stub singleton (never freed; constant refcounts).
pub fn create() -> *mut c_void {
    (&raw const SHADER_VALIDATOR).cast::<c_void>().cast_mut()
}

extern "system" fn sv_query_interface(
    _this: *mut c_void,
    _iid: *const Guid,
    out: *mut *mut c_void,
) -> i32 {
    null_out(out);
    E_NOINTERFACE
}

// Constant refcounts mirroring Wine: the singleton is never created or freed,
// so `AddRef` reports 2 and `Release` reports 1.
const extern "system" fn sv_add_ref(_this: *mut c_void) -> u32 {
    2
}

const extern "system" fn sv_release(_this: *mut c_void) -> u32 {
    1
}

const extern "system" fn sv_begin(
    _this: *mut c_void,
    _callback: *mut c_void,
    _context: *mut c_void,
    _arg3: usize,
) -> i32 {
    D3D_OK
}

const extern "system" fn sv_instruction(
    _this: *mut c_void,
    _file: *const c_void,
    _line: i32,
    _tokens: *const u32,
    _token_count: u32,
) -> i32 {
    D3D_OK
}

const extern "system" fn sv_end(_this: *mut c_void) -> i32 {
    D3D_OK
}
