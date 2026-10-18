//! Whether this process runs translated, read from the kernel.
//!
//! An `x86_64` `mtld3d.so` runs natively on an Intel Mac and through Rosetta
//! 2 on Apple Silicon, and the two differ in cost: the startup line adds
//! `, under Rosetta 2` to the arch in the second case. macOS answers through
//! the `sysctl.proc_translated` sysctl, 1 for a process Rosetta translates
//! and 0 for a native one. A kernel without Rosetta, an Intel Mac's, has no
//! such name and answers `ENOENT`, which means native too.

use core::ffi::{c_int, c_void};

/// The sysctl that answers whether the calling process runs translated.
const PROC_TRANSLATED: &core::ffi::CStr = c"sysctl.proc_translated";

/// Whether the calling process runs under Rosetta 2; `None` when the kernel would not say.
pub fn translated() -> Option<bool> {
    let mut value: c_int = 0;
    let mut size = size_of::<c_int>();
    // SAFETY: `PROC_TRANSLATED` is NUL-terminated; `value` and `size` are
    // live locals the call writes one `c_int` and its length into; no new
    // value is set.
    let status = unsafe {
        libc::sysctlbyname(
            PROC_TRANSLATED.as_ptr(),
            (&raw mut value).cast::<c_void>(),
            &raw mut size,
            core::ptr::null_mut(),
            0,
        )
    };
    if status == 0 {
        return Some(value == 1);
    }
    if std::io::Error::last_os_error().raw_os_error() == Some(libc::ENOENT) {
        return Some(false);
    }
    mtld3d_shared::log_once_warn!(
        target: crate::LOG_TARGET,
        "sysctl.proc_translated: {}; whether this process runs translated is unknown",
        std::io::Error::last_os_error()
    );
    None
}

#[cfg(test)]
mod tests;
