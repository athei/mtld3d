//! The x87 control word a device creation leaves on the application's thread.
//!
//! `mtld3d_core::fpu` owns which word and when; this module reads and writes
//! the register, which has to happen on the thread that called `CreateDevice`.

/// Give the calling thread the device's x87 control word, unless `behavior_flags` keeps it.
#[cfg(target_arch = "x86")]
pub fn apply_device_control_word(behavior_flags: u32) {
    if !mtld3d_core::fpu::sets_control_word(behavior_flags) {
        return;
    }
    let mut current: u16 = 0;
    // SAFETY: `fnstcw` stores the 16-bit control word through the pointer,
    // which names `current`, a live writable `u16`; it touches no flags or stack.
    unsafe {
        core::arch::asm!(
            "fnstcw word ptr [{}]",
            in(reg) &raw mut current,
            options(nostack, preserves_flags),
        );
    }
    let word = mtld3d_core::fpu::device_control_word(current);
    // SAFETY: `fldcw` reads the 16-bit control word from `word`, a live `u16`;
    // every exception is masked in it, so loading it raises none.
    unsafe {
        core::arch::asm!(
            "fldcw word ptr [{}]",
            in(reg) &raw const word,
            options(nostack, readonly, preserves_flags),
        );
    }
}

/// A 64-bit device leaves the control word alone, so there is nothing to apply.
#[cfg(not(target_arch = "x86"))]
pub const fn apply_device_control_word(_behavior_flags: u32) {}
