//! Exports whose first bytes are a fixed prologue that inline-hook engines can relocate.
//!
//! Overlays and capture tools hook `d3d9.dll` by its exports: they decode the
//! first instructions of the export with their own length decoder, copy at
//! least five bytes of whole instructions into a trampoline, and write a
//! `jmp rel32` over them. A decoder that does not know one of those
//! instructions skips the hook, and some decoders only know the instruction
//! forms MSVC emits at a function entry. What the compiler emits for a short
//! Rust function is neither fixed nor chosen with that in mind, so every
//! export the d3d9 API defines is a naked entry that starts with a prologue
//! of its own and jumps to the Rust function that does the work:
//!
//! - i386: `8B FF 55 8B EC 5D` (`mov edi, edi; push ebp; mov ebp, esp; pop
//!   ebp`), the hot-patch prologue every system DLL export starts with. A hook
//!   engine copies the first three instructions, exactly five bytes.
//! - `x86_64`: `0F 1F 44 00 00`, the five-byte NOP. One whole instruction
//!   covers the five bytes, the entry leaves `rsp` untouched, so it needs no
//!   unwind data, and the first instruction is at least two bytes long, which
//!   is the x64 hot-patch requirement.
//!
//! The `jmp` to the body follows the prologue, so no relative branch sits in
//! the bytes a hook engine displaces. The entry does not touch the arguments
//! or the stack the body sees, so the body runs with the caller's frame as if
//! it had been called directly.
//!
//! The ARM64X build exports the body through a plain wrapper: its code is
//! ARM64EC or ARM64, which no x86 hook engine decodes either way.

/// Export `$body` under the name `$export` behind the fixed hook prologue.
///
/// The entry repeats the body's signature, so on i386 the symbol carries the
/// same stdcall decoration (`_Name@N`) the body would. `const fn` marks a body
/// that is a `const fn`: the ARM64X wrapper is then one too, while a naked
/// entry has no `const` form and ignores it.
macro_rules! hookable_export {
    ($(#[$attr:meta])* $export:literal => const fn $($rest:tt)*) => {
        hookable_export!(@entry [const] $(#[$attr])* $export => fn $($rest)*);
    };
    ($(#[$attr:meta])* $export:literal => fn $($rest:tt)*) => {
        hookable_export!(@entry [] $(#[$attr])* $export => fn $($rest)*);
    };
    (
        @entry [$($qualifier:tt)*]
        $(#[$attr:meta])*
        $export:literal => fn $entry:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? = $body:path;
    ) => {
        $(#[$attr])*
        #[cfg(target_arch = "x86")]
        #[unsafe(export_name = $export)]
        // SAFETY: the entry runs the hot-patch prologue, which leaves every
        // register and the stack as it found them (`ebp` is pushed and popped
        // again, `mov` of a register onto itself changes nothing), then jumps
        // to a body with the same signature and calling convention, so the
        // body sees the caller's arguments and returns to the caller.
        #[unsafe(naked)]
        pub extern "system" fn $entry($($arg: $ty),*) $(-> $ret)? {
            core::arch::naked_asm!(
                ".byte 0x8b, 0xff",
                ".byte 0x55",
                ".byte 0x8b, 0xec",
                ".byte 0x5d",
                "jmp {body}",
                body = sym $body,
            )
        }

        $(#[$attr])*
        #[cfg(target_arch = "x86_64")]
        #[unsafe(export_name = $export)]
        // SAFETY: the entry runs one NOP, then jumps to a body with the same
        // signature and calling convention, so the body sees the caller's
        // arguments and stack and returns to the caller.
        #[unsafe(naked)]
        pub extern "system" fn $entry($($arg: $ty),*) $(-> $ret)? {
            core::arch::naked_asm!(
                ".byte 0x0f, 0x1f, 0x44, 0x00, 0x00",
                "jmp {body}",
                body = sym $body,
            )
        }

        $(#[$attr])*
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        #[unsafe(export_name = $export)]
        pub $($qualifier)* extern "system" fn $entry($($arg: $ty),*) $(-> $ret)? {
            $body($($arg),*)
        }
    };
}
