//! What MSVC's static CRT supplies that the ARM64X link has no library for.
//!
//! The i386 and `x86_64` DLLs link MSVC's `msvcrt.lib`, whose static part defines
//! everything below. The two ARM64X halves link llvm-mingw's CRT instead (see
//! `windows-arm64x` in the Makefile), which defines none of it, and MSVC's
//! ARM64 and ARM64EC libraries are not available to this build.

use core::ffi::c_void;

use crate::DLL_THREAD_DETACH;

/// The `type_info` vtable, under its MSVC decorated name.
///
/// std's `catch_unwind` emits a C++ type descriptor for the Rust panic, and the
/// descriptor's first field points at this vtable, so the name has to resolve.
/// Nothing ever reads through it. Every profile in `windows/Cargo.toml` builds
/// with `panic = "abort"`, so no Rust panic is thrown as a C++ exception for a
/// handler to match. And a handler that did match one would not call through
/// it either: the C++ frame handler (`__CxxFrameHandler3`, which Wine
/// implements) compares a thrown type with a caught one by descriptor address
/// and then by decorated name, never through the vtable.
#[unsafe(export_name = "??_7type_info@@6B@")]
pub static TYPE_INFO_VFTABLE: [usize; 1] = [0];

/// The TLS callback that tears down snmalloc's per-thread allocator.
///
/// The loader finds it in this image's TLS callback array, which the linker
/// assembles from the `.CRT$XL*` sections in name order, so it needs no object
/// to reach and a static is the only form it can take. It holds a function
/// pointer and nothing that changes. `$XLY` sorts it after the CRT's own
/// callbacks (`$XLC`, `$XLD`) and before the array's end marker (`$XLZ`).
///
/// The callback is kept in the link by [`attach`]: this crate is linked as a
/// static library, and the linker takes an object out of one only for a symbol
/// something else needs.
#[unsafe(link_section = ".CRT$XLY")]
#[used]
static THREAD_EXIT_CALLBACK: extern "system" fn(*mut c_void, u32, *mut c_void) = on_tls_event;

unsafe extern "C" {
    /// snmalloc's per-thread teardown, defined under `SNMALLOC_USE_THREAD_CLEANUP`.
    ///
    /// It flushes the calling thread's allocator, which returns what it caches
    /// and posts its frees to other threads, and hands the allocator back to
    /// the pool the next thread takes one from. A thread that never allocated
    /// has nothing to tear down and returns at once, and one that allocates
    /// again afterwards gets an allocator again.
    fn _malloc_thread_cleanup();
}

/// Keeps [`THREAD_EXIT_CALLBACK`] in the link; `DllMain` calls it on attach.
///
/// Taking the static's address where the optimiser cannot see it used is what
/// leaves a relocation against it in `DllMain`'s object, and so pulls the
/// callback's object into the image.
pub fn attach() {
    core::hint::black_box(&THREAD_EXIT_CALLBACK);
}

/// Runs snmalloc's per-thread teardown when a thread exits.
///
/// This is the part of MSVC's CRT that the x86 DLLs get from `__tlregdtor` and
/// the TLS callback behind it, which run snmalloc's C++ `thread_local`
/// destructor. llvm-mingw's `__tlregdtor` accepts a destructor and never runs
/// it, so the ARM64X halves build snmalloc to be told about thread exit
/// instead, and are told here.
///
/// Unlike the x86 DLLs, whose CRT callback runs the destructors on a process
/// detach too, this one leaves a process detach alone, on purpose. Wine calls
/// TLS callbacks before the DLL's entry point, so a teardown there would run
/// ahead of `DllMain`, whose detach at the exit of a process that created a
/// device ends it so that the allocator is not torn down on Wine's 1 MB
/// main-thread stack. What it gives up is small: the process is ending or the
/// image, which no device ever pinned, is being unmapped, and snmalloc's
/// memory stays reserved either way.
extern "system" fn on_tls_event(_module: *mut c_void, reason: u32, _reserved: *mut c_void) {
    if reason == DLL_THREAD_DETACH {
        // SAFETY: snmalloc's teardown may run at any point on the calling
        // thread; here it runs as the thread exits.
        unsafe { _malloc_thread_cleanup() };
    }
}
