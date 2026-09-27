use core::ffi::c_void;

use mtld3d_shared::Thunk;
#[cfg(perf_tracking)]
use mtld3d_shared::encoder_protocol::EncoderSubmitMode;

#[link(name = "mtld3d", kind = "raw-dylib")]
unsafe extern "C" {
    fn mtld3d_unix_call(code: u32, args: *mut c_void) -> i32;
}

pub fn unix_call<T: Thunk>(params: &mut T) -> i32 {
    mtld3d_shared::crumb!(
        "ucall:begin",
        u64::from(T::CODE),
        std::ptr::from_ref::<T>(params) as usize as u64,
    );
    // SAFETY: `params` is a live `&mut T` and `T::CODE` is the matching
    // thunk discriminant for `T`; the unix-side dispatcher casts back to
    // `*mut T` using the same code.
    let status =
        unsafe { mtld3d_unix_call(T::CODE, std::ptr::from_mut::<T>(params).cast::<c_void>()) };
    mtld3d_shared::crumb!(
        "ucall:end",
        u64::from(T::CODE),
        u64::from(status.cast_unsigned()),
    );
    // This diagnostic runs only in PERF builds with this trace target enabled.
    // WriteLog is excluded because the logger forwards this line through it.
    #[cfg(perf_tracking)]
    if let Some(kind) = mtld3d_core::perf::unix_dispatch_kind(T::CODE) {
        log::trace!(
            target: "mtld3d::perf::thunks",
            "thunk_dispatch code={} kind={kind} status={status} type={} submit_mode={}",
            T::CODE,
            std::any::type_name::<T>(),
            match params.perf_submit_mode() {
                Some(EncoderSubmitMode::Queue) => "queue",
                Some(EncoderSubmitMode::WaitForSubmit) => "wait_for_submit",
                Some(EncoderSubmitMode::WaitForGpu) => "wait_for_gpu",
                None => "none",
            },
        );
    }
    status
}
