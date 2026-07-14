use core::ffi::c_void;

use mtld3d_shared::Thunk;

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
    status
}
