//! The adapter LUID, from the display driver or allocated for the process.

use mtld3d_types::LUID;

use crate::{LOG_TARGET, direct3d9::PRIMARY_DISPLAY_NAME};

#[link(name = "gdi32")]
unsafe extern "system" {
    fn D3DKMTOpenAdapterFromGdiDisplayName(desc: *mut D3dkmtOpenAdapterFromGdiDisplayName) -> u32;
    fn D3DKMTCloseAdapter(desc: *const D3dkmtCloseAdapter) -> u32;
}

#[link(name = "advapi32")]
unsafe extern "system" {
    fn AllocateLocallyUniqueId(luid: *mut LUID) -> i32;
}

/// Read the adapter LUID the driver reports, or allocate one when it reports none.
pub fn adapter_luid() -> (u32, i32) {
    let mut open = D3dkmtOpenAdapterFromGdiDisplayName {
        device_name: [0; 32],
        adapter: 0,
        adapter_luid: LUID {
            low_part: 0,
            high_part: 0,
        },
        vid_pn_source_id: 0,
    };
    for (slot, unit) in open
        .device_name
        .iter_mut()
        .zip(PRIMARY_DISPLAY_NAME.encode_utf16())
    {
        *slot = unit;
    }
    // SAFETY: gdi32 export; `open` is a live, initialised struct of the
    // documented layout whose name is NUL-terminated (the array is zeroed past it).
    let status = unsafe { D3DKMTOpenAdapterFromGdiDisplayName(&raw mut open) };
    if status == 0 {
        let close = D3dkmtCloseAdapter {
            adapter: open.adapter,
        };
        // SAFETY: gdi32 export; `close` names the adapter handle just opened.
        unsafe { D3DKMTCloseAdapter(&raw const close) };
        if open.adapter_luid.low_part != 0 || open.adapter_luid.high_part != 0 {
            return (open.adapter_luid.low_part, open.adapter_luid.high_part);
        }
    }
    let mut allocated = LUID {
        low_part: 0,
        high_part: 0,
    };
    // SAFETY: advapi32 export; `allocated` is a writable `LUID`.
    let ok = unsafe { AllocateLocallyUniqueId(&raw mut allocated) };
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "GetAdapterLUID: the display driver reports no LUID for {PRIMARY_DISPLAY_NAME} \
         (status {status:#x}); the adapter is named by a LUID allocated for the process \
         (allocated={})",
        ok != 0
    );
    (allocated.low_part, allocated.high_part)
}

/// Win32 `D3DKMT_OPENADAPTERFROMGDIDISPLAYNAME`: a GDI display name in, its adapter out.
#[repr(C)]
struct D3dkmtOpenAdapterFromGdiDisplayName {
    device_name: [u16; 32],
    adapter: u32,
    adapter_luid: LUID,
    vid_pn_source_id: u32,
}

/// Win32 `D3DKMT_CLOSEADAPTER`: the adapter handle an open handed out.
#[repr(C)]
struct D3dkmtCloseAdapter {
    adapter: u32,
}
