//! HRESULT codes returned across the COM ABI.
//!
//! These are `u32` bit patterns the COM ABI hands back as `i32`. Use
//! `cast_signed()` for the wrap rather than `as i32` so the bit-reinterpret is
//! explicit (and clippy stops flagging it as accidental sign loss).

pub const D3D_OK: i32 = 0;
pub const E_FAIL: i32 = 0x8000_4005_u32.cast_signed();
pub const E_NOINTERFACE: i32 = 0x8000_4002_u32.cast_signed();
/// `E_NOTIMPL` — "not implemented".
///
/// A plain (non-Ex) `IDirect3DDevice9` returns this when a caller passes a
/// non-NULL `pSharedHandle` to a `Create*` method: shared resource handles are
/// a D3D9Ex-only feature.
pub const E_NOTIMPL: i32 = 0x8000_4001_u32.cast_signed();
pub const D3DERR_INVALIDCALL: i32 = 0x8876_086C_u32.cast_signed();
pub const D3DERR_NOTAVAILABLE: i32 = 0x8876_086A_u32.cast_signed();
pub const D3DERR_MOREDATA: i32 = 0x8876_0867_u32.cast_signed();
pub const D3DERR_NOTFOUND: i32 = 0x8876_0866_u32.cast_signed();
/// `D3DOK_NOAUTOGEN` — a *success* code (`SUCCEEDED` is true).
///
/// The format is valid but cannot auto-generate mipmaps because it is not
/// render-targetable. `MAKE_D3DSTATUS(2159)`.
pub const D3DOK_NOAUTOGEN: i32 = 0x0876_086F;
