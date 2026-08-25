use core::ffi::c_void;
use std::sync::LazyLock;

use log::{error, info, trace, warn};
use mtld3d_core::caps;
use mtld3d_shared::{
    AttachMetalLayerParams, CreateBackbufferParams, CreateCommandQueueParams,
    CreateDepthTextureParams, DestroyCommandQueueParams, GetDeviceInfoParams, InPtrMut,
    MetalHandle, OutPtr, VtableThis,
    mtl_handle::{MTLTextureKind, NSViewKind},
};
use mtld3d_types::{
    D3DADAPTER_IDENTIFIER9, D3DCAPS9, D3DDEVTYPE_HAL, D3DDISPLAYMODE, D3DFMT_A1R5G5B5,
    D3DFMT_A8R8G8B8, D3DFMT_A16B16G16R16, D3DFMT_A16B16G16R16F, D3DFMT_A32B32G32R32F, D3DFMT_ATI1,
    D3DFMT_D16, D3DFMT_D24S8, D3DFMT_D24X8, D3DFMT_D32, D3DFMT_DF16, D3DFMT_DF24, D3DFMT_DXT1,
    D3DFMT_DXT3, D3DFMT_DXT5, D3DFMT_G16R16, D3DFMT_G16R16F, D3DFMT_G32R32F, D3DFMT_INTZ,
    D3DFMT_R5G6B5, D3DFMT_R16F, D3DFMT_R32F, D3DFMT_UYVY, D3DFMT_X8R8G8B8, D3DFMT_YUY2,
    D3DMULTISAMPLE_NONE, D3DOK_NOAUTOGEN, D3DPRESENT_PARAMETERS, D3DRTYPE_CUBETEXTURE,
    D3DRTYPE_SURFACE, D3DRTYPE_TEXTURE, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DEPTHSTENCIL,
    D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING, D3DUSAGE_QUERY_SRGBREAD, D3DUSAGE_QUERY_SRGBWRITE,
    D3DUSAGE_QUERY_VERTEXTEXTURE, D3DUSAGE_RENDERTARGET, Guid, IDirect3D9Vtbl,
};

use super::{
    D3D_OK, D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, LOG_TARGET,
    device::Direct3DDevice9,
    encoder::{EncoderThread, FrameData, FrameInit},
    fullscreen::Rect,
    stage_bindings::STAGE_COUNT,
    unix_call::unix_call,
};

// Dynamic display-mode list served by GetAdapterModeCount / EnumAdapterModes /
// GetAdapterDisplayMode. Built once on first enumeration call from the host
// display's current mode in the Win32 view (`EnumDisplaySettingsW`, see
// `build_adapter_modes`), combined with a bank of common gaming resolutions
// <= host. macOS doesn't do D3D9-style mode-setting — CAMetalLayer renders at
// whatever size we ask, the WindowServer composites onto the actual desktop —
// so the list shapes the game's UI dropdown, not actual rendering behaviour.
// The first entry doubles as the current adapter display mode.
static ADAPTER_MODES: LazyLock<Vec<D3DDISPLAYMODE>> = LazyLock::new(build_adapter_modes);

// Common gaming resolutions filtered down to those <= host display. Host
// native is prepended separately so GetAdapterDisplayMode returns it.
const RES_BANK: &[(u32, u32)] = &[
    (3840, 2160),
    (3456, 2234),
    (3024, 1964),
    (2880, 1800),
    (2560, 1600),
    (2560, 1440),
    (1920, 1200),
    (1920, 1080),
    (1680, 1050),
    (1600, 900),
    (1440, 900),
    (1366, 768),
    (1280, 1024),
    (1280, 800),
    (1280, 720),
    (1024, 768),
    (800, 600),
    (640, 480),
];

// Adapter color formats enumerated. X8R8G8B8 = "32-bit" in most game UIs
// (32-bit container, 24 useful color bits), R5G6B5 = "16-bit".
// A2R10G10B10 deliberately excluded — CAMetalLayer is hardcoded BGRA8;
// advertising 10-bit would silently downgrade an HDR opt-in.
const ADAPTER_FORMATS: &[u32] = &[D3DFMT_X8R8G8B8, D3DFMT_R5G6B5];

/// Sub-target for the display-enumeration diagnostic probes.
///
/// Confirms which `IDirect3D9` enumeration endpoints a given game actually
/// exercises at video-menu open time. Permanent probe (zero-cost when off);
/// `RUST_LOG=mtld3d::d3d9::display=trace` opts in. Useful for distinguishing
/// games whose video-menu dropdowns are D3D9-driven vs Win32-driven (Wine's
/// `EnumDisplaySettings` → macdrv → `CGDisplayCopyAllDisplayModes`).
const DISPLAY_TRACE_TARGET: &str = "mtld3d::d3d9::display";

/// Maximum tolerated difference between a candidate mode's aspect ratio and the host display's.
///
/// Expressed as a fraction of the host aspect.
///
/// 15 % keeps 4:3 (1.333), 16:10 (1.6), and 16:9 (1.778) alongside the
/// MBP-native 3:2-ish (1.547) host, and drops 5:4 (1.250, ~19 % off) and
/// 21:9 (2.333). The intent is "no obviously-wrong aspect in the
/// resolution dropdown" — not a hard mathematical filter; if a future host
/// aspect surprises us, widen this number.
const ASPECT_TOLERANCE: f64 = 0.15;

/// The current adapter display mode — the live host-native desktop resolution.
///
/// This is what `GetAdapterDisplayMode` reports (the first `ADAPTER_MODES`
/// entry), returned as `(width, height)` for callers that bound against the
/// desktop.
pub fn adapter_display_mode_dims() -> (u32, u32) {
    (ADAPTER_MODES[0].width, ADAPTER_MODES[0].height)
}

/// The adapter display-mode format (`D3DFMT_*`) — the format `GetAdapterDisplayMode` reports.
///
/// A windowed back buffer requested as `D3DFMT_UNKNOWN` resolves to this.
pub fn adapter_display_format() -> u32 {
    ADAPTER_MODES[0].format
}

/// The display mode a device or its implicit swap chain reports.
///
/// A fullscreen device owns the (emulated) mode, so the honored back-buffer
/// size is the current mode; the refresh rate is the requested one, or the
/// host's when the request left it zero. A windowed device never owns the
/// mode and reports the desktop's, exactly like native D3D9.
pub fn reported_display_mode(pp: &D3DPRESENT_PARAMETERS) -> D3DDISPLAYMODE {
    if pp.windowed == 0 {
        let refresh_rate = if pp.full_screen_refresh_rate_in_hz != 0 {
            pp.full_screen_refresh_rate_in_hz
        } else {
            ADAPTER_MODES[0].refresh_rate
        };
        D3DDISPLAYMODE {
            width: pp.back_buffer_width,
            height: pp.back_buffer_height,
            refresh_rate,
            format: D3DFMT_X8R8G8B8,
        }
    } else {
        ADAPTER_MODES[0]
    }
}

fn build_adapter_modes() -> Vec<D3DDISPLAYMODE> {
    // The host mode comes from the Win32 view (`EnumDisplaySettingsW` →
    // macdrv), NOT from `NSScreen`: win32u validates the tests' and games'
    // `ChangeDisplaySettingsW` calls against this view and derives
    // `GetMonitorInfoW` from it, so a mode enumerated here is consistent
    // with the window-management side on every display. An `NSScreen` read
    // disagreed by the Retina factor on displays where the two scale
    // differently (a CI runner's virtual display), splitting
    // `GetAdapterDisplayMode` from the monitor rect.
    let host = crate::fullscreen::current_display_mode();
    let (w, h, hz) = host.unwrap_or((1920, 1080, 60));
    let host_w = if w > 0 { w } else { 1920 };
    let host_h = if h > 0 { h } else { 1080 };
    let host_hz = if hz > 0 { hz } else { 60 };
    let host_aspect = f64::from(host_w) / f64::from(host_h);

    let mut sizes: Vec<(u32, u32)> = Vec::with_capacity(RES_BANK.len() + 1);
    sizes.push((host_w, host_h));
    for &(w, h) in RES_BANK {
        let is_host = w == host_w && h == host_h;
        let fits = w <= host_w && h <= host_h;
        let aspect = f64::from(w) / f64::from(h);
        let aspect_off = (aspect - host_aspect).abs() / host_aspect;
        if fits && !is_host && aspect_off <= ASPECT_TOLERANCE {
            sizes.push((w, h));
        }
    }

    let mut modes = Vec::with_capacity(sizes.len() * ADAPTER_FORMATS.len());
    for &fmt in ADAPTER_FORMATS {
        for &(w, h) in &sizes {
            modes.push(D3DDISPLAYMODE {
                width: w,
                height: h,
                refresh_rate: host_hz,
                format: fmt,
            });
        }
    }

    info!(
        target: LOG_TARGET,
        "adapter modes: host {host_w}x{host_h}@{host_hz}Hz aspect={host_aspect:.3}, {} entries",
        modes.len()
    );
    modes
}

static DIRECT3D9_VTBL: IDirect3D9Vtbl = IDirect3D9Vtbl {
    query_interface: d3d9_query_interface,
    add_ref: d3d9_add_ref,
    release: d3d9_release,
    register_software_device: d3d9_register_software_device,
    get_adapter_count: d3d9_get_adapter_count,
    get_adapter_identifier: d3d9_get_adapter_identifier,
    get_adapter_mode_count: d3d9_get_adapter_mode_count,
    enum_adapter_modes: d3d9_enum_adapter_modes,
    get_adapter_display_mode: d3d9_get_adapter_display_mode,
    check_device_type: d3d9_check_device_type,
    check_device_format: d3d9_check_device_format,
    check_device_multi_sample_type: d3d9_check_device_multi_sample_type,
    check_depth_stencil_match: d3d9_check_depth_stencil_match,
    check_device_format_conversion: d3d9_check_device_format_conversion,
    get_device_caps: d3d9_get_device_caps,
    get_adapter_monitor: d3d9_get_adapter_monitor,
    create_device: d3d9_create_device,
};

// ── IDirect3D9 COM object ──

#[repr(C)]
pub struct Direct3D9 {
    vtbl: *const IDirect3D9Vtbl,
    refcount: u32,
}

impl Direct3D9 {
    pub fn new() -> Self {
        Self {
            vtbl: &raw const DIRECT3D9_VTBL,
            refcount: 1,
        }
    }
}

// Display formats accepted as the adapter_format / back_buffer_format pair.
const fn is_display_format(fmt: u32) -> bool {
    // Adapter/display formats only — alpha formats (A8R8G8B8) can be a
    // backbuffer but never a display mode, so they are excluded here.
    matches!(fmt, D3DFMT_X8R8G8B8 | D3DFMT_R5G6B5)
}

/// 32-bit RGB colour family whose members interconvert at present time.
///
/// (X8R8G8B8 / A8R8G8B8 — the alpha channel is ignored on present).
const fn is_32bit_rgb(fmt: u32) -> bool {
    matches!(fmt, D3DFMT_X8R8G8B8 | D3DFMT_A8R8G8B8)
}

/// Whether a fullscreen backbuffer of `src` can be presented to a `dst` display format.
///
/// The same format, or another member of the same 32-bit colour family. This
/// is the fullscreen rule only: the spec allows no present-time conversion
/// there, the display and backbuffer formats must match ignoring alpha.
/// Windowed mode goes through [`is_format_conversion_supported`] instead.
const fn is_present_compatible(src: u32, dst: u32) -> bool {
    src == dst || (is_32bit_rgb(src) && is_32bit_rgb(dst))
}

/// Formats `StretchRect` can read as the source of a format conversion.
///
/// The render-quad path samples any colour format the device can render and
/// decodes the two packed 4:2:2 YUV formats (`YUY2` / `UYVY`) in its fragment
/// function; the offscreen-plain CPU converter covers the same set.
const fn is_conversion_source(fmt: u32) -> bool {
    is_render_target_format(fmt) || matches!(fmt, D3DFMT_YUY2 | D3DFMT_UYVY)
}

/// `CheckDeviceFormatConversion`: whether `StretchRect` converts `src` into `dst`.
///
/// A format always converts to itself (the identity rows hold for any code,
/// mapped or not). Otherwise the source must be something the blit can read
/// ([`is_conversion_source`]) and the destination a renderable colour format
/// (`is_render_target_format`), which is what the render-quad writes into.
/// Windowed `CheckDeviceType` shares this predicate on purpose: the runtime
/// asserts `CheckDeviceType(windowed) == CheckDeviceFormat(RT, bb) &&
/// CheckDeviceFormatConversion(bb, display)`, so the two must never drift.
const fn is_format_conversion_supported(src: u32, dst: u32) -> bool {
    if src == dst {
        return true;
    }
    is_conversion_source(src) && is_render_target_format(dst)
}

// Formats the texture pool can sample or receive uploads in.
//
// Derived from the create path rather than listed: `CreateTexture` accepts
// exactly the formats `map_d3d_format` maps, so an independent list here
// drifts — the answer then disagrees with what a create actually does, and
// callers that probe first (every engine that picks a scene format from
// `CheckDeviceFormat`) take a fallback path for a format we support. That
// covers the odd ones deliberately: YUY2/UYVY back a creatable, lockable RG8
// surface with no YUV sampling, and ATI1 is a creatable BC4 texture.
//
// The FOURCC sampleable-depth formats (`INTZ` / `DF24` / `DF16`) belong here
// too, and are not colour mappings: D3D9-era engines (incl. WoW's CSM path)
// probe them via `CheckDeviceFormat(rtype=TEXTURE, fmt=INTZ)` without
// `USAGE_DEPTHSTENCIL` in the query, and only enable hardware shadow mapping
// when at least one comes back available.
const fn is_texture_format(fmt: u32) -> bool {
    // ATI1 is the one carve-out, for the same reason it is excluded from the
    // cube answer: it creates, but its lock reports the BC4 block pitch
    // (8 bytes per 4x4 block) where D3D9 reports ATI1N a byte per pixel, so
    // advertising it would hand callers a pitch they cannot use.
    (mtld3d_core::format::is_mapped_color_format(fmt) && !matches!(fmt, D3DFMT_ATI1))
        || mtld3d_core::format::is_raw_depth_fetch_format(fmt)
}

/// Sampleable cube colour formats backed by `MTLTextureTypeCube`.
///
/// ATI1 requires extension-specific cube lock semantics that are not
/// implemented. Packed YUV has no shader sampling path, and depth cube maps
/// are not implemented.
const fn is_cube_texture_format(fmt: u32) -> bool {
    mtld3d_core::format::is_mapped_color_format(fmt)
        && !matches!(fmt, D3DFMT_ATI1 | D3DFMT_YUY2 | D3DFMT_UYVY)
        && !is_depth_stencil_format(fmt)
}

// Formats the Metal backend can render into.
//
// `R5G6B5` and `A1R5G5B5` map bit-for-bit to the native `B5G6R5Unorm` /
// `BGR5A1Unorm` Metal formats, which are colour-renderable on Apple GPUs, so
// they are valid render targets. `A4R4G4B4` is excluded: its native Metal
// format (`ABGR4Unorm`) has a different channel order that is corrected with a
// sampler swizzle, and a swizzle only affects reads — render writes would land
// in the wrong bits — so `A4R4G4B4` is a sampling-only format. Backbuffer /
// CAMetalLayer is hardcoded BGRA8, so a request for one of these as a backbuffer
// format hits the existing substitute-warn at `d3d9_create_device`.
//
// The float family is renderable: every one of R16F / G16R16F /
// A16B16G16R16F / R32F / G32R32F / A32B32G32R32F builds a colour pipeline —
// blend-enabled included — on Apple Silicon, and `supports32BitFloatFiltering`
// is true there, so the 32-bit members sample with linear filtering as well.
// Engines that render HDR internally (an off-screen float scene target, their
// own tone-map into the 8-bit backbuffer) probe exactly this before choosing
// their scene format.
pub const fn is_render_target_format(fmt: u32) -> bool {
    matches!(
        fmt,
        D3DFMT_A8R8G8B8
            | D3DFMT_X8R8G8B8
            | D3DFMT_R5G6B5
            | D3DFMT_A1R5G5B5
            | D3DFMT_G16R16
            | D3DFMT_A16B16G16R16
            | D3DFMT_R16F
            | D3DFMT_G16R16F
            | D3DFMT_A16B16G16R16F
            | D3DFMT_R32F
            | D3DFMT_G32R32F
            | D3DFMT_A32B32G32R32F
    )
}

/// Depth-stencil formats.
///
/// Includes the FOURCC sampleable-depth formats (`INTZ` / `DF24` / `DF16`)
/// — created with `USAGE_DEPTHSTENCIL`, bound as the depth target during a
/// caster pass and sampled as a depth texture in the receiver pass. Apple
/// Silicon promotes all of them to `Depth32Float` (see
/// `format::map_d3d_depth_format`).
pub const fn is_depth_stencil_format(fmt: u32) -> bool {
    matches!(
        fmt,
        D3DFMT_D16
            | D3DFMT_D24S8
            | D3DFMT_D24X8
            | D3DFMT_D32
            | D3DFMT_INTZ
            | D3DFMT_DF24
            | D3DFMT_DF16
    )
}

/// Subset of depth-stencil formats that carry a stencil plane.
///
/// Drives Metal pipeline state: pipelines matched against depth-only
/// attachments must leave `stencilAttachmentPixelFormat` at Invalid, or Metal
/// rejects the pipeline.
pub const fn depth_format_has_stencil(fmt: u32) -> bool {
    // Must agree with `map_d3d_depth_format`: every D3D depth/stencil format
    // that maps to the combined Metal `Depth32Float_Stencil8` texture carries a
    // stencil plane the render pipeline MUST also declare, or the pipeline's
    // depth/stencil attachment formats desync from the bound depth texture — a
    // Metal validation failure, and heap-corrupting undefined behaviour with
    // the layer off. Deriving from the same mapping keeps them in lockstep:
    // D15S1 and D24X4S4 are combined formats too, not just D24S8/D24FS8.
    matches!(
        mtld3d_core::format::map_d3d_depth_format(fmt),
        Some(mtld3d_shared::mtl::PixelFormat::Depth32FloatStencil8)
    )
}

// D3D9 colour formats whose Metal counterpart has an sRGB twin. Mirror of
// the PE-side `PixelFormat::srgb_twin()` table in `unix/shared/src/mtl.rs`
// — drives the answer `CheckDeviceFormat` returns for
// `D3DUSAGE_QUERY_SRGBREAD` / `D3DUSAGE_QUERY_SRGBWRITE`. Adding a new
// linear/sRGB pair to `PixelFormat` requires extending this list too.
const fn has_srgb_twin(fmt: u32) -> bool {
    matches!(
        fmt,
        D3DFMT_A8R8G8B8 | D3DFMT_X8R8G8B8 | D3DFMT_DXT1 | D3DFMT_DXT3 | D3DFMT_DXT5
    )
}

// Formats for which `D3DUSAGE_QUERY_SRGBREAD` (the `D3DSAMP_SRGBTEXTURE`
// sampling decode) is honoured. The 32-bit RGB formats have an sRGB twin for
// SRGBWRITE (eager render-target views, `has_srgb_twin`) but NO runtime
// SRGBTEXTURE sampling decode, so advertising SRGBREAD for them over-promises
// (a sampler expecting the sRGB decode then reads them un-decoded). The
// block-compressed formats keep their existing SRGBREAD advertisement.
const fn has_srgb_read_decode(fmt: u32) -> bool {
    matches!(fmt, D3DFMT_DXT1 | D3DFMT_DXT3 | D3DFMT_DXT5)
}

// ── IUnknown implementation (IDirect3D9) ──

extern "system" fn d3d9_query_interface(
    this: *mut c_void,
    riid: *const Guid,
    ppv: *mut *mut c_void,
) -> i32 {
    // SAFETY: vtable thunk; `this`, `riid` and `ppv` are the caller's per the
    // IUnknown::QueryInterface ABI.
    unsafe {
        crate::com_ref::com_query_interface(
            this,
            riid,
            ppv,
            &[mtld3d_types::IID_IUNKNOWN, mtld3d_types::IID_IDIRECT3D9],
            d3d9_add_ref,
            "IDirect3D9",
        )
    }
}

extern "system" fn d3d9_add_ref(this: *mut c_void) -> u32 {
    // SAFETY: D3D9 AddRef — null `this` is UB per spec; we preserve the
    // crash semantic so refcount miscounts surface as a null-deref.
    // SAFETY: IDirect3D9 IUnknown thunk; D3D9 ABI guarantees `this` is *mut Direct3D9.
    let mut wrap = unsafe { VtableThis::<Direct3D9>::new(this) };
    let obj: &mut Direct3D9 = &mut wrap;
    obj.refcount += 1;
    obj.refcount
}

extern "system" fn d3d9_release(this: *mut c_void) -> u32 {
    // SAFETY: D3D9 Release — same contract as AddRef above.
    // SAFETY: IDirect3D9 IUnknown thunk; D3D9 ABI guarantees `this` is *mut Direct3D9.
    let mut wrap = unsafe { VtableThis::<Direct3D9>::new(this) };
    let obj: &mut Direct3D9 = &mut wrap;
    obj.refcount -= 1;
    let rc = obj.refcount;
    if rc == 0 {
        // SAFETY: refcount reached zero; `this` is the original
        // `Box::into_raw(Direct3D9)` allocation from `Direct3DCreate9`,
        // and no other reference can survive a zero refcount.
        drop(unsafe { Box::from_raw(this.cast::<Direct3D9>()) });
    }
    rc
}

// ── IDirect3D9 methods ──

extern "system" fn d3d9_register_software_device(_this: *mut c_void, _init_fn: *mut c_void) -> i32 {
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3D9::RegisterSoftwareDevice → INVALIDCALL");
    D3DERR_INVALIDCALL
}

const extern "system" fn d3d9_get_adapter_count(_this: *mut c_void) -> u32 {
    1
}

extern "system" fn d3d9_get_adapter_identifier(
    _this: *mut c_void,
    adapter: u32,
    _flags: u32,
    id: *mut D3DADAPTER_IDENTIFIER9,
) -> i32 {
    trace!(target: LOG_TARGET, "IDirect3D9::GetAdapterIdentifier(adapter={adapter})");
    if adapter != 0 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable out-param; `id` is *mut D3DADAPTER_IDENTIFIER9 per IDirect3D9 ABI.
    let Some(mut id) = (unsafe { InPtrMut::<D3DADAPTER_IDENTIFIER9>::opt(id.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: D3DADAPTER_IDENTIFIER9 is a plain-data #[repr(C)] FFI struct
    // (fixed-size buffers + u32 fields); zeroed bytes are a valid value.
    unsafe { core::ptr::write_bytes(std::ptr::from_mut::<D3DADAPTER_IDENTIFIER9>(&mut id), 0, 1) };

    id.driver[..7].copy_from_slice(b"mtld3d\0");
    id.vendor_id = 0x106B; // Apple

    // GDI-style display-device name for adapter 0. D3D9 reports the adapter's
    // GDI name here; the conformance suite (and real apps enumerating adapters)
    // require it to be non-empty.
    let device_name = b"\\\\.\\DISPLAY1\0";
    id.device_name[..device_name.len()].copy_from_slice(device_name);

    let mut name_buf = [0u8; 256];
    let mut params = GetDeviceInfoParams {
        name_ptr: name_buf.as_mut_ptr() as u64,
        name_buf_len: 256,
        name_len: 0,
        registry_id: 0,
        supports_sampler_border: 0,
    };
    unix_call(&mut params);

    // Clamp to what `name_buf` actually holds (the unix side fills at most
    // `name_buf_len` bytes). `name_len` is the untruncated length and can exceed
    // the buffer, so clamping only to `id.description`'s size would slice
    // `name_buf` out of bounds. `description` (512 B) comfortably holds <= 256.
    let len = usize::try_from(params.name_len)
        .unwrap_or(usize::MAX)
        .min(name_buf.len());
    id.description[..len].copy_from_slice(&name_buf[..len]);
    // D3DADAPTER_IDENTIFIER9.device_id is u32 by D3D9 spec; mask to 16 bits.
    id.device_id = u32::try_from(params.registry_id & 0xFFFF).expect("16-bit mask fits u32");

    // `adapter.spoof`: report a consistent well-known GPU identity. Engines of
    // this era key whole render paths (depth copies, shadow filtering) off the
    // vendor id, sniff the description string for a marketing name, and gate
    // on a minimum driver version, so all of these move together.
    let spoof = match crate::config::CONFIG.adapter_spoof {
        mtld3d_core::config::AdapterSpoof::None => None,
        mtld3d_core::config::AdapterSpoof::Nvidia => Some(SpoofIdentity {
            vendor: 0x10DE,
            device: 0x0611, // GeForce 8800 GT, in every launch-era device table
            description: b"NVIDIA GeForce 8800 GT\0",
            driver: b"nvd3dum.dll\0",
            // 8.17.11.9745 (a WDDM 1.1 driver, NVIDIA 197.45) as
            // LARGE_INTEGER LowPart / HighPart. The first field matters:
            // 6.x is the XP driver model, and a title running on an NT 6
            // prefix can reject or mis-parse an XP-model version.
            version: [0x000B_2611, 0x0008_0011],
        }),
        mtld3d_core::config::AdapterSpoof::Amd => Some(SpoofIdentity {
            vendor: 0x1002,
            device: 0x9440, // Radeon HD 4870
            description: b"ATI Radeon HD 4800 Series\0",
            driver: b"atiumdag.dll\0",
            // 8.17.10.1129 (a WDDM 1.1 Catalyst driver) as LARGE_INTEGER
            // LowPart / HighPart; see the NVIDIA arm for why 8.x.
            version: [0x000A_0469, 0x0008_0011],
        }),
    };
    if let Some(s) = spoof {
        id.vendor_id = s.vendor;
        id.device_id = s.device;
        id.description = [0; 512];
        id.description[..s.description.len()].copy_from_slice(s.description);
        id.driver = [0; 512];
        id.driver[..s.driver.len()].copy_from_slice(s.driver);
        id.driver_version = s.version;
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "adapter.spoof: reporting vendor {:#06x} device {:#06x}", s.vendor, s.device
        );
    }

    0 // S_OK
}

/// Whether the Metal device can create border-colour samplers.
///
/// Queried once per process from the unix side; `GetDeviceCaps` strips
/// `D3DPTADDRESSCAPS_BORDER` when it cannot (virtualized CI devices), and
/// the unix sampler path clamps to edge for a title that ignores the cap.
pub fn sampler_border_supported() -> bool {
    static SUPPORTED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let mut params = GetDeviceInfoParams {
            name_ptr: 0,
            name_buf_len: 0,
            name_len: 0,
            registry_id: 0,
            supports_sampler_border: 0,
        };
        unix_call(&mut params);
        params.supports_sampler_border != 0
    });
    *SUPPORTED
}

/// One spoofed adapter identity: everything a game sniffs, kept consistent.
struct SpoofIdentity {
    vendor: u32,
    device: u32,
    description: &'static [u8],
    driver: &'static [u8],
    /// Win32 `LARGE_INTEGER` driver version as `LowPart` / `HighPart`.
    version: [u32; 2],
}

extern "system" fn d3d9_get_adapter_mode_count(
    _this: *mut c_void,
    adapter: u32,
    format: u32,
) -> u32 {
    if adapter != 0 || !is_display_format(format) {
        warn!(target: LOG_TARGET, "reject GetAdapterModeCount(adapter={adapter}, format={format}) → 0");
        return 0;
    }
    let count = u32::try_from(ADAPTER_MODES.iter().filter(|m| m.format == format).count())
        .expect("ADAPTER_MODES is a small static table");
    mtld3d_shared::log_once_trace_by!(
        target: DISPLAY_TRACE_TARGET,
        key: u64::from(format),
        "GetAdapterModeCount(format={format}) → {count}"
    );
    count
}

extern "system" fn d3d9_enum_adapter_modes(
    _this: *mut c_void,
    adapter: u32,
    format: u32,
    mode: u32,
    display_mode: *mut c_void,
) -> i32 {
    if adapter != 0 || display_mode.is_null() || !is_display_format(format) {
        warn!(
            target: LOG_TARGET,
            "reject EnumAdapterModes(adapter={adapter}, format={format}, mode={mode}) → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    let Some(entry) = ADAPTER_MODES
        .iter()
        .filter(|m| m.format == format)
        .nth(mode as usize)
    else {
        trace!(
            target: LOG_TARGET,
            "reject EnumAdapterModes(adapter={adapter}, format={format}, mode={mode}) → INVALIDCALL (out of range)"
        );
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable out-param; `display_mode` is *mut D3DDISPLAYMODE per IDirect3D9 ABI.
    unsafe { OutPtr::write_opt(display_mode.cast::<D3DDISPLAYMODE>(), *entry) };
    D3D_OK
}

extern "system" fn d3d9_get_adapter_display_mode(
    _this: *mut c_void,
    adapter: u32,
    mode: *mut c_void,
) -> i32 {
    if adapter != 0 || mode.is_null() {
        warn!(target: LOG_TARGET, "reject GetAdapterDisplayMode(adapter={adapter}) → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    // `ADAPTER_MODES[0]` is the host's native mode. Nothing we do changes the
    // desktop mode — a fullscreen device takes a monitor-covering window
    // instead — so the desktop is always in it.
    let current = ADAPTER_MODES[0];
    // SAFETY: vtable out-param; `mode` is *mut D3DDISPLAYMODE per IDirect3D9 ABI.
    unsafe { OutPtr::write_opt(mode.cast::<D3DDISPLAYMODE>(), current) };
    mtld3d_shared::log_once_trace_by!(
        target: DISPLAY_TRACE_TARGET,
        key: 0u64,
        "GetAdapterDisplayMode → {}x{}@{}Hz fmt={}",
        current.width, current.height, current.refresh_rate, current.format
    );
    D3D_OK
}

extern "system" fn d3d9_check_device_type(
    _this: *mut c_void,
    adapter: u32,
    dev_type: u32,
    adapter_format: u32,
    bb_format: u32,
    windowed: i32,
) -> i32 {
    if adapter != 0 || dev_type != D3DDEVTYPE_HAL || !is_display_format(adapter_format) {
        warn!(
            target: LOG_TARGET,
            "reject CheckDeviceType(adapter={adapter}, dev_type={dev_type}, adapter_fmt={adapter_format}, bb_fmt={bb_format}, windowed={windowed}) → NOTAVAILABLE"
        );
        return D3DERR_NOTAVAILABLE;
    }
    // Windowed mode accepts D3DFMT_UNKNOWN as "use the display format".
    let effective_bb = if windowed != 0 && bb_format == 0 {
        adapter_format
    } else {
        bb_format
    };
    // The backbuffer must be a renderable colour surface, and presentable to
    // the display format: in windowed mode via a supported format conversion
    // (the same predicate `CheckDeviceFormatConversion` answers with, so the
    // two agree for every pair); in fullscreen it must match the display
    // format's colour family directly. A 16-bit windowed backbuffer is
    // therefore advertised; `CreateDevice` substitutes the BGRA8 layer format
    // for it (`warn_unsupported_backbuffer_format`).
    let presentable = is_render_target_format(effective_bb)
        && if windowed != 0 {
            is_format_conversion_supported(effective_bb, adapter_format)
        } else {
            is_present_compatible(effective_bb, adapter_format)
        };
    if !presentable {
        trace!(
            target: LOG_TARGET,
            "reject CheckDeviceType(adapter_fmt={adapter_format}, bb_fmt={bb_format}, windowed={windowed}) → NOTAVAILABLE"
        );
        return D3DERR_NOTAVAILABLE;
    }
    mtld3d_shared::log_once_trace_by!(
        target: DISPLAY_TRACE_TARGET,
        key: (u64::from(adapter_format) << 32) | u64::from(bb_format),
        "CheckDeviceType(adapter_fmt={adapter_format}, bb_fmt={bb_format}, windowed={windowed}) → OK"
    );
    D3D_OK
}

/// The 'RESZ' pseudo-format fourcc: probed to detect the depth-resolve hack.
const D3DFMT_RESZ: u32 = 0x5A53_4552;

extern "system" fn d3d9_check_device_format(
    _this: *mut c_void,
    adapter: u32,
    dev_type: u32,
    adapter_format: u32,
    usage: u32,
    rtype: u32,
    check_format: u32,
) -> i32 {
    // One line per distinct probe: which formats a title asks about (and in
    // what usage/rtype shape) is the map of the render path it is choosing.
    mtld3d_shared::log_once_debug_by!(
        target: LOG_TARGET,
        key: (u64::from(usage) << 40) ^ (u64::from(rtype) << 32) ^ u64::from(check_format),
        "CheckDeviceFormat probe: usage={usage:#x} rtype={rtype} fmt={check_format:#x}"
    );
    // A D3DFMT_UNKNOWN (0) adapter format is never a valid query — the runtime
    // rejects it with INVALIDCALL ahead of any availability check, for every
    // device type.
    if adapter_format == 0 {
        return D3DERR_INVALIDCALL;
    }
    if adapter != 0 || dev_type != D3DDEVTYPE_HAL || !is_display_format(adapter_format) {
        warn!(
            target: LOG_TARGET,
            "reject CheckDeviceFormat(adapter={adapter}, dev_type={dev_type}, adapter_fmt={adapter_format}, usage={usage:#x}, rtype={rtype}, check_fmt={check_format}) → NOTAVAILABLE"
        );
        return D3DERR_NOTAVAILABLE;
    }
    // D3DFMT_UNKNOWN is the "no format" sentinel — spec-correct to reject, and
    // games routinely probe it, so don't clutter the log with it.
    if check_format == 0 {
        return D3DERR_NOTAVAILABLE;
    }
    // `caps.dfFormats = false` hides the DF fourccs (INTZ stays): an engine
    // finding both DF and INTZ can pick a mixed depth path no real GPU had.
    if matches!(check_format, D3DFMT_DF24 | D3DFMT_DF16) && !crate::config::CONFIG.df_formats {
        return D3DERR_NOTAVAILABLE;
    }
    // The RESZ pseudo-format: probing it asks "is the RESZ depth resolve
    // supported" (`SetRenderState(POINTSIZE, 0x7fa05000)`, implemented in
    // the device). No surface of this format is ever created.
    if check_format == D3DFMT_RESZ {
        return if rtype == D3DRTYPE_SURFACE && usage & D3DUSAGE_RENDERTARGET != 0 {
            D3D_OK
        } else {
            D3DERR_NOTAVAILABLE
        };
    }
    // Vertex texture fetch: any sampleable texture format can be read from
    // the vertex stage (Metal binds textures to vertex functions natively),
    // matching the non-zero `VertexTextureFilterCaps`. Strip the bit and
    // let the remaining usage bits evaluate normally, so combined queries
    // (RENDERTARGET | QUERY_VERTEXTEXTURE, the render-then-fetch pattern)
    // answer on their other halves.
    let usage = if usage & D3DUSAGE_QUERY_VERTEXTEXTURE != 0 {
        if !is_texture_format(check_format) {
            return D3DERR_NOTAVAILABLE;
        }
        usage & !D3DUSAGE_QUERY_VERTEXTEXTURE
    } else {
        usage
    };
    // D3DUSAGE_QUERY_SRGBWRITE asks whether a format works as an sRGB-encoding
    // render target. On a plain offscreen SURFACE — which can never be a render
    // target without D3DUSAGE_RENDERTARGET — the combination is invalid. (A
    // TEXTURE is a meaningful SRGBWRITE query even without the bit, since it can
    // later be bound as a render target, so it falls through to the cap check.)
    if rtype == D3DRTYPE_SURFACE
        && usage & D3DUSAGE_QUERY_SRGBWRITE != 0
        && usage & D3DUSAGE_RENDERTARGET == 0
    {
        return D3DERR_NOTAVAILABLE;
    }
    // D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING asks whether the format blends
    // as a render target. Every colour attachment blends on Metal, so the
    // answer is the render-target question itself, whether or not the caller
    // also passed D3DUSAGE_RENDERTARGET.
    if usage & D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING != 0
        && !is_render_target_format(check_format)
    {
        return D3DERR_NOTAVAILABLE;
    }
    let supported = if rtype == D3DRTYPE_CUBETEXTURE {
        if usage & D3DUSAGE_DEPTHSTENCIL != 0 {
            false
        } else if usage & D3DUSAGE_RENDERTARGET != 0 {
            is_render_target_format(check_format)
                && (usage & D3DUSAGE_QUERY_SRGBWRITE == 0 || has_srgb_twin(check_format))
        } else if usage & D3DUSAGE_QUERY_SRGBREAD != 0 && !has_srgb_read_decode(check_format) {
            false
        } else {
            is_cube_texture_format(check_format)
        }
    } else if usage & D3DUSAGE_DEPTHSTENCIL != 0 {
        is_depth_stencil_format(check_format)
    } else if usage & D3DUSAGE_RENDERTARGET != 0 {
        // SRGBWRITE is only meaningful for render-targetable colour
        // formats. Restrict to formats whose Metal twin has an sRGB
        // pair, otherwise the caller is asking "can I write linear
        // through an sRGB-encoded RT view?" which we can't honour.
        if usage & D3DUSAGE_QUERY_SRGBWRITE != 0 && !has_srgb_twin(check_format) {
            false
        } else {
            is_render_target_format(check_format)
        }
    } else if rtype == D3DRTYPE_SURFACE || rtype == D3DRTYPE_TEXTURE {
        // SRGBREAD: per-format gate for whether the runtime/game can
        // ask for `D3DSAMP_SRGBTEXTURE=1`. Matches the eager sRGB-view
        // path in `unix/unix/src/metal/texture.rs::create_texture`
        // — only formats with an MTLPixelFormat sRGB twin succeed.
        if usage & D3DUSAGE_QUERY_SRGBREAD != 0 && !has_srgb_read_decode(check_format) {
            false
        } else {
            is_texture_format(check_format)
        }
    } else {
        false
    };
    if !supported {
        trace!(
            target: LOG_TARGET,
            "reject CheckDeviceFormat(adapter_fmt={adapter_format}, usage={usage:#x}, rtype={rtype}, check_fmt={check_format}) → NOTAVAILABLE"
        );
        return D3DERR_NOTAVAILABLE;
    }
    // D3DUSAGE_AUTOGENMIPMAP needs render-target capability even when the
    // query does not include D3DUSAGE_RENDERTARGET. Metal mip generation is
    // available for renderable 2D and cube color formats.
    if usage & D3DUSAGE_AUTOGENMIPMAP != 0 && !is_render_target_format(check_format) {
        return D3DOK_NOAUTOGEN;
    }
    mtld3d_shared::log_once_debug_by!(
        target: DISPLAY_TRACE_TARGET,
        key: (u64::from(adapter_format) << 48) | (u64::from(usage) << 32) | (u64::from(rtype) << 16) | u64::from(check_format),
        "CheckDeviceFormat(adapter_fmt={adapter_format}, usage={usage:#x}, rtype={rtype}, check_fmt={check_format}) → OK"
    );
    D3D_OK
}

extern "system" fn d3d9_check_device_multi_sample_type(
    _this: *mut c_void,
    adapter: u32,
    _dev_type: u32,
    _surface_format: u32,
    _windowed: i32,
    multi_sample_type: u32,
    quality_levels: *mut u32,
) -> i32 {
    // SAFETY: vtable out-param; `quality_levels` is *mut u32 per IDirect3D9 ABI.
    unsafe { OutPtr::write_opt(quality_levels, 1) };
    if adapter != 0 {
        warn!(
            target: LOG_TARGET,
            "reject CheckDeviceMultiSampleType: adapter={adapter} → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    // D3D9 advertises "no MSAA" by returning NOTAVAILABLE for every non-NONE
    // sample type. Games poll all 16 levels at init, so logging the expected
    // "no" would spam WARN — this is the spec contract, not a fallback.
    if multi_sample_type != D3DMULTISAMPLE_NONE {
        return D3DERR_NOTAVAILABLE;
    }
    D3D_OK
}

extern "system" fn d3d9_check_depth_stencil_match(
    _this: *mut c_void,
    adapter: u32,
    dev_type: u32,
    adapter_format: u32,
    rt_format: u32,
    ds_format: u32,
) -> i32 {
    if adapter != 0
        || dev_type != D3DDEVTYPE_HAL
        || !is_display_format(adapter_format)
        || !is_render_target_format(rt_format)
        || !is_depth_stencil_format(ds_format)
    {
        warn!(
            target: LOG_TARGET,
            "reject CheckDepthStencilMatch(adapter={adapter}, dev_type={dev_type}, adapter_fmt={adapter_format}, rt_fmt={rt_format}, ds_fmt={ds_format}) → NOTAVAILABLE"
        );
        return D3DERR_NOTAVAILABLE;
    }
    // Mirror the CheckDeviceFormat gate: hidden DF fourccs stay hidden here.
    if matches!(ds_format, D3DFMT_DF24 | D3DFMT_DF16) && !crate::config::CONFIG.df_formats {
        return D3DERR_NOTAVAILABLE;
    }
    D3D_OK
}

extern "system" fn d3d9_check_device_format_conversion(
    _this: *mut c_void,
    adapter: u32,
    dev_type: u32,
    source_format: u32,
    target_format: u32,
) -> i32 {
    if adapter != 0
        || dev_type != D3DDEVTYPE_HAL
        || !is_format_conversion_supported(source_format, target_format)
    {
        trace!(
            target: LOG_TARGET,
            "reject CheckDeviceFormatConversion(adapter={adapter}, dev_type={dev_type}, src_fmt={source_format}, dst_fmt={target_format}) → NOTAVAILABLE"
        );
        return D3DERR_NOTAVAILABLE;
    }
    D3D_OK
}

extern "system" fn d3d9_get_device_caps(
    _this: *mut c_void,
    adapter: u32,
    _device_type: u32,
    caps: *mut D3DCAPS9,
) -> i32 {
    trace!(target: LOG_TARGET, "IDirect3D9::GetDeviceCaps(adapter={adapter})");
    if adapter != 0 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable out-param; `caps` is *mut D3DCAPS9 per IDirect3D9 ABI.
    let Some(mut caps) = (unsafe { InPtrMut::<D3DCAPS9>::opt(caps.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    caps::fill(
        &mut caps,
        crate::config::CONFIG.caps_all,
        sampler_border_supported(),
    );
    0 // S_OK
}

extern "system" fn d3d9_get_adapter_monitor(_this: *mut c_void, _adapter: u32) -> *mut c_void {
    // Single-adapter model: the primary display's monitor. GetMonitorInfo on
    // the result reports MONITORINFOF_PRIMARY, as the D3D9 spec requires for
    // adapter 0.
    crate::fullscreen::primary_monitor()
}

#[link(name = "user32")]
unsafe extern "system" {
    fn GetClientRect(hwnd: *mut c_void, rect: *mut Rect) -> i32;
}

/// Client-area pixel dimensions of `hwnd`, or `None` when the call fails or the rect is empty.
///
/// The single `GetClientRect` boundary is concentrated here so the call site
/// stays unsafe-free.
fn client_rect_dims(hwnd: *mut c_void) -> Option<(u32, u32)> {
    if hwnd.is_null() {
        return None;
    }
    let mut rect = Rect::EMPTY;
    // SAFETY: GetClientRect accepts any HWND and writes a RECT through the
    // out pointer; `rect` is an owned local, so non-null + aligned + writable
    // holds. A bad HWND yields a zero return, handled below.
    let ok = unsafe { GetClientRect(hwnd, &raw mut rect) };
    if ok == 0 {
        return None;
    }
    let w = u32::try_from(rect.width()).ok()?;
    let h = u32::try_from(rect.height()).ok()?;
    (w != 0 && h != 0).then_some((w, h))
}

/// `true` when `width`x`height` is a mode `EnumAdapterModes` serves.
///
/// The membership test behind the fullscreen honor-or-follow split in
/// [`resolve_backbuffer_dims`]. Format-agnostic: the mode table repeats every
/// size once per adapter format.
pub fn is_enumerable_mode(width: u32, height: u32) -> bool {
    ADAPTER_MODES
        .iter()
        .any(|m| m.width == width && m.height == height)
}

/// Resolve the back buffer's *logical* size.
///
/// Logical size is what D3D9 reports and the space every game-supplied
/// coordinate lives in. Three rules, keyed on who decides the resolution:
///
/// - **Fullscreen, requesting an enumerable mode**: the request stands,
///   exactly as it would under a native mode-set, so viewports and scissors
///   the game sizes from its own request cover the frame. The window still
///   covers the monitor and present resolves the size difference at the
///   drawable (`MetalFX` when enlarging), the same resample `render.scale`
///   rides. A request that is *not* an enumerable mode is one native would
///   reject outright, so no game can depend on it being honored; such games
///   carry their window size into the request and size their rendering and
///   input from the window, so the client rect wins there — the lenient
///   answer that keeps the window, back buffer and mouse in one space.
/// - **Maximized window**: the window manager sizes the window, not the game,
///   so the client area wins and the requested resolution is ignored;
///   `render.scale` is the resolution control in that mode.
/// - **Ordinary window**: the game's explicit request stands. Only the D3D9
///   "a zero dimension means the client area" rule is applied, so a zeroed
///   present-params struct (the conformance `stateblock` device, additional
///   swap chains) never forwards a 0-dimension texture descriptor to Metal.
///
/// A window whose client rect cannot be read leaves the request untouched; a
/// dimension still zero afterwards is rejected by the caller.
pub fn resolve_backbuffer_dims(hwnd: u64, pp: &mut D3DPRESENT_PARAMETERS) {
    if pp.windowed == 0 {
        // Callers reject a zero-dimension fullscreen request before the window
        // moves, so the request is always concrete here.
        let client = client_rect_dims(hwnd as *mut c_void);
        if is_enumerable_mode(pp.back_buffer_width, pp.back_buffer_height) {
            // The log line is the breadcrumb tying an upscaled frame back to
            // the size the game asked for; a client rect that cannot be read
            // only costs that line.
            if let Some((client_w, client_h)) = client
                && (pp.back_buffer_width != client_w || pp.back_buffer_height != client_h)
            {
                mtld3d_shared::log_once_info!(
                    target: LOG_TARGET,
                    "fullscreen device: honoring the requested {}x{} back buffer; the window \
                     covers the monitor ({}x{}) and present scales the frame",
                    pp.back_buffer_width, pp.back_buffer_height, client_w, client_h,
                );
            }
            return;
        }
        let Some((client_w, client_h)) = client else {
            return;
        };
        if pp.back_buffer_width != client_w || pp.back_buffer_height != client_h {
            mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "fullscreen device: requested {}x{} is not an enumerable display mode, so the \
                 back buffer follows the window ({}x{}) instead",
                pp.back_buffer_width, pp.back_buffer_height, client_w, client_h,
            );
        }
        pp.back_buffer_width = client_w;
        pp.back_buffer_height = client_h;
        return;
    }
    let Some((client_w, client_h)) = client_rect_dims(hwnd as *mut c_void) else {
        return;
    };
    if crate::fullscreen::is_maximized(hwnd as *mut c_void) {
        if pp.back_buffer_width != client_w || pp.back_buffer_height != client_h {
            mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "maximized device: back buffer follows the window ({}x{}), requested {}x{} \
                 ignored; use render.scale to pick the render resolution",
                client_w, client_h, pp.back_buffer_width, pp.back_buffer_height,
            );
        }
        pp.back_buffer_width = client_w;
        pp.back_buffer_height = client_h;
        return;
    }
    if pp.back_buffer_width == 0 {
        pp.back_buffer_width = client_w;
    }
    if pp.back_buffer_height == 0 {
        pp.back_buffer_height = client_h;
    }
}

extern "system" fn d3d9_create_device(
    this: *mut c_void,
    adapter: u32,
    dev_type: u32,
    focus_window: *mut c_void,
    behavior_flags: u32,
    present_params: *mut c_void,
    device: *mut *mut c_void,
) -> i32 {
    crate::USED.store(true, std::sync::atomic::Ordering::Relaxed);

    if adapter != 0 || device.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable in/out-param; per the D3D9 ABI `present_params` points to a
    // readable+writable `D3DPRESENT_PARAMETERS` — CreateDevice resolves and
    // reports the effective geometry back through it.
    let Some(mut pp_in) = (unsafe { InPtrMut::<D3DPRESENT_PARAMETERS>::opt(present_params) })
    else {
        return D3DERR_INVALIDCALL;
    };
    // Own a mutable copy so a windowed zero-dimension request can be resolved
    // against the device window's client area (below) and the resolved size
    // flows uniformly to the layer, backbuffer, and depth/stencil creates.
    let mut pp = *pp_in;

    // Reject invalid swap-effect / back-buffer-count / presentation-interval
    // combinations up front, before any Metal resource is created.
    if !crate::device::present_params_are_valid(&pp) {
        warn!(
            target: LOG_TARGET,
            "reject CreateDevice — invalid present params (swap_effect={}, bb_count={}, interval={:#x})",
            pp.swap_effect, pp.back_buffer_count, pp.presentation_interval,
        );
        return D3DERR_INVALIDCALL;
    }

    // Create Metal device + command queue
    let mut cq_params = CreateCommandQueueParams {
        device_handle: MetalHandle::NULL,
        queue_handle: MetalHandle::NULL,
        unified_memory: 0,
        min_linear_texture_align: 0,
    };
    let status = unix_call(&mut cq_params);
    if status != 0 {
        error!(target: LOG_TARGET, "CreateCommandQueue failed (0x{status:08X})");
        return D3DERR_INVALIDCALL;
    }

    // Determine HWND for layer attachment
    let hwnd = if pp.device_window != 0 {
        pp.device_window as u64
    } else {
        focus_window as u64
    };

    // A fullscreen request must still be well-formed even though its size is
    // not used: the D3D9 "zero means the client area" rule is windowed-only,
    // so zero dimensions here are a malformed request. Checked before the
    // window moves, so a rejected create leaves it untouched.
    if pp.windowed == 0 && (pp.back_buffer_width == 0 || pp.back_buffer_height == 0) {
        warn!(
            target: LOG_TARGET,
            "reject CreateDevice({}x{}) — a fullscreen request may not carry zero dimensions",
            pp.back_buffer_width, pp.back_buffer_height,
        );
        destroy_partial_device(&cq_params, MetalHandle::NULL, MetalHandle::NULL);
        return D3DERR_INVALIDCALL;
    }

    // A fullscreen device owns its window. Take it over first: the back-buffer
    // size is resolved from the resulting client rect, and the metal view is
    // sized from the window Wine hands us at attach time, so both have to see
    // the window already covering the monitor.
    let fullscreen = (pp.windowed == 0).then(|| crate::fullscreen::enter(hwnd as *mut c_void));

    // Resolve the logical backbuffer size against the window now that its
    // geometry is final, before any Metal resource is sized from it.
    resolve_backbuffer_dims(hwnd, &mut pp);
    // Resolve D3DFMT_UNKNOWN to the display format and a zero back-buffer count
    // to one, so the geometry written back to the caller's present params is
    // concrete.
    if pp.windowed != 0 && pp.back_buffer_format == 0 {
        pp.back_buffer_format = adapter_display_format();
    }
    pp.back_buffer_count = pp.back_buffer_count.max(1);

    warn_unsupported_backbuffer_format(pp.back_buffer_format);
    crate::device::warn_present_params_fields_once(&pp);

    let layer_params = attach_metal_layer(hwnd, &cq_params, &pp);

    // A still-zero dimension here (no usable client rect, or a fullscreen
    // request with zero dims) would abort Metal's texture validation. Reject
    // it as INVALIDCALL instead, matching `device_reset`.
    if pp.back_buffer_width == 0 || pp.back_buffer_height == 0 {
        warn!(
            target: LOG_TARGET,
            "reject CreateDevice — zero backbuffer dims (windowed={}, hwnd=0x{hwnd:x})",
            pp.windowed,
        );
        destroy_partial_device(&cq_params, layer_params.view_handle, MetalHandle::NULL);
        restore_from_fullscreen(fullscreen.as_ref());
        return D3DERR_INVALIDCALL;
    }
    let (cursor_scale, scale_origin) = resolve_initial_cursor_scale(layer_params.backing_scale);
    info!(
        target: LOG_TARGET,
        "hardware cursor scale: {cursor_scale}x ({scale_origin})"
    );

    // `render.scale` splits the back buffer in two from here on: `pp` keeps
    // the logical size D3D9 reports, while the Metal texture is rasterized at
    // `render_scale` of it and MetalFX resamples on present. Without MetalFX
    // there is nothing that could resample, so the scale is forced to
    // identity rather than silently presenting a mis-sized frame.
    let render_scale = resolve_render_scale(layer_params.metalfx_available != 0);
    let render_width = render_scale.dimension(pp.back_buffer_width);
    let render_height = render_scale.dimension(pp.back_buffer_height);

    // Create backbuffer texture
    let mut bb_params = CreateBackbufferParams {
        device_handle: cq_params.device_handle,
        queue_handle: cq_params.queue_handle,
        width: render_width,
        height: render_height,
        texture_handle: MetalHandle::NULL,
    };
    let status = unix_call(&mut bb_params);
    if status != 0 {
        error!(target: LOG_TARGET, "CreateBackbuffer failed (0x{status:08X})");
        destroy_partial_device(&cq_params, layer_params.view_handle, MetalHandle::NULL);
        restore_from_fullscreen(fullscreen.as_ref());
        return D3DERR_INVALIDCALL;
    }

    // Create depth/stencil texture if requested
    let depth_handle = match create_auto_depth_stencil(&cq_params, &layer_params, &bb_params, &pp) {
        Ok(handle) => handle,
        Err(hr) => {
            restore_from_fullscreen(fullscreen.as_ref());
            return hr;
        }
    };

    let mut render_states = mtld3d_types::render_state_defaults();
    if depth_handle.is_null() {
        render_states[mtld3d_types::D3DRS_ZENABLE as usize] = 0;
    }

    addref_parent_direct3d9(this);
    spawn_tsc_warmup();
    let (encoder, prewarm) = spawn_encoder_and_prewarm(&cq_params);

    let dev = Direct3DDevice9::new(crate::device::DeviceCreateInfo {
        device_handle: cq_params.device_handle,
        queue_handle: cq_params.queue_handle,
        view_handle: layer_params.view_handle,
        layer_handle: layer_params.layer_handle,
        backbuffer_handle: bb_params.texture_handle,
        depth_stencil_handle: depth_handle,
        depth_stencil_format: if depth_handle.is_null() {
            0
        } else {
            pp.auto_depth_stencil_format
        },
        backbuffer_width: pp.back_buffer_width,
        backbuffer_height: pp.back_buffer_height,
        render_scale,
        encoder,
        prewarm,
        current_frame: FrameData::new(&FrameInit {
            device_handle: cq_params.device_handle,
            queue_handle: cq_params.queue_handle,
            backbuffer_handle: bb_params.texture_handle,
            layer_handle: layer_params.layer_handle,
            view_handle: layer_params.view_handle,
            // Logical, paired with the scale below: `PassState::reset_frame`
            // derives the rasterized extent from the two.
            backbuffer_width: pp.back_buffer_width,
            backbuffer_height: pp.back_buffer_height,
            backbuffer_format: mtld3d_shared::mtl::PixelFormat::Bgra8Unorm,
            render_scale,
            depth_texture: depth_handle,
            depth_has_stencil: depth_format_has_stencil(pp.auto_depth_stencil_format),
            apply_display_sync_enabled: None,
        }),
        render_states,
        sampler_states: [mtld3d_types::sampler_state_defaults(); STAGE_COUNT],
        direct3d: this as u64,
        creation_adapter: adapter,
        creation_device_type: dev_type,
        creation_behavior_flags: behavior_flags,
        creation_focus_window: focus_window as usize,
        present_params: {
            // The implicit swapchain reports a back-buffer count of at least
            // one (D3D9 treats a requested 0 as 1) and resolves a NULL
            // hDeviceWindow to the real target window, so GetPresentParameters
            // hands back concrete values.
            let mut stored = pp;
            stored.back_buffer_count = stored.back_buffer_count.max(1);
            if stored.device_window == 0 {
                // A NULL device window resolves to the focus window — the same
                // resolution `hwnd` used above (pointer→usize, no truncation).
                stored.device_window = focus_window as usize;
            }
            stored
        },
        hwnd: hwnd as *mut c_void,
        cursor_scale,
        fullscreen,
    });

    // Install the cursor wndproc subclass. Must happen after `DeviceInner` is
    // boxed so the subclass's global back-pointer resolves to a live device.
    let inner_ptr = std::ptr::from_mut::<crate::device::DeviceInner>(dev.inner());
    // SAFETY: `inner_ptr` was just derived from a live `DeviceInner` we
    // own via `dev`; the borrow is local to this expression and `dev`
    // outlives it.
    unsafe { (*inner_ptr).cursor_mut().install_subclass(inner_ptr) };

    let dev_ptr = Box::into_raw(Box::new(dev));
    // Stamp the wrapper pointer so resource `GetDevice` thunks can hand it back
    // (AddRef'd) instead of leaving the caller's out-param uninitialised.
    // SAFETY: `dev_ptr` is a freshly-boxed, live `Direct3DDevice9`.
    unsafe {
        (*dev_ptr)
            .inner()
            .set_device_wrapper(dev_ptr.cast::<c_void>());
    };
    // Report the resolved geometry back to the caller. D3D9 leaves hDeviceWindow
    // and the mode flags as the caller set them.
    pp_in.back_buffer_width = pp.back_buffer_width;
    pp_in.back_buffer_height = pp.back_buffer_height;
    pp_in.back_buffer_count = pp.back_buffer_count;
    pp_in.back_buffer_format = pp.back_buffer_format;
    // SAFETY: vtable out-param; `device` is *mut *mut c_void per IDirect3D9 ABI.
    unsafe { OutPtr::write_opt(device, dev_ptr.cast::<c_void>()) };
    info!(target: LOG_TARGET, "CreateDevice succeeded");
    D3D_OK
}

/// Attach a `CAMetalLayer` to the game window.
///
/// Optional — `hwnd == 0` produces a fully-initialised
/// `AttachMetalLayerParams` with `view_handle == 0`, which the rest of
/// `CreateDevice` treats as "no presentation surface" rather than an error.
/// Failures to attach when an HWND is present are also non-fatal: the device
/// works, but Present is a no-op.
fn attach_metal_layer(
    hwnd: u64,
    cq: &CreateCommandQueueParams,
    pp: &D3DPRESENT_PARAMETERS,
) -> AttachMetalLayerParams {
    let display_sync_enabled = crate::device::resolve_display_sync(pp.presentation_interval);
    let mut layer_params = AttachMetalLayerParams {
        hwnd,
        device_handle: cq.device_handle,
        width: pp.back_buffer_width,
        height: pp.back_buffer_height,
        view_handle: MetalHandle::NULL,
        layer_handle: MetalHandle::NULL,
        backing_scale: 1,
        display_sync_enabled: u32::from(display_sync_enabled),
        hdr_enable: u32::from(crate::config::CONFIG.hdr_enable),
        color_space: crate::config::CONFIG.color_space,
        max_fps: crate::config::CONFIG.present_max_fps,
        metalfx_available: 0,
    };
    if hwnd != 0 {
        unix_call(&mut layer_params);
    }
    layer_params
}

/// Retain the parent `IDirect3D9` for `IDirect3DDevice9::GetDirect3D`.
///
/// The call hands back the same interface with `AddRef` semantics rather than
/// a dangling handle after the caller Releases its outer reference.
fn addref_parent_direct3d9(this: *mut c_void) {
    if !this.is_null() {
        // SAFETY: IDirect3D9 IUnknown thunk; D3D9 ABI guarantees `this` is *mut Direct3D9.
        let mut parent_wrap = unsafe { VtableThis::<Direct3D9>::new(this) };
        let parent: &mut Direct3D9 = &mut parent_wrap;
        parent.refcount += 1;
    }
}

/// Warm the TSC calibration in the background.
///
/// The encoder thread's first 5-second-window check then finds a ready
/// `tsc_hz()` value instead of paying the 50 ms calibration sleep itself.
/// Deliberately not spawned from `DllMain` or `Direct3DCreate9`: mod /
/// launcher DLLs commonly probe-call `Direct3DCreate9` early enough that the
/// spawned thread's stdlib thread-entry (TLS, `env_logger` lazy init) still
/// races the host process's own init and can blow a 2 MB Wine stack or fault
/// with a corrupt TEB. `CreateDevice` runs past all of that.
/// `tsc_hz()` is internally latched by a `OnceLock`, so a second
/// `CreateDevice` call just returns the cached value.
fn spawn_tsc_warmup() {
    let _ = std::thread::Builder::new()
        .name("mtld3d-tsc-warmup".into())
        .spawn(|| {
            let _ = mtld3d_shared::tsc::tsc_hz();
        });
}

/// Spawn the encoder thread plus the shader-cache pre-warm thread.
///
/// Reads `<host-exe-dir>/mtld3d_shaders.bin`, compiles every cached MSL
/// via the existing `CompileShaderLibrary` thunk, and ships the
/// `MTLLibrary` handles to the encoder over the dedicated prewarm
/// channel. The encoder blocks on that channel before draining its first
/// `EncoderMessage`, so live miss-compiles can never race the prewarm.
/// Cold launch (no file) sends an empty payload — that's still the
/// "cache file is fresh, you may start writing" signal the encoder needs
/// to flip `cache_ready`.
fn spawn_encoder_and_prewarm(
    cq: &CreateCommandQueueParams,
) -> (EncoderThread, crate::shader_prewarm::PrewarmHandle) {
    let gpu_caps = mtld3d_core::gpu_caps::GpuCaps {
        unified_memory: cq.unified_memory != 0,
        min_linear_texture_align: cq.min_linear_texture_align,
    };
    let encoder = EncoderThread::spawn(gpu_caps);
    let prewarm = crate::shader_prewarm::spawn(cq.device_handle, encoder.prewarm_sender());
    (encoder, prewarm)
}

/// `CAMetalLayer.pixelFormat` and the backbuffer are hardcoded to `BGRA8Unorm` on the unix side.
///
/// That matches `D3DFMT_A8R8G8B8` / `D3DFMT_X8R8G8B8` byte-for-byte, which is
/// all `WoW` requests. Windowed `CheckDeviceType` advertises the 16-bit and
/// float backbuffer formats too (it answers with the `StretchRect` conversion
/// predicate, as the runtime requires), and such a request is substituted by
/// decision rather than plumbed: a real 16-bit backbuffer would need a
/// conversion pass on every present, and a float one would need the whole
/// present path to carry a format (a second drawable format, a present
/// pipeline per format, and format-derived read-back pitches). Games that
/// render HDR internally do it in their own off-screen float targets and
/// tone-map into an 8-bit backbuffer, so nothing has needed it. Warn once so a
/// game that asked shows up.
fn warn_unsupported_backbuffer_format(format: u32) {
    if !matches!(format, D3DFMT_A8R8G8B8 | D3DFMT_X8R8G8B8) {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "CreateDevice: back_buffer_format {format:#x} requested but layer/backbuffer is hardcoded BGRA8Unorm — substituting"
        );
    }
}

/// Match the hardware cursor bitmap to the display's retina factor by default.
///
/// Wine's HCURSOR path then produces a proportionally-sized pointer on a
/// retina Mac. `cursor.scale` in `mtld3d.conf` overrides: `auto` (the default)
/// follows `backingScaleFactor`; a positive integer forces a fixed multiplier.
/// Both paths clamp to `[1, 8]` — the downstream HCURSOR builder asserts that
/// range.
fn resolve_initial_cursor_scale(backing_scale: u32) -> (u32, &'static str) {
    match crate::config::CONFIG.cursor_scale {
        mtld3d_core::config::CursorScale::Auto => (
            backing_scale.clamp(1, 8),
            "auto from display backingScaleFactor",
        ),
        mtld3d_core::config::CursorScale::Fixed(n) => (n.clamp(1, 8), "cursor.scale override"),
    }
}

/// Resolve `render.scale` against what the GPU can actually do.
///
/// `MetalFX` is the only thing that can resample a frame at present time, so a
/// GPU without it has to render at exactly the presented size. Say so once
/// rather than quietly ignoring the user's setting.
fn resolve_render_scale(metalfx_available: bool) -> mtld3d_core::render_scale::RenderScale {
    use mtld3d_core::render_scale::RenderScale;

    let requested = RenderScale::from_percent(crate::config::CONFIG.render_scale_percent);
    if requested.is_identity() {
        return RenderScale::IDENTITY;
    }
    if !metalfx_available {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "render.scale = {} ignored — this GPU has no MetalFX spatial upscaler, so the \
             frame is rendered at full resolution",
            f64::from(requested.percent()) / 100.0,
        );
        return RenderScale::IDENTITY;
    }
    info!(
        target: LOG_TARGET,
        "render.scale = {} — rendering at {}% of the presented resolution",
        f64::from(requested.percent()) / 100.0,
        requested.percent(),
    );
    requested
}

/// Give the window back when device creation fails mid-way.
///
/// A fullscreen `CreateDevice` takes the window over before it builds
/// anything, so every later failure has to undo it. Otherwise a rejected
/// device leaves the game's window stripped of its decoration and pinned over
/// the monitor.
fn restore_from_fullscreen(saved: Option<&crate::fullscreen::SavedWindow>) {
    if let Some(saved) = saved {
        crate::fullscreen::leave(saved);
    }
}

/// Tear down the partial device handles assembled so far.
///
/// Called on any failure between `CreateCommandQueue` and the final
/// `Box::into_raw`. `view_handle` / `backbuffer_handle` are `MetalHandle::NULL`
/// when the failure happens before that object was created.
fn destroy_partial_device(
    cq: &CreateCommandQueueParams,
    view_handle: MetalHandle<NSViewKind>,
    backbuffer_handle: MetalHandle<MTLTextureKind>,
) {
    let mut destroy = DestroyCommandQueueParams {
        device_handle: cq.device_handle,
        queue_handle: cq.queue_handle,
        view_handle,
        backbuffer_handle,
        pipeline_handle: MetalHandle::NULL,
        depth_texture_handle: MetalHandle::NULL,
    };
    unix_call(&mut destroy);
}

/// Create the auto depth/stencil texture if the present params requested one.
///
/// Returns `Ok(MetalHandle::NULL)` when no depth was requested, `Ok(handle)`
/// on success, or `Err(hr)` after tearing down the partial device.
fn create_auto_depth_stencil(
    cq_params: &CreateCommandQueueParams,
    layer_params: &AttachMetalLayerParams,
    bb_params: &CreateBackbufferParams,
    pp: &D3DPRESENT_PARAMETERS,
) -> Result<MetalHandle<MTLTextureKind>, i32> {
    if pp.enable_auto_depth_stencil == 0 || pp.auto_depth_stencil_format == 0 {
        return Ok(MetalHandle::NULL);
    }
    let Some(ds_pixel_format) =
        mtld3d_core::format::map_d3d_depth_format(pp.auto_depth_stencil_format)
    else {
        error!(
            target: LOG_TARGET,
            "auto depth-stencil format {} has no Metal mapping",
            pp.auto_depth_stencil_format
        );
        destroy_partial_device(
            cq_params,
            layer_params.view_handle,
            bb_params.texture_handle,
        );
        return Err(D3DERR_INVALIDCALL);
    };
    // Sized from the back buffer rather than the present params: the depth
    // attachment has to match the colour one exactly, and the back buffer is
    // already at render resolution when `render.scale` is in play.
    let mut ds_params = CreateDepthTextureParams {
        device_handle: cq_params.device_handle,
        width: bb_params.width,
        height: bb_params.height,
        pixel_format: ds_pixel_format,
        pad0: 0,
        texture_handle: MetalHandle::NULL,
    };
    let status = unix_call(&mut ds_params);
    if status != 0 {
        error!(target: LOG_TARGET, "CreateDepthTexture failed (0x{status:08X})");
        destroy_partial_device(
            cq_params,
            layer_params.view_handle,
            bb_params.texture_handle,
        );
        return Err(D3DERR_INVALIDCALL);
    }
    info!(
        target: LOG_TARGET,
        "created depth/stencil texture (format={})",
        pp.auto_depth_stencil_format
    );
    Ok(ds_params.texture_handle)
}
