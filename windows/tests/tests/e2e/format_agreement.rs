//! Every create path agrees with the capability answer for the same format.
//!
//! A title that probes first and creates second has to get the same answer
//! twice: a format `CheckDeviceFormat`, `CheckDepthStencilMatch` or
//! `CheckDeviceMultiSampleType` refuses must not create, and one they accept
//! must. The sweeps walk every numbered `D3DFORMAT` and the FOURCC formats
//! titles probe, through every create entry point and pool the answer speaks
//! for, and report every disagreement at once. Under `make test INTEL=1` the
//! same sweeps run against the answers of a device without the packed 16-bit
//! formats or 32-bit float filtering.

use mtld3d_tests::{CubeTexture, Harness, HarnessConfig, PosColorVertex, Texture, VolumeTexture};
use mtld3d_types::{
    D3D_OK, D3DCLEAR_STENCIL, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_EQUAL, D3DCMP_LESS,
    D3DERR_INVALIDCALL, D3DFMT_A1R5G5B5, D3DFMT_A2R10G10B10, D3DFMT_A8R8G8B8, D3DFMT_ATI1,
    D3DFMT_ATOC, D3DFMT_D15S1, D3DFMT_D16, D3DFMT_D16_LOCKABLE, D3DFMT_D24FS8, D3DFMT_D24S8,
    D3DFMT_D24X4S4, D3DFMT_D24X8, D3DFMT_D32, D3DFMT_D32_LOCKABLE, D3DFMT_D32F_LOCKABLE,
    D3DFMT_DF16, D3DFMT_DF24, D3DFMT_DXT1, D3DFMT_DXT2, D3DFMT_DXT3, D3DFMT_DXT4, D3DFMT_DXT5,
    D3DFMT_INDEX16, D3DFMT_INDEX32, D3DFMT_INTZ, D3DFMT_NV12, D3DFMT_R5G6B5, D3DFMT_RESZ,
    D3DFMT_UYVY, D3DFMT_VERTEXDATA, D3DFMT_X1R5G5B5, D3DFMT_X8R8G8B8, D3DFMT_YUY2, D3DFMT_YV12,
    D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DMULTISAMPLE_2_SAMPLES, D3DMULTISAMPLE_4_SAMPLES,
    D3DMULTISAMPLE_8_SAMPLES, D3DMULTISAMPLE_NONE, D3DPOOL_DEFAULT, D3DPOOL_MANAGED,
    D3DPOOL_SYSTEMMEM, D3DPRESENT_PARAMETERS, D3DPT_TRIANGLELIST, D3DRS_LIGHTING,
    D3DRS_STENCILENABLE, D3DRS_STENCILFUNC, D3DRS_STENCILMASK, D3DRS_STENCILREF, D3DRS_ZENABLE,
    D3DRS_ZFUNC, D3DRS_ZWRITEENABLE, D3DRTYPE_CUBETEXTURE, D3DRTYPE_INDEXBUFFER, D3DRTYPE_SURFACE,
    D3DRTYPE_TEXTURE, D3DRTYPE_VERTEXBUFFER, D3DRTYPE_VOLUMETEXTURE, D3DSWAPEFFECT_DISCARD,
    D3DTEXF_NONE, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_DYNAMIC,
    D3DUSAGE_RENDERTARGET, D3DUSAGE_SOFTWAREPROCESSING,
};

/// The display format every query here is asked against.
const DISPLAY: u32 = D3DFMT_X8R8G8B8;

const BLACK: u32 = 0xFF00_0000;
const GREEN: u32 = 0xFF00_FF00;
const RED: u32 = 0xFFFF_0000;

/// The D3D9 back-buffer formats, the only ones a device swap chain is specified for.
const BACK_BUFFER_FORMATS: [u32; 6] = [
    D3DFMT_A2R10G10B10,
    D3DFMT_A8R8G8B8,
    D3DFMT_X8R8G8B8,
    D3DFMT_A1R5G5B5,
    D3DFMT_X1R5G5B5,
    D3DFMT_R5G6B5,
];

const fn fourcc(code: [u8; 4]) -> u32 {
    u32::from_le_bytes(code)
}

/// The FOURCC formats the sweeps walk besides the numbered ones.
///
/// The block-compressed and YUV formats, the vendor depth formats, and the
/// vendor codes titles probe for without the layer serving them (`ATI2`,
/// `RAWZ`, `NULL`, `INST`, `NVDB`, `A2M0`, `A2M1`, the packed RGBG pair and
/// `MET1`).
const FOURCCS: [u32; 24] = [
    D3DFMT_DXT1,
    D3DFMT_DXT2,
    D3DFMT_DXT3,
    D3DFMT_DXT4,
    D3DFMT_DXT5,
    D3DFMT_YUY2,
    D3DFMT_UYVY,
    D3DFMT_YV12,
    D3DFMT_NV12,
    D3DFMT_ATI1,
    fourcc(*b"ATI2"),
    D3DFMT_INTZ,
    D3DFMT_DF16,
    D3DFMT_DF24,
    fourcc(*b"RAWZ"),
    fourcc(*b"NULL"),
    fourcc(*b"INST"),
    fourcc(*b"NVDB"),
    fourcc(*b"A2M0"),
    fourcc(*b"A2M1"),
    fourcc(*b"RGBG"),
    fourcc(*b"GRGB"),
    fourcc(*b"MET1"),
    fourcc(*b"AL16"),
];

/// Every format the sweeps walk: each numbered `D3DFORMAT` and [`FOURCCS`].
///
/// `RESZ` and `ATOC` are left out: they are capability tokens a query
/// answers for a device feature, not formats any resource is created in.
fn candidate_formats() -> Vec<u32> {
    (1..=130)
        .chain(FOURCCS)
        .filter(|format| !matches!(*format, D3DFMT_RESZ | D3DFMT_ATOC))
        .collect()
}

/// A format as a failure message prints it: its number, or its four characters.
fn format_name(format: u32) -> String {
    if format < 0x100 {
        format.to_string()
    } else {
        String::from_utf8_lossy(&format.to_le_bytes()).into_owned()
    }
}

const fn succeeded(hr: i32) -> bool {
    hr >= 0
}

/// Whether a disagreement is one of the kept divergences, which the sweeps skip.
///
/// `ATI1` creates as a 2D texture, with any usage but a target one, and as
/// an offscreen plain while every query for it answers no: the layer keeps the
/// create for a title that makes one without probing, and does not advertise
/// a format whose lock pitch is the BC4 block pitch rather than the one D3D9
/// reports. `D16_LOCKABLE` and `D32F_LOCKABLE` create single-sampled as the
/// auto depth-stencil and as a depth-stencil texture while no depth query
/// offers them: their depth is served, the depth-surface `LockRect` they
/// promise is not. The planar YUV pair is advertised as a plain surface but
/// exists in `D3DPOOL_DEFAULT` only, which a query, having no pool, cannot
/// say. All three are in `docs/STATUS.md`.
fn kept_divergence(shape: &str, format: u32) -> bool {
    match format {
        D3DFMT_ATI1 => shape.starts_with("texture") || shape.starts_with("offscreen"),
        D3DFMT_D16_LOCKABLE | D3DFMT_D32F_LOCKABLE => {
            shape == "texture usage=DEPTHSTENCIL DEFAULT"
                || shape == "CreateDevice auto depth, 0 samples"
                || shape == "Reset auto depth, 0 samples"
        }
        D3DFMT_YV12 | D3DFMT_NV12 => shape == "offscreen SYSTEMMEM",
        _ => false,
    }
}

/// The disagreements one sweep found.
struct Sweep {
    mismatches: Vec<String>,
}

impl Sweep {
    const fn new() -> Self {
        Self {
            mismatches: Vec::new(),
        }
    }

    /// Record one create against the answer that speaks for it.
    fn compare(&mut self, shape: &str, format: u32, advertised: bool, create_hr: i32) {
        if advertised != succeeded(create_hr) && !kept_divergence(shape, format) {
            self.mismatches.push(format!(
                "{shape} format {}: query {}, create 0x{:08X}",
                format_name(format),
                if advertised { "yes" } else { "no" },
                create_hr,
            ));
        }
    }

    fn assert_agrees(self, what: &str) {
        assert!(
            self.mismatches.is_empty(),
            "{what}: {} disagreement(s):\n{}",
            self.mismatches.len(),
            self.mismatches.join("\n")
        );
    }
}

fn check(h: &Harness, usage: u32, rtype: u32, format: u32) -> bool {
    succeeded(h.check_device_format(DISPLAY, usage, rtype, format))
}

/// Whether the device answers `format` as a depth-stencil format.
///
/// D3D9 keeps depth formats in `D3DPOOL_DEFAULT` whatever the query says,
/// so the pool sweeps need to tell them apart.
fn is_depth(h: &Harness, format: u32) -> bool {
    check(h, D3DUSAGE_DEPTHSTENCIL, D3DRTYPE_TEXTURE, format)
        || check(h, D3DUSAGE_DEPTHSTENCIL, D3DRTYPE_SURFACE, format)
}

fn texture_hr(h: &Harness, usage: u32, levels: u32, format: u32, pool: u32) -> i32 {
    let (hr, out) = h.try_create_texture(4, 4, levels, usage, format, pool);
    if succeeded(hr) && !out.is_null() {
        drop(Texture::from_raw(out));
    }
    hr
}

fn cube_hr(h: &Harness, usage: u32, levels: u32, format: u32, pool: u32) -> i32 {
    let (hr, out) = h.try_create_cube_texture(4, levels, usage, format, pool);
    if succeeded(hr) && !out.is_null() {
        drop(CubeTexture::from_raw(out));
    }
    hr
}

fn volume_hr(h: &Harness, usage: u32, format: u32, pool: u32) -> i32 {
    let (hr, volume): (i32, Option<VolumeTexture<'_>>) =
        h.try_create_volume_texture([4, 4, 2], 1, usage, format, pool);
    drop(volume);
    hr
}

/// 2D textures in every pool and usage the texture query speaks for.
///
/// Usage 0 in the default, managed and system-memory pools, each against the
/// `D3DRTYPE_TEXTURE` answer (a depth format only in the default pool, the
/// one D3D9 allows it in), and render-target, depth-stencil, dynamic and
/// mip-generation usage in the default pool against the answer for that
/// usage. `D3DOK_NOAUTOGEN` is a success: the create still succeeds.
#[test]
fn texture_creates_agree_with_the_texture_query() {
    let h = Harness::new();
    let mut sweep = Sweep::new();
    for format in candidate_formats() {
        let plain = check(&h, 0, D3DRTYPE_TEXTURE, format);
        let depth = is_depth(&h, format);
        for (pool, name) in [
            (D3DPOOL_DEFAULT, "DEFAULT"),
            (D3DPOOL_MANAGED, "MANAGED"),
            (D3DPOOL_SYSTEMMEM, "SYSTEMMEM"),
        ] {
            let advertised = plain && (pool == D3DPOOL_DEFAULT || !depth);
            sweep.compare(
                &format!("texture usage=0 {name}"),
                format,
                advertised,
                texture_hr(&h, 0, 1, format, pool),
            );
        }
        for (usage, name) in [
            (D3DUSAGE_RENDERTARGET, "RENDERTARGET"),
            (D3DUSAGE_DEPTHSTENCIL, "DEPTHSTENCIL"),
            (D3DUSAGE_DYNAMIC, "DYNAMIC"),
            (D3DUSAGE_AUTOGENMIPMAP, "AUTOGENMIPMAP"),
        ] {
            let levels = u32::from(usage != D3DUSAGE_AUTOGENMIPMAP);
            sweep.compare(
                &format!("texture usage={name} DEFAULT"),
                format,
                check(&h, usage, D3DRTYPE_TEXTURE, format),
                texture_hr(&h, usage, levels, format, D3DPOOL_DEFAULT),
            );
        }
    }
    sweep.assert_agrees("CreateTexture against CheckDeviceFormat(D3DRTYPE_TEXTURE)");
}

/// Cube and volume textures in every pool and usage their queries speak for.
#[test]
fn cube_and_volume_creates_agree_with_their_queries() {
    let h = Harness::new();
    let mut sweep = Sweep::new();
    for format in candidate_formats() {
        let cube = check(&h, 0, D3DRTYPE_CUBETEXTURE, format);
        let volume = check(&h, 0, D3DRTYPE_VOLUMETEXTURE, format);
        for (pool, name) in [
            (D3DPOOL_DEFAULT, "DEFAULT"),
            (D3DPOOL_MANAGED, "MANAGED"),
            (D3DPOOL_SYSTEMMEM, "SYSTEMMEM"),
        ] {
            sweep.compare(
                &format!("cube usage=0 {name}"),
                format,
                cube,
                cube_hr(&h, 0, 1, format, pool),
            );
            sweep.compare(
                &format!("volume usage=0 {name}"),
                format,
                volume,
                volume_hr(&h, 0, format, pool),
            );
        }
        for (usage, name) in [
            (D3DUSAGE_RENDERTARGET, "RENDERTARGET"),
            (D3DUSAGE_DYNAMIC, "DYNAMIC"),
            (D3DUSAGE_AUTOGENMIPMAP, "AUTOGENMIPMAP"),
        ] {
            let levels = u32::from(usage != D3DUSAGE_AUTOGENMIPMAP);
            sweep.compare(
                &format!("cube usage={name} DEFAULT"),
                format,
                check(&h, usage, D3DRTYPE_CUBETEXTURE, format),
                cube_hr(&h, usage, levels, format, D3DPOOL_DEFAULT),
            );
        }
        sweep.compare(
            "volume usage=DYNAMIC DEFAULT",
            format,
            check(&h, D3DUSAGE_DYNAMIC, D3DRTYPE_VOLUMETEXTURE, format),
            volume_hr(&h, D3DUSAGE_DYNAMIC, format, D3DPOOL_DEFAULT),
        );
    }
    sweep.assert_agrees("CreateCubeTexture / CreateVolumeTexture against their queries");
}

/// Offscreen plains, render targets and depth-stencil surfaces against the surface query.
#[test]
fn surface_creates_agree_with_the_surface_query() {
    let h = Harness::new();
    let mut sweep = Sweep::new();
    for format in candidate_formats() {
        let plain = check(&h, 0, D3DRTYPE_SURFACE, format);
        for (pool, name) in [
            (D3DPOOL_DEFAULT, "DEFAULT"),
            (D3DPOOL_SYSTEMMEM, "SYSTEMMEM"),
        ] {
            let (hr, _) = h.create_offscreen_plain_surface_seeded(4, 4, format, pool);
            sweep.compare(&format!("offscreen {name}"), format, plain, hr);
        }
        let (hr, rt) = h.create_render_target_ms_hr((4, 4), format, (D3DMULTISAMPLE_NONE, 0), 0);
        drop(rt);
        sweep.compare(
            "CreateRenderTarget",
            format,
            check(&h, D3DUSAGE_RENDERTARGET, D3DRTYPE_SURFACE, format),
            hr,
        );
        let (hr, ds) =
            h.create_depth_stencil_surface_ms_hr((4, 4), format, (D3DMULTISAMPLE_NONE, 0));
        drop(ds);
        sweep.compare(
            "CreateDepthStencilSurface",
            format,
            check(&h, D3DUSAGE_DEPTHSTENCIL, D3DRTYPE_SURFACE, format),
            hr,
        );
    }
    sweep.assert_agrees("surface creates against CheckDeviceFormat(D3DRTYPE_SURFACE)");
}

/// Multisampled render targets and depth surfaces against the multisample query.
///
/// Every format the surface query accepts as a render target or a depth
/// stencil is created at two, four and eight samples, and the create has to
/// succeed exactly where `CheckDeviceMultiSampleType` does.
#[test]
fn multisampled_surface_creates_agree_with_the_multisample_query() {
    let h = Harness::new();
    let mut sweep = Sweep::new();
    for format in candidate_formats() {
        let rt = check(&h, D3DUSAGE_RENDERTARGET, D3DRTYPE_SURFACE, format);
        let ds = check(&h, D3DUSAGE_DEPTHSTENCIL, D3DRTYPE_SURFACE, format);
        if !rt && !ds {
            continue;
        }
        for samples in [
            D3DMULTISAMPLE_2_SAMPLES,
            D3DMULTISAMPLE_4_SAMPLES,
            D3DMULTISAMPLE_8_SAMPLES,
        ] {
            let advertised = succeeded(h.check_device_multi_sample_type(format, 1, samples).0);
            let (hr, surface) = if rt {
                h.create_render_target_ms_hr((4, 4), format, (samples, 0), 0)
            } else {
                h.create_depth_stencil_surface_ms_hr((4, 4), format, (samples, 0))
            };
            drop(surface);
            sweep.compare(&format!("{samples} samples"), format, advertised, hr);
        }
    }
    sweep.assert_agrees("multisampled surface creates against CheckDeviceMultiSampleType");
}

/// Every advertised multisampled render-target format renders and resolves.
///
/// A multisampled colour target is a multisampled companion resolved into a
/// single-sample texture, so each format the multisample query accepts has
/// to take a clear and a resolving `StretchRect` into a single-sample target
/// of its own format. The Metal validation layer the suite runs under ends
/// the process on a resolve the format does not support.
#[test]
fn every_advertised_multisampled_render_target_resolves() {
    let h = Harness::new();
    let back_buffer = h.render_target(0);
    for format in candidate_formats() {
        if !check(&h, D3DUSAGE_RENDERTARGET, D3DRTYPE_SURFACE, format) {
            continue;
        }
        for samples in [
            D3DMULTISAMPLE_2_SAMPLES,
            D3DMULTISAMPLE_4_SAMPLES,
            D3DMULTISAMPLE_8_SAMPLES,
        ] {
            if !succeeded(h.check_device_multi_sample_type(format, 1, samples).0) {
                continue;
            }
            let name = format_name(format);
            let multisampled = h.create_render_target_ms((16, 16), format, (samples, 0));
            let single = h.create_render_target_ms((16, 16), format, (D3DMULTISAMPLE_NONE, 0));
            assert_eq!(
                h.set_render_target(0, &multisampled),
                D3D_OK,
                "{name} x{samples}"
            );
            assert_eq!(h.begin_scene(), D3D_OK, "{name} x{samples}");
            assert_eq!(h.clear_target(GREEN), D3D_OK, "{name} x{samples} clear");
            assert_eq!(h.end_scene(), D3D_OK, "{name} x{samples}");
            assert_eq!(
                h.stretch_rect(&multisampled, &single, D3DTEXF_NONE),
                D3D_OK,
                "{name} x{samples} resolving StretchRect"
            );
            assert_eq!(
                h.set_render_target(0, &back_buffer),
                D3D_OK,
                "{name} x{samples}"
            );
            assert_eq!(h.present(), D3D_OK, "{name} x{samples} present");
        }
    }
}

/// Present parameters for a windowed 64x64 device with an auto depth-stencil of `format`.
const fn auto_depth_params(format: u32, samples: u32) -> D3DPRESENT_PARAMETERS {
    D3DPRESENT_PARAMETERS {
        back_buffer_width: 64,
        back_buffer_height: 64,
        back_buffer_format: D3DFMT_X8R8G8B8,
        back_buffer_count: 1,
        multi_sample_type: samples,
        multi_sample_quality: 0,
        swap_effect: D3DSWAPEFFECT_DISCARD,
        device_window: 0,
        windowed: 1,
        enable_auto_depth_stencil: 1,
        auto_depth_stencil_format: format,
        flags: 0,
        full_screen_refresh_rate_in_hz: 0,
        presentation_interval: 0,
    }
}

/// The formats the auto depth-stencil sweeps walk.
///
/// Every numbered depth format, the vendor depth formats, a colour format
/// and five codes no D3D9 format uses: one device per format, so the list
/// stays short. The unknown codes are deliberate, the gaps D3D9 leaves
/// between its depth formats and one far outside any range.
const AUTO_DEPTH_FORMATS: [u32; 19] = [
    D3DFMT_D16_LOCKABLE,
    D3DFMT_D32,
    D3DFMT_D15S1,
    D3DFMT_D24S8,
    D3DFMT_D24X8,
    D3DFMT_D24X4S4,
    D3DFMT_D16,
    D3DFMT_D32F_LOCKABLE,
    D3DFMT_D24FS8,
    D3DFMT_D32_LOCKABLE,
    D3DFMT_INTZ,
    D3DFMT_DF16,
    D3DFMT_DF24,
    D3DFMT_A8R8G8B8,
    72,
    74,
    76,
    78,
    0x00FF_00FF,
];

/// Whether the depth answers accept `format` as the auto depth-stencil of an X8R8G8B8 back buffer.
fn auto_depth_advertised(h: &Harness, format: u32, samples: u32) -> bool {
    check(h, D3DUSAGE_DEPTHSTENCIL, D3DRTYPE_SURFACE, format)
        && succeeded(h.check_depth_stencil_match(DISPLAY, D3DFMT_X8R8G8B8, format))
        && succeeded(h.check_device_multi_sample_type(format, 1, samples).0)
}

/// `CreateDevice` takes an auto depth-stencil format exactly where the depth answers do.
///
/// Single-sampled for every candidate, and at four samples for every one the
/// device multisamples an X8R8G8B8 back buffer at, so a depth format the
/// multisample query refuses (`INTZ` and the lockable ones) is refused there
/// too, and one it accepts (`DF16` and `DF24` among them) is created.
#[test]
fn create_device_auto_depth_agrees_with_the_depth_answers() {
    let h = Harness::factory_only();
    let multisample = succeeded(
        h.check_device_multi_sample_type(D3DFMT_X8R8G8B8, 1, D3DMULTISAMPLE_4_SAMPLES)
            .0,
    );
    let mut sweep = Sweep::new();
    for format in AUTO_DEPTH_FORMATS {
        for samples in [D3DMULTISAMPLE_NONE, D3DMULTISAMPLE_4_SAMPLES] {
            if samples != D3DMULTISAMPLE_NONE && !multisample {
                continue;
            }
            let mut pp = auto_depth_params(format, samples);
            sweep.compare(
                &format!("CreateDevice auto depth, {samples} samples"),
                format,
                auto_depth_advertised(&h, format, samples),
                h.create_device_hr(&mut pp),
            );
        }
    }
    sweep.assert_agrees("CreateDevice auto depth-stencil against the depth answers");
}

/// `Reset` takes an auto depth-stencil format exactly where the depth answers do.
///
/// A refused `Reset` leaves the device to be reset again, which the next
/// one with a plain depth format does, so one device walks every candidate.
#[test]
fn reset_auto_depth_agrees_with_the_depth_answers() {
    let h = Harness::new();
    let multisample = succeeded(
        h.check_device_multi_sample_type(D3DFMT_X8R8G8B8, 1, D3DMULTISAMPLE_4_SAMPLES)
            .0,
    );
    let mut sweep = Sweep::new();
    for format in AUTO_DEPTH_FORMATS {
        for samples in [D3DMULTISAMPLE_NONE, D3DMULTISAMPLE_4_SAMPLES] {
            if samples != D3DMULTISAMPLE_NONE && !multisample {
                continue;
            }
            let mut pp = auto_depth_params(format, samples);
            pp.device_window = h.hwnd();
            sweep.compare(
                &format!("Reset auto depth, {samples} samples"),
                format,
                auto_depth_advertised(&h, format, samples),
                h.reset_params(&mut pp),
            );
            let mut plain = auto_depth_params(D3DFMT_D24S8, D3DMULTISAMPLE_NONE);
            plain.device_window = h.hwnd();
            assert_eq!(
                h.reset_params(&mut plain),
                D3D_OK,
                "a plain Reset after format {} recovers the device",
                format_name(format)
            );
        }
    }
    sweep.assert_agrees("Reset auto depth-stencil against the depth answers");
}

/// The two depth formats no device serves are refused by every depth answer and create.
///
/// `D15S1` and `D24X4S4` carry a stencil narrower than any Metal format's.
#[test]
fn unserved_depth_formats_are_refused_everywhere() {
    let h = Harness::new();
    for format in [D3DFMT_D15S1, D3DFMT_D24X4S4] {
        let name = format_name(format);
        assert!(!is_depth(&h, format), "{name} is advertised");
        assert!(
            !succeeded(h.check_depth_stencil_match(DISPLAY, D3DFMT_X8R8G8B8, format)),
            "{name} matches a render target"
        );
        assert!(
            !succeeded(texture_hr(
                &h,
                D3DUSAGE_DEPTHSTENCIL,
                1,
                format,
                D3DPOOL_DEFAULT
            )),
            "{name} depth texture creates"
        );
        let mut pp = auto_depth_params(format, D3DMULTISAMPLE_NONE);
        pp.device_window = h.hwnd();
        assert_eq!(
            h.reset_params(&mut pp),
            D3DERR_INVALIDCALL,
            "{name} auto depth Reset"
        );
        let mut plain = auto_depth_params(D3DFMT_D24S8, D3DMULTISAMPLE_NONE);
        plain.device_window = h.hwnd();
        assert_eq!(h.reset_params(&mut plain), D3D_OK, "recovering Reset");
    }
}

/// The lockable depth formats serve depth where they create, and refuse the lock.
///
/// No depth query offers `D16_LOCKABLE` or `D32F_LOCKABLE`, and a standalone
/// depth-stencil surface refuses them, but an auto depth-stencil and a
/// depth-stencil texture take either one, for a title that names one without
/// probing. The auto depth-stencil depth-tests a farther quad away behind a
/// nearer one, and a `LockRect` of the depth surface, the one thing the
/// format promises beyond depth, fails cleanly on both surfaces.
#[test]
fn lockable_depth_formats_serve_depth_without_the_lock() {
    for format in [D3DFMT_D16_LOCKABLE, D3DFMT_D32F_LOCKABLE] {
        let name = format_name(format);
        let h = Harness::create(&HarnessConfig {
            depth_format: Some(format),
            ..HarnessConfig::default()
        });
        assert!(!is_depth(&h, format), "{name} is advertised");
        assert!(
            !succeeded(h.check_depth_stencil_match(DISPLAY, D3DFMT_X8R8G8B8, format)),
            "{name} matches a render target"
        );
        let (hr, standalone) =
            h.create_depth_stencil_surface_ms_hr((64, 64), format, (D3DMULTISAMPLE_NONE, 0));
        assert_eq!(hr, D3DERR_INVALIDCALL, "{name} standalone depth surface");
        assert!(standalone.is_none());

        assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
        assert_eq!(h.clear_texture(0), D3D_OK);
        h.select_diffuse_stage(0);
        assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESS), D3D_OK);
        h.render_once(BLACK, |d| {
            assert_eq!(
                d.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
                D3D_OK
            );
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(GREEN, 0.25)),
                D3D_OK
            );
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(RED, 0.75)),
                D3D_OK
            );
        });
        assert_eq!(
            h.read_pixel(320, 240),
            GREEN,
            "{name}: the auto depth-stencil keeps the nearer quad"
        );

        let depth = h
            .depth_stencil_surface()
            .expect("the auto depth-stencil surface");
        let (hr, _) = depth.lock_rect_probe(0);
        assert_eq!(hr, D3DERR_INVALIDCALL, "{name}: auto depth LockRect");
        drop(depth);

        let (hr, out) =
            h.try_create_texture(64, 64, 1, D3DUSAGE_DEPTHSTENCIL, format, D3DPOOL_DEFAULT);
        assert_eq!(hr, D3D_OK, "{name} depth-stencil texture");
        let texture = Texture::from_raw(out);
        let (hr, _) = texture.lock_rect_probe(0, 0);
        assert_eq!(hr, D3DERR_INVALIDCALL, "{name}: depth texture LockRect");
        let (hr, _) = texture.surface_level(0).lock_rect_probe(0);
        assert_eq!(
            hr, D3DERR_INVALIDCALL,
            "{name}: depth texture level LockRect"
        );
        drop(texture);
        let (hr, _) = h.try_create_texture(64, 64, 1, 0, format, D3DPOOL_DEFAULT);
        assert_eq!(hr, D3DERR_INVALIDCALL, "{name} plain depth texture");
    }
}

/// A 4x swap chain takes DF16 and DF24 as its auto depth-stencil and depth-tests through it.
///
/// The multisample query accepts both formats, so the auto depth-stencil of
/// a multisampled swap chain may be either. As an attachment it is never
/// sampled, so it is a plain multisampled depth: a nearer quad stays in
/// front of a farther one drawn after it.
#[test]
fn a_multisampled_swap_chain_depth_tests_through_df16_and_df24() {
    let probe = Harness::factory_only();
    if !succeeded(
        probe
            .check_device_multi_sample_type(D3DFMT_X8R8G8B8, 1, D3DMULTISAMPLE_4_SAMPLES)
            .0,
    ) {
        return;
    }
    for format in [D3DFMT_DF16, D3DFMT_DF24] {
        assert_eq!(
            probe
                .check_device_multi_sample_type(format, 1, D3DMULTISAMPLE_4_SAMPLES)
                .0,
            D3D_OK,
            "{} at four samples",
            format_name(format)
        );
    }
    drop(probe);
    for format in [D3DFMT_DF16, D3DFMT_DF24] {
        let name = format_name(format);
        let h = Harness::create(&HarnessConfig {
            depth_format: Some(format),
            multi_sample_type: D3DMULTISAMPLE_4_SAMPLES,
            ..HarnessConfig::default()
        });
        assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
        assert_eq!(h.clear_texture(0), D3D_OK);
        h.select_diffuse_stage(0);
        assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESS), D3D_OK);
        h.render_once(BLACK, |d| {
            assert_eq!(
                d.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
                D3D_OK
            );
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(GREEN, 0.25)),
                D3D_OK
            );
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(RED, 0.75)),
                D3D_OK
            );
        });
        assert_eq!(
            h.read_pixel(320, 240),
            GREEN,
            "{name}: the multisampled auto depth-stencil keeps the nearer quad"
        );
    }
}

/// A full clip-space quad at depth `z`.
fn quad(color: u32, z: f32) -> [PosColorVertex; 6] {
    let v = |x: f32, y: f32| PosColorVertex { x, y, z, color };
    [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ]
}

/// D24FS8 is advertised and serves a depth test and an eight-bit stencil.
///
/// The stencil is cleared to a value with its high bit set, so a reference
/// that differs from it in that bit alone must fail where the stored value
/// passes: a stencil plane narrower than eight bits could not tell the two
/// apart. The depth plane gates a nearer quad in and a farther one out.
#[test]
fn d24fs8_serves_depth_and_an_eight_bit_stencil() {
    const STORED: u32 = 0xC3;
    let probe = Harness::factory_only();
    assert!(
        check(
            &probe,
            D3DUSAGE_DEPTHSTENCIL,
            D3DRTYPE_SURFACE,
            D3DFMT_D24FS8
        ),
        "D24FS8 depth-stencil surface query"
    );
    assert!(
        check(
            &probe,
            D3DUSAGE_DEPTHSTENCIL,
            D3DRTYPE_TEXTURE,
            D3DFMT_D24FS8
        ),
        "D24FS8 depth-stencil texture query"
    );
    assert_eq!(
        probe.check_depth_stencil_match(DISPLAY, D3DFMT_X8R8G8B8, D3DFMT_D24FS8),
        D3D_OK,
        "D24FS8 depth-stencil match"
    );
    drop(probe);

    let h = Harness::create(&HarnessConfig {
        depth_format: Some(D3DFMT_D24FS8),
        ..HarnessConfig::default()
    });
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    assert_eq!(h.clear_texture(0), D3D_OK);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK);
    let clear = D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL;

    // Stencil: only the reference equal to the stored value in all eight bits paints.
    h.render_once(BLACK, |d| {
        assert_eq!(d.clear(clear, BLACK, 1.0, STORED), D3D_OK, "clear");
        assert_eq!(d.set_render_state(D3DRS_STENCILENABLE, 1), D3D_OK);
        assert_eq!(d.set_render_state(D3DRS_STENCILMASK, 0xFF), D3D_OK);
        assert_eq!(d.set_render_state(D3DRS_STENCILFUNC, D3DCMP_EQUAL), D3D_OK);
        assert_eq!(d.set_render_state(D3DRS_STENCILREF, STORED & 0x7F), D3D_OK);
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(RED, 0.5)),
            D3D_OK
        );
        assert_eq!(d.set_render_state(D3DRS_STENCILREF, STORED), D3D_OK);
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(GREEN, 0.5)),
            D3D_OK
        );
    });
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "the stencil test compares all eight bits"
    );

    // Depth: a quad nearer than the cleared depth passes, a farther one does not.
    assert_eq!(h.set_render_state(D3DRS_STENCILENABLE, 0), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 0), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESS), D3D_OK);
    h.render_once(BLACK, |d| {
        assert_eq!(d.clear(clear, BLACK, 0.5, 0), D3D_OK, "clear");
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(GREEN, 0.25)),
            D3D_OK
        );
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(RED, 0.75)),
            D3D_OK
        );
    });
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "the depth test keeps the nearer quad"
    );

    // A standalone surface and a depth texture of the format create too.
    drop(h.create_depth_stencil_surface(64, 64, D3DFMT_D24FS8));
    let (hr, out) = h.try_create_texture(
        64,
        64,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_D24FS8,
        D3DPOOL_DEFAULT,
    );
    assert_eq!(hr, D3D_OK, "D24FS8 depth texture");
    drop(Texture::from_raw(out));
}

/// The windowed and fullscreen device-type answers offer D3D9 back-buffer formats only.
///
/// A swap chain is specified for six formats, and a title that probes for its
/// back buffer has to land on one of them: a float or wide format the device
/// renders into is still no back buffer. X8R8G8B8 and A8R8G8B8 stay
/// advertised in both modes.
#[test]
fn check_device_type_offers_only_back_buffer_formats() {
    let h = Harness::factory_only();
    for format in candidate_formats() {
        for windowed in [true, false] {
            let hr = h.check_device_type(DISPLAY, format, windowed);
            assert!(
                !succeeded(hr) || BACK_BUFFER_FORMATS.contains(&format),
                "CheckDeviceType(windowed={windowed}) offers {} as a back buffer",
                format_name(format)
            );
        }
    }
    for windowed in [true, false] {
        for format in [D3DFMT_X8R8G8B8, D3DFMT_A8R8G8B8] {
            assert_eq!(
                h.check_device_type(DISPLAY, format, windowed),
                D3D_OK,
                "CheckDeviceType(windowed={windowed}, {})",
                format_name(format)
            );
        }
    }
}

/// A buffer resource type is an invalid format question, not an unavailable one.
///
/// Vertex and index buffers carry no format a query could weigh, so D3D9
/// rejects the call itself, with or without a usage.
#[test]
fn a_buffer_resource_type_is_an_invalid_format_query() {
    let h = Harness::factory_only();
    for (usage, rtype, format) in [
        (0, D3DRTYPE_VERTEXBUFFER, D3DFMT_VERTEXDATA),
        (0, D3DRTYPE_INDEXBUFFER, D3DFMT_VERTEXDATA),
        (0, D3DRTYPE_INDEXBUFFER, D3DFMT_INDEX16),
        (0, D3DRTYPE_INDEXBUFFER, D3DFMT_INDEX32),
        (
            D3DUSAGE_SOFTWAREPROCESSING,
            D3DRTYPE_VERTEXBUFFER,
            D3DFMT_VERTEXDATA,
        ),
        (D3DUSAGE_DYNAMIC, D3DRTYPE_INDEXBUFFER, D3DFMT_INDEX16),
    ] {
        assert_eq!(
            h.check_device_format(DISPLAY, usage, rtype, format),
            D3DERR_INVALIDCALL,
            "CheckDeviceFormat(usage={usage:#x}, rtype={rtype}, format={format})"
        );
    }
}
