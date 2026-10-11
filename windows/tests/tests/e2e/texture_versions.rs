//! Textures rewritten between the draws that sample them keep every draw's texels.
//!
//! An upload into a texture a draw earlier in the frame sampled goes to
//! another version of the texture, so that draw keeps what it sampled. The
//! version it replaces becomes a spare once the GPU has finished with it, and
//! a later rewrite of the same texture takes that spare and copies into it
//! what the texture changed since it retired: for a texture written only by
//! uploads, the box of the tiles written after that point. Each test patches
//! a texture between draws several times a frame, over several frames with a
//! read-back between them so one frame's versions are spares by the next,
//! and checks that every draw sampled the texture exactly as the patches up
//! to it left it. The patches are partial `UpdateSurface` copies from system
//! memory, partial `LockRect`s of a managed texture, or a whole-level lock
//! that changes one rectangle and publishes it with `AddDirtyRect`.

use mtld3d_tests::{
    CubeTexture, Harness, Surface, Texture, TexturedVertex, VolumeVertex, assert_pixel_approx,
};
use mtld3d_types::{
    D3DFMT_A4R4G4B4, D3DFMT_A8R8G8B8, D3DFMT_DXT1, D3DFVF_DIFFUSE, D3DFVF_TEX1,
    D3DFVF_TEXTUREFORMAT3, D3DFVF_XYZ, D3DLOCK_NO_DIRTY_UPDATE, D3DLOCK_READONLY, D3DPOOL_DEFAULT,
    D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DRECT, D3DSAMP_ADDRESSU,
    D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER,
    D3DSAMP_SRGBTEXTURE, D3DTADDRESS_CLAMP, D3DTEXF_POINT, D3DUSAGE_AUTOGENMIPMAP,
};

const BLACK: u32 = 0xFF00_0000;
const RED: u32 = 0xFFFF_0000;
const GREEN: u32 = 0xFF00_FF00;
const BLUE: u32 = 0xFF00_00FF;
const YELLOW: u32 = 0xFFFF_FF00;
const CYAN: u32 = 0xFF00_FFFF;
const MAGENTA: u32 = 0xFFFF_00FF;
const WHITE: u32 = 0xFFFF_FFFF;
const GRAY: u32 = 0xFF80_8080;

/// The harness back buffer's extent.
const BACK: (u32, u32) = (640, 480);

/// Tolerance of a sampled colour.
///
/// Point sampling returns the texel, and the scaled leg's read-back
/// resamples well inside a uniform block.
const TOLERANCE: u8 = 2;

/// A rectangle of the back buffer that one draw samples a texture into.
struct Cell {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl Cell {
    /// Cell `index` of a `columns` by `rows` grid over the back buffer, row by row.
    const fn of_grid(index: u32, columns: u32, rows: u32) -> Self {
        let width = BACK.0 / columns;
        let height = BACK.1 / rows;
        Self {
            x: (index % columns) * width,
            y: (index / columns) * height,
            width,
            height,
        }
    }

    /// The cell's corners in clip space: left, top, right, bottom.
    fn clip(&self) -> [f32; 4] {
        let float = |v: u32| f32::from(u16::try_from(v).expect("back-buffer coordinate"));
        let half = (float(BACK.0) / 2.0, float(BACK.1) / 2.0);
        [
            float(self.x) / half.0 - 1.0,
            1.0 - float(self.y) / half.1,
            float(self.x + self.width) / half.0 - 1.0,
            1.0 - float(self.y + self.height) / half.1,
        ]
    }

    /// Two clockwise triangles covering the cell, sampling texture coordinates `(0, 0)..span`.
    fn quad(&self, span: (f32, f32)) -> [TexturedVertex; 6] {
        let [left, top, right, bottom] = self.clip();
        let vertex = |x, y, u, v| TexturedVertex {
            x,
            y,
            z: 0.5,
            color: WHITE,
            u,
            v,
        };
        [
            vertex(left, top, 0.0, 0.0),
            vertex(right, top, span.0, 0.0),
            vertex(left, bottom, 0.0, span.1),
            vertex(right, top, span.0, 0.0),
            vertex(right, bottom, span.0, span.1),
            vertex(left, bottom, 0.0, span.1),
        ]
    }

    /// Two clockwise triangles covering the cell with the directions of a cube's -Y face.
    ///
    /// The face's two other coordinates run from -1 to 1 across the cell. A
    /// test reading the face keeps its pattern symmetric under every flip and
    /// transpose of the face, so it does not depend on the face's orientation.
    fn negative_y_face_quad(&self) -> [VolumeVertex; 6] {
        let [left, top, right, bottom] = self.clip();
        let vertex = |x, y, u, w| VolumeVertex {
            x,
            y,
            z: 0.5,
            color: WHITE,
            u,
            v: -1.0,
            w,
        };
        [
            vertex(left, top, -1.0, 1.0),
            vertex(right, top, 1.0, 1.0),
            vertex(left, bottom, -1.0, -1.0),
            vertex(right, top, 1.0, 1.0),
            vertex(right, bottom, 1.0, -1.0),
            vertex(left, bottom, -1.0, -1.0),
        ]
    }

    /// The pixel of the cell that samples texel `texel` of the `view` texels the quad spans.
    const fn pixel(&self, texel: (u32, u32), view: (u32, u32)) -> (u32, u32) {
        (
            self.x + (2 * texel.0 + 1) * self.width / (2 * view.0),
            self.y + (2 * texel.1 + 1) * self.height / (2 * view.1),
        )
    }
}

/// The back buffer as `0xAARRGGBB` pixels, read once after a frame.
struct Readback {
    pixels: Vec<u32>,
    stride: usize,
}

impl Readback {
    /// `GetRenderTargetData` into a system-memory copy, which waits for the frame.
    fn of(h: &Harness) -> Self {
        let target = h.render_target(0);
        let (hr, desc) = target.desc();
        assert_eq!(hr, 0, "GetDesc of the back buffer");
        let copy = h.create_offscreen_plain_surface(
            desc.width,
            desc.height,
            D3DFMT_A8R8G8B8,
            D3DPOOL_SYSTEMMEM,
        );
        assert_eq!(
            h.get_render_target_data_hr(&target, &copy),
            0,
            "GetRenderTargetData"
        );
        let locked = copy.lock_rect(D3DLOCK_READONLY);
        let stride = usize::try_from(locked.pitch()).expect("positive pitch") / 4;
        let rows = usize::try_from(desc.height).expect("height fits usize");
        Self {
            pixels: locked.as_u32(stride * rows).to_vec(),
            stride,
        }
    }

    fn at(&self, (x, y): (u32, u32)) -> u32 {
        self.pixels[y as usize * self.stride + x as usize]
    }
}

/// What a run of rectangle fills over a base colour leaves at each texel of one level.
struct Painted {
    base: u32,
    fills: Vec<([u32; 4], u32)>,
}

impl Painted {
    const fn new(base: u32) -> Self {
        Self {
            base,
            fills: Vec::new(),
        }
    }

    /// Fill `[x0, y0, x1, y1)` with `color`.
    fn fill(&mut self, rect: [u32; 4], color: u32) {
        self.fills.push((rect, color));
    }

    fn at(&self, (x, y): (u32, u32)) -> u32 {
        self.fills
            .iter()
            .rev()
            .find(|([x0, y0, x1, y1], _)| (*x0..*x1).contains(&x) && (*y0..*y1).contains(&y))
            .map_or(self.base, |&(_, color)| color)
    }
}

/// A texel format a test writes, with how one colour is encoded in it.
enum Format {
    /// `D3DFMT_A4R4G4B4`; colours whose channels are 0x00, 0x88 or 0xFF.
    Argb4,
    /// `D3DFMT_A8R8G8B8`.
    Argb8,
    /// `D3DFMT_DXT1`, one solid block; colours whose channels are 0x00 or 0xFF.
    Dxt1,
}

impl Format {
    const fn d3d(&self) -> u32 {
        match self {
            Self::Argb4 => D3DFMT_A4R4G4B4,
            Self::Argb8 => D3DFMT_A8R8G8B8,
            Self::Dxt1 => D3DFMT_DXT1,
        }
    }

    /// Texels one encoded unit covers along each axis.
    const fn block(&self) -> u32 {
        match self {
            Self::Argb4 | Self::Argb8 => 1,
            Self::Dxt1 => 4,
        }
    }

    /// `rect` filled with `color`: its row length in bytes, its rows, and the rows' bytes.
    fn fill(&self, rect: [u32; 4], color: u32) -> (usize, usize, Vec<u8>) {
        let [x0, y0, x1, y1] = rect;
        let unit = self.encode(color);
        let columns = (x1 - x0).div_ceil(self.block()) as usize;
        let rows = (y1 - y0).div_ceil(self.block()) as usize;
        let bytes = unit
            .iter()
            .copied()
            .cycle()
            .take(unit.len() * columns * rows)
            .collect();
        (unit.len() * columns, rows, bytes)
    }

    /// One texel, or one block, of `color` (`0xAARRGGBB`).
    fn encode(&self, color: u32) -> Vec<u8> {
        let [b, g, r, a] = color.to_le_bytes();
        match self {
            Self::Argb4 => {
                let nibble = |c: u8| u16::from(c >> 4);
                let texel = (nibble(a) << 12) | (nibble(r) << 8) | (nibble(g) << 4) | nibble(b);
                texel.to_le_bytes().to_vec()
            }
            Self::Argb8 => color.to_le_bytes().to_vec(),
            Self::Dxt1 => {
                let c565 = (u16::from(r >> 3) << 11) | (u16::from(g >> 2) << 5) | u16::from(b >> 3);
                let [lo, hi] = c565.to_le_bytes();
                // Both endpoints equal and every index 0: the whole block is
                // the first endpoint.
                vec![lo, hi, lo, hi, 0, 0, 0, 0]
            }
        }
    }
}

/// A system-memory surface the size of a destination level, copied over one rectangle at a time.
struct Patcher<'h> {
    source: Surface<'h>,
    format: Format,
}

impl<'h> Patcher<'h> {
    fn new(h: &'h Harness, size: (u32, u32), format: Format) -> Self {
        Self {
            source: h.create_offscreen_plain_surface(
                size.0,
                size.1,
                format.d3d(),
                D3DPOOL_SYSTEMMEM,
            ),
            format,
        }
    }

    /// Fill `rect` of the source with `color` and `UpdateSurface` it to the same place in `dst`.
    fn patch(&self, h: &Harness, dst: &Surface<'_>, rect: [u32; 4], color: u32) {
        let (row_bytes, rows, bytes) = self.format.fill(rect, color);
        let signed = rect.map(|v| i32::try_from(v).expect("texel coordinate fits i32"));
        self.source
            .lock_rect_partial(&signed, 0)
            .write_u8_rect(row_bytes, rows, &bytes);
        let src_rect = D3DRECT {
            x1: signed[0],
            y1: signed[1],
            x2: signed[2],
            y2: signed[3],
        };
        assert_eq!(
            h.update_surface_region_hr(&self.source, &src_rect, dst, (signed[0], signed[1])),
            0,
            "UpdateSurface of {rect:?}",
        );
    }
}

/// Point-sampled, clamped stage 0, sampling the texture's colour alone.
fn point_sampling(h: &Harness, fvf: u32) {
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_MIPFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "SetSamplerState");
    }
    assert_eq!(h.set_fvf(fvf), 0, "SetFVF");
}

const FVF_2D: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1;
/// One three-component coordinate: `D3DFVF_TEXCOORDSIZE3(0)`.
const FVF_CUBE: u32 = FVF_2D | D3DFVF_TEXTUREFORMAT3 << 16;

/// What one draw of a frame must have sampled: its cell, and the colour at each probed pixel.
struct Expected {
    cell: u32,
    probes: Vec<((u32, u32), u32)>,
}

/// Compare a frame's read-back with what each of its draws had to sample.
fn check_frame(h: &Harness, expected: &[Expected], context: &str) {
    let readback = Readback::of(h);
    for draw in expected {
        for &(pixel, color) in &draw.probes {
            assert_pixel_approx(
                readback.at(pixel),
                color,
                TOLERANCE,
                &format!("{context}: draw {} at pixel {pixel:?}", draw.cell),
            );
        }
    }
}

/// Run `frames` frames of one patch-then-draw round per cell of `grid`, checking every draw.
///
/// `round(cell)` patches the texture and returns the probes its draw must
/// show and the draw itself, which follows inside the scene. Every round but
/// each frame's first renames the texture, since the frame sampled it
/// already. The read-back after each frame waits for the GPU, so the next
/// frame's renames find the previous frame's versions as spares.
fn run_rounds(
    h: &Harness,
    frames: u32,
    grid: (u32, u32),
    mut round: impl FnMut(&Cell) -> (Vec<((u32, u32), u32)>, Draw),
    context: &str,
) {
    for frame in 0..frames {
        let mut expected = Vec::new();
        h.render_once(BLACK, |d| {
            for index in 0..grid.0 * grid.1 {
                let cell = Cell::of_grid(index, grid.0, grid.1);
                let (probes, draw) = round(&cell);
                draw.issue(d);
                expected.push(Expected {
                    cell: index,
                    probes,
                });
            }
        });
        check_frame(h, &expected, &format!("{context}, frame {frame}"));
    }
}

/// The draw a round issues after its patch.
enum Draw {
    Flat([TexturedVertex; 6]),
    Cube([VolumeVertex; 6]),
}

impl Draw {
    fn issue(&self, h: &Harness) {
        let hr = match self {
            Self::Flat(quad) => h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, quad),
            Self::Cube(quad) => h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, quad),
        };
        assert_eq!(hr, 0, "sampling draw");
    }
}

/// Pin the sampled level to `level` for a magnifying draw.
fn sample_level(h: &Harness, level: u32) {
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
}

/// Sizes of the sprites the scratch tests write at `(0, 0)`, in whole 32-texel blocks.
const SPRITES: [(u32, u32); 12] = [
    (32, 32),
    (96, 64),
    (256, 256),
    (64, 160),
    (160, 96),
    (128, 128),
    (224, 32),
    (32, 224),
    (192, 192),
    (64, 64),
    (256, 96),
    (96, 256),
];

/// Centres of 32-texel blocks in the scratch's top-left 256 texels, where the sprites land.
const SPRITE_PROBES: [(u32, u32); 10] = [
    (16, 16),
    (80, 16),
    (16, 80),
    (112, 112),
    (176, 48),
    (48, 176),
    (240, 240),
    (208, 144),
    (144, 208),
    (240, 16),
];

/// How the sprite-scratch scene writes a sprite.
enum Writer {
    /// A default-pool texture, each sprite an `UpdateSurface` from system memory.
    UpdateSurface,
    /// A managed texture, each sprite a `LockRect` of its rectangle alone, as the game does.
    PartialLock,
}

/// The sprite-scratch scene: sprites of every size at `(0, 0)` of a 512-texel texture.
///
/// `palette[0]` is the fill the texture starts with. Each cell shows the
/// texture's top-left 256 texels, and every probe sits in the middle of a
/// 32-texel block, which a sprite covers whole or not at all. With
/// `evict_before`, `EvictManagedResources` runs once, ahead of that sprite.
fn sprite_scratch(
    h: &Harness,
    (format, writer): (Format, Writer),
    palette: &[u32],
    (frames, grid): (u32, (u32, u32)),
    evict_before: Option<usize>,
    context: &str,
) {
    const SIZE: u32 = 512;
    const VIEW: u32 = 256;
    let base = palette[0];
    let pool = match writer {
        Writer::UpdateSurface => D3DPOOL_DEFAULT,
        Writer::PartialLock => D3DPOOL_MANAGED,
    };
    let patcher = Patcher::new(h, (SIZE, SIZE), format);
    let tex = h.create_texture(SIZE, SIZE, 1, 0, patcher.format.d3d(), pool);
    let level = tex.surface_level(0);
    let write = |rect: [u32; 4], color: u32| match writer {
        Writer::UpdateSurface => patcher.patch(h, &level, rect, color),
        Writer::PartialLock => {
            let (row_bytes, rows, bytes) = patcher.format.fill(rect, color);
            let signed = rect.map(|v| i32::try_from(v).expect("texel coordinate fits i32"));
            let mut lock = tex.lock_rect_partial(0, &signed, 0);
            lock.write_u8_rect(row_bytes, rows, &bytes);
            assert_eq!(lock.unlock(), 0, "UnlockRect of {rect:?}");
        }
    };
    write([0, 0, SIZE, SIZE], base);
    let mut painted = Painted::new(base);
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    point_sampling(h, FVF_2D);
    let mut written = 0usize;
    run_rounds(
        h,
        frames,
        grid,
        |cell| {
            if evict_before == Some(written) {
                assert_eq!(h.evict_managed_resources(), 0, "EvictManagedResources");
            }
            let (w, hgt) = SPRITES[(written * 7) % SPRITES.len()];
            let color = palette[1 + written % (palette.len() - 1)];
            written += 1;
            write([0, 0, w, hgt], color);
            painted.fill([0, 0, w, hgt], color);
            let probes = SPRITE_PROBES
                .iter()
                .map(|&texel| (cell.pixel(texel, (VIEW, VIEW)), painted.at(texel)))
                .collect();
            (probes, Draw::Flat(cell.quad((0.5, 0.5))))
        },
        context,
    );
    // The rest of the texture never changed.
    h.render_once(BLACK, |d| {
        Draw::Flat(Cell::of_grid(0, 1, 1).quad((1.0, 1.0))).issue(d);
    });
    let outside = Cell::of_grid(0, 1, 1).pixel((400, 400), (SIZE, SIZE));
    check_frame(
        h,
        &[Expected {
            cell: 0,
            probes: vec![(outside, base)],
        }],
        &format!("{context}, outside the sprites"),
    );
    assert_eq!(h.clear_texture(0), 0);
}

/// The A4R4G4B4 scratch tests' colours, the first the fill they start with.
///
/// Nibbles of 0, 8 and 15, which A4R4G4B4 stores as 0x00, 0x88 and 0xFF.
const ARGB4_PALETTE: [u32; 13] = [
    0xFF88_8888,
    RED,
    GREEN,
    BLUE,
    YELLOW,
    CYAN,
    MAGENTA,
    0xFF88_0000,
    0xFF00_8800,
    0xFF00_0088,
    0xFF88_8800,
    0xFF00_8888,
    WHITE,
];

/// The sprite-scratch shape: a 512x512 A4R4G4B4 texture rewritten at `(0, 0)` 32 times a frame.
///
/// Each frame's 31 renames after its first draw retire 31 versions, which are
/// spares by the next frame, so from the second frame on every rename reuses
/// a version that holds the texture as it was one or two frames ago and has
/// to receive every sprite written since: the box at `(0, 0)` they cover.
#[test]
fn a_sprite_scratch_rewritten_between_draws_keeps_each_draws_texels() {
    let h = Harness::new();
    sprite_scratch(
        &h,
        (Format::Argb4, Writer::UpdateSurface),
        &ARGB4_PALETTE,
        (4, (8, 4)),
        None,
        "A4R4G4B4 sprite scratch",
    );
}

/// The game's own pattern: a managed A4R4G4B4 scratch relocked at `(0, 0)` between draws.
///
/// Each sprite is a `LockRect` of its rectangle alone, written and unlocked
/// before the draw that shows it, 32 times a frame. Every draw shows its own
/// sprite, and the texels no later sprite covered keep what earlier ones
/// wrote, whichever version a rename took.
#[test]
fn a_managed_scratch_relocked_between_draws_keeps_each_draws_texels() {
    let h = Harness::new();
    sprite_scratch(
        &h,
        (Format::Argb4, Writer::PartialLock),
        &ARGB4_PALETTE,
        (4, (8, 4)),
        None,
        "managed A4R4G4B4 scratch, partial locks",
    );
}

/// `EvictManagedResources` between the renames of a frame.
///
/// The eviction drops the texture from the encoder's cache while the 27
/// versions its renames retired that frame are still in flight, and the
/// texture comes back under the same id at its next draw, where its next
/// renames start a new write history. Those versions belong to the
/// texture's former life and must be destroyed, not parked as spares of the
/// new one: each holds a write count of the old history, higher than any
/// the new one has reached a frame later, so a rename reusing one would copy
/// nothing into it and show the sprites of the frame before.
#[test]
fn an_eviction_between_renames_keeps_the_former_versions_out_of_the_pool() {
    let h = Harness::new();
    sprite_scratch(
        &h,
        (Format::Argb4, Writer::PartialLock),
        &ARGB4_PALETTE,
        (4, (8, 4)),
        Some(28),
        "managed A4R4G4B4 scratch evicted mid-frame",
    );
}

/// The sprite scratch in A8R8G8B8, read through its sRGB view.
///
/// A reused version must come back with its sampling views registered, so
/// the sRGB reads see the version the draw bound. The colours are 0 or 255
/// per channel, which the sRGB decode keeps.
#[test]
fn an_srgb_read_scratch_keeps_each_draws_texels() {
    const PALETTE: [u32; 8] = [MAGENTA, RED, GREEN, BLUE, YELLOW, CYAN, WHITE, BLACK];
    let h = Harness::new();
    assert_eq!(h.set_sampler_state(0, D3DSAMP_SRGBTEXTURE, 1), 0);
    sprite_scratch(
        &h,
        (Format::Argb8, Writer::UpdateSurface),
        &PALETTE,
        (3, (4, 4)),
        None,
        "sRGB A8R8G8B8 sprite scratch",
    );
    assert_eq!(h.set_sampler_state(0, D3DSAMP_SRGBTEXTURE, 0), 0);
}

/// One patch per frame in a new corner: a reused version gets every corner written since.
///
/// The frame-1 rename retires the first version; frame 2 reuses it and must
/// copy frame 1's corner into it; frame 3 reuses the version frame 1 made,
/// which holds frame 1's corner but not frame 2's, at the opposite corner.
#[test]
fn a_reused_version_receives_writes_from_every_frame_since_it_retired() {
    const SIZE: u32 = 256;
    let h = Harness::new();
    let tex = h.create_texture(SIZE, SIZE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let level = tex.surface_level(0);
    let patcher = Patcher::new(&h, (SIZE, SIZE), Format::Argb8);
    patcher.patch(&h, &level, [0, 0, SIZE, SIZE], GRAY);
    let mut painted = Painted::new(GRAY);
    assert_eq!(h.set_texture(0, &tex), 0);
    point_sampling(&h, FVF_2D);
    let corners = [
        ([0, 0, 64, 64], RED),
        ([192, 192, 256, 256], GREEN),
        ([192, 0, 256, 64], BLUE),
        ([0, 192, 64, 256], YELLOW),
        ([96, 96, 160, 160], CYAN),
    ];
    let probes = [
        (32, 32),
        (224, 224),
        (224, 32),
        (32, 224),
        (128, 128),
        (128, 32),
    ];
    let screen = Cell::of_grid(0, 1, 1);
    for (frame, (rect, color)) in corners.into_iter().enumerate() {
        h.render_once(BLACK, |d| {
            Draw::Flat(screen.quad((1.0, 1.0))).issue(d);
            patcher.patch(d, &level, rect, color);
            painted.fill(rect, color);
            Draw::Flat(screen.quad((1.0, 1.0))).issue(d);
        });
        let expected = Expected {
            cell: 0,
            probes: probes
                .iter()
                .map(|&texel| (screen.pixel(texel, (SIZE, SIZE)), painted.at(texel)))
                .collect(),
        };
        check_frame(&h, &[expected], &format!("corner patch of frame {frame}"));
    }
    assert_eq!(h.clear_texture(0), 0);
}

/// Half-extents of the nested squares the cube test writes, centred on its 64-texel level.
const SQUARES: [u32; 6] = [8, 16, 24, 4, 20, 12];

/// Offsets from the centre of the cube test's probes, each between two square edges.
const RING_OFFSETS: [u32; 7] = [1, 6, 10, 14, 18, 22, 28];

/// Level 1 of face 3 of a cube, patched with centred squares, the other faces and levels untouched.
///
/// Every square is centred on the 64-texel level and every probe is mirrored
/// on both axes, so the check holds whatever orientation the face has.
#[test]
fn a_cube_face_level_rewritten_between_draws_keeps_each_draws_texels() {
    const EDGE: u32 = 128;
    const LEVEL1: u32 = EDGE / 2;
    const FACES: [u32; 6] = [RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA];
    let h = Harness::new();
    let cube: CubeTexture<'_> =
        h.create_cube_texture_owned(EDGE, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let level0 = Patcher::new(&h, (EDGE, EDGE), Format::Argb8);
    let level1 = Patcher::new(&h, (LEVEL1, LEVEL1), Format::Argb8);
    for (face, color) in (0..).zip(FACES) {
        level0.patch(&h, &cube.surface(face, 0), [0, 0, EDGE, EDGE], color);
        level1.patch(&h, &cube.surface(face, 1), [0, 0, LEVEL1, LEVEL1], GRAY);
    }
    let target = cube.surface(3, 1);
    let mut painted = Painted::new(GRAY);
    assert_eq!(h.set_cube_texture(0, &cube), 0);
    point_sampling(&h, FVF_CUBE);
    sample_level(&h, 1);
    let probes: Vec<(u32, u32)> = RING_OFFSETS
        .iter()
        .flat_map(|&r| {
            let c = LEVEL1 / 2;
            [(c + r, c), (c - 1 - r, c), (c, c + r), (c, c - 1 - r)]
        })
        .collect();
    let colors = [WHITE, RED, GREEN, BLUE, YELLOW, BLACK];
    let mut written = 0usize;
    run_rounds(
        &h,
        3,
        (2, 2),
        |cell| {
            let half = SQUARES[written % SQUARES.len()];
            let color = colors[written % colors.len()];
            written += 1;
            let c = LEVEL1 / 2;
            let rect = [c - half, c - half, c + half, c + half];
            level1.patch(&h, &target, rect, color);
            painted.fill(rect, color);
            let expected = probes
                .iter()
                .map(|&texel| (cell.pixel(texel, (LEVEL1, LEVEL1)), painted.at(texel)))
                .collect();
            (expected, Draw::Cube(cell.negative_y_face_quad()))
        },
        "cube face 3 level 1",
    );
    // Level 0 of the same face kept its colour.
    sample_level(&h, 0);
    let screen = Cell::of_grid(0, 1, 1);
    h.render_once(BLACK, |d| {
        Draw::Cube(screen.negative_y_face_quad()).issue(d);
    });
    check_frame(
        &h,
        &[Expected {
            cell: 0,
            probes: vec![((320, 240), FACES[3])],
        }],
        "level 0 of face 3",
    );
    assert_eq!(h.clear_texture(0), 0);
}

/// Two levels of one texture patched in turn: a reused version gets the writes of both.
#[test]
fn two_levels_rewritten_between_draws_keep_each_draws_texels() {
    const EDGE: u32 = 128;
    let h = Harness::new();
    let tex: Texture<'_> = h.create_texture(EDGE, EDGE, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let levels = [tex.surface_level(0), tex.surface_level(1)];
    let patchers = [
        Patcher::new(&h, (EDGE, EDGE), Format::Argb8),
        Patcher::new(&h, (EDGE / 2, EDGE / 2), Format::Argb8),
    ];
    patchers[0].patch(&h, &levels[0], [0, 0, EDGE, EDGE], GRAY);
    patchers[1].patch(&h, &levels[1], [0, 0, EDGE / 2, EDGE / 2], WHITE);
    let mut painted = [Painted::new(GRAY), Painted::new(WHITE)];
    assert_eq!(h.set_texture(0, &tex), 0);
    point_sampling(&h, FVF_2D);
    // Per level: rectangles in whole 32-texel (level 0) or 16-texel (level 1)
    // blocks, and the centres of those blocks to probe.
    let rects: [[[u32; 4]; 4]; 2] = [
        [
            [0, 0, 64, 64],
            [64, 64, 128, 128],
            [32, 0, 96, 32],
            [0, 96, 128, 128],
        ],
        [
            [0, 0, 32, 32],
            [32, 32, 64, 64],
            [16, 0, 48, 16],
            [0, 48, 64, 64],
        ],
    ];
    let probes: [Vec<(u32, u32)>; 2] = [32, 16].map(|block: u32| {
        (0..4)
            .flat_map(|y| (0..4).map(move |x| (x * block + block / 2, y * block + block / 2)))
            .collect()
    });
    let colors = [RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, BLACK];
    let mut written = 0usize;
    run_rounds(
        &h,
        3,
        (2, 2),
        |cell| {
            let level = written % 2;
            let rect = rects[level][(written / 2) % 4];
            let color = colors[written % colors.len()];
            written += 1;
            patchers[level].patch(&h, &levels[level], rect, color);
            painted[level].fill(rect, color);
            sample_level(&h, u32::try_from(level).expect("level 0 or 1"));
            let extent = EDGE >> level;
            let expected = probes[level]
                .iter()
                .map(|&texel| {
                    (
                        cell.pixel(texel, (extent, extent)),
                        painted[level].at(texel),
                    )
                })
                .collect();
            (expected, Draw::Flat(cell.quad((1.0, 1.0))))
        },
        "two levels",
    );
    sample_level(&h, 0);
    assert_eq!(h.clear_texture(0), 0);
}

/// An `AUTOGENMIPMAP` texture patched at level 0: level 1 follows each patch.
///
/// The generated level is a box filter of level 0, and every patch covers
/// whole 32-texel blocks, so each 16-texel block of level 1 is one colour.
/// A reused version must receive the regenerated level whole.
#[test]
fn an_autogen_texture_rewritten_between_draws_regenerates_for_each_draw() {
    const EDGE: u32 = 128;
    let h = Harness::new();
    let tex = h.create_texture(
        EDGE,
        EDGE,
        0,
        D3DUSAGE_AUTOGENMIPMAP,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let level = tex.surface_level(0);
    let patcher = Patcher::new(&h, (EDGE, EDGE), Format::Argb8);
    patcher.patch(&h, &level, [0, 0, EDGE, EDGE], GRAY);
    let mut painted = Painted::new(GRAY);
    assert_eq!(h.set_texture(0, &tex), 0);
    point_sampling(&h, FVF_2D);
    sample_level(&h, 1);
    let rects = [
        [0, 0, 64, 64],
        [64, 64, 128, 128],
        [32, 32, 96, 96],
        [96, 0, 128, 32],
        [0, 96, 32, 128],
    ];
    let colors = [RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, WHITE];
    let probes: Vec<(u32, u32)> = (0..4)
        .flat_map(|y| (0..4).map(move |x| (x * 16 + 8, y * 16 + 8)))
        .collect();
    let mut written = 0usize;
    run_rounds(
        &h,
        3,
        (2, 2),
        |cell| {
            let rect = rects[written % rects.len()];
            let color = colors[written % colors.len()];
            written += 1;
            patcher.patch(&h, &level, rect, color);
            painted.fill(rect, color);
            let expected = probes
                .iter()
                .map(|&(x, y)| {
                    (
                        cell.pixel((x, y), (EDGE / 2, EDGE / 2)),
                        painted.at((2 * x, 2 * y)),
                    )
                })
                .collect();
            (expected, Draw::Flat(cell.quad((1.0, 1.0))))
        },
        "AUTOGENMIPMAP level 1",
    );
    sample_level(&h, 0);
    assert_eq!(h.clear_texture(0), 0);
}

/// A 100x60 DXT1 texture patched in whole blocks, some ending at the level's odd edge.
#[test]
fn a_compressed_texture_rewritten_between_draws_keeps_each_draws_texels() {
    const SIZE: (u32, u32) = (100, 60);
    let h = Harness::new();
    let tex = h.create_texture(SIZE.0, SIZE.1, 1, 0, D3DFMT_DXT1, D3DPOOL_DEFAULT);
    let level = tex.surface_level(0);
    let patcher = Patcher::new(&h, SIZE, Format::Dxt1);
    patcher.patch(&h, &level, [0, 0, SIZE.0, SIZE.1], WHITE);
    let mut painted = Painted::new(WHITE);
    assert_eq!(h.set_texture(0, &tex), 0);
    point_sampling(&h, FVF_2D);
    let rects = [
        [0, 0, 32, 16],
        [64, 32, 100, 60],
        [32, 16, 96, 48],
        [0, 40, 24, 60],
        [76, 0, 100, 24],
        [8, 8, 40, 40],
        [48, 4, 92, 28],
    ];
    let colors = [RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, BLACK];
    // The centre texel of every other block: each block is one colour.
    let probes: Vec<(u32, u32)> = (0..SIZE.1 / 8)
        .flat_map(|y| (0..SIZE.0.div_ceil(8)).map(move |x| (x * 8 + 2, y * 8 + 2)))
        .collect();
    let mut written = 0usize;
    run_rounds(
        &h,
        3,
        (2, 2),
        |cell| {
            let rect = rects[written % rects.len()];
            let color = colors[written % colors.len()];
            written += 1;
            patcher.patch(&h, &level, rect, color);
            painted.fill(rect, color);
            let expected = probes
                .iter()
                .map(|&texel| (cell.pixel(texel, SIZE), painted.at(texel)))
                .collect();
            (expected, Draw::Flat(cell.quad((1.0, 1.0))))
        },
        "DXT1 100x60",
    );
    assert_eq!(h.clear_texture(0), 0);
}

/// A 64x64 A8R8G8B8 texture rewritten between draws, checked per draw.
///
/// A default-pool texture takes `UpdateSurface` copies; a managed one
/// (`managed`) a whole-level lock that rewrites the level as `painted` says
/// and publishes only the changed rectangle with `AddDirtyRect`.
fn small_scratch(
    h: &Harness,
    tex: &Texture<'_>,
    managed: bool,
    painted: &mut Painted,
    colors: &[u32],
    frames: u32,
    context: &str,
) {
    const EDGE: u32 = 64;
    let patcher = Patcher::new(h, (EDGE, EDGE), Format::Argb8);
    let level = tex.surface_level(0);
    let rects = [
        [0, 0, 32, 32],
        [16, 16, 48, 48],
        [32, 0, 64, 64],
        [0, 32, 64, 64],
    ];
    let probes: Vec<(u32, u32)> = (0..4)
        .flat_map(|y| (0..4).map(move |x| (x * 16 + 8, y * 16 + 8)))
        .collect();
    assert_eq!(h.set_texture(0, tex), 0);
    point_sampling(h, FVF_2D);
    let mut written = 0usize;
    run_rounds(
        h,
        frames,
        (2, 2),
        |cell| {
            let rect = rects[written % rects.len()];
            let color = colors[written % colors.len()];
            written += 1;
            painted.fill(rect, color);
            if managed {
                let image: Vec<u32> = (0..EDGE * EDGE)
                    .map(|i| painted.at((i % EDGE, i / EDGE)))
                    .collect();
                tex.lock_rect(0, D3DLOCK_NO_DIRTY_UPDATE).write_u32_rect(
                    EDGE as usize,
                    EDGE as usize,
                    &image,
                );
                let signed = rect.map(|v| i32::try_from(v).expect("small"));
                assert_eq!(tex.add_dirty_rect_partial(&signed), 0, "AddDirtyRect");
            } else {
                patcher.patch(h, &level, rect, color);
            }
            let expected = probes
                .iter()
                .map(|&texel| (cell.pixel(texel, (EDGE, EDGE)), painted.at(texel)))
                .collect();
            (expected, Draw::Flat(cell.quad((1.0, 1.0))))
        },
        context,
    );
    assert_eq!(h.clear_texture(0), 0);
}

/// Releasing a texture with spares parked, then making another of the same shape.
///
/// The released texture's spares and retiring versions are destroyed behind
/// the frame that released it; the new texture starts with no spares, and
/// nothing of the old one's contents shows in it.
#[test]
fn a_texture_released_with_spares_parked_leaves_its_successor_clean() {
    const EDGE: u32 = 64;
    let h = Harness::new();
    {
        let first = h.create_texture(EDGE, EDGE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
        Patcher::new(&h, (EDGE, EDGE), Format::Argb8).patch(
            &h,
            &first.surface_level(0),
            [0, 0, EDGE, EDGE],
            GRAY,
        );
        let mut painted = Painted::new(GRAY);
        small_scratch(
            &h,
            &first,
            false,
            &mut painted,
            &[RED, GREEN, BLUE],
            2,
            "first texture",
        );
        // One more frame parks the last frame's versions as spares.
        assert_eq!(h.present(), 0);
    }
    assert_eq!(h.present(), 0, "the frame that releases the first texture");
    let second = h.create_texture(EDGE, EDGE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    Patcher::new(&h, (EDGE, EDGE), Format::Argb8).patch(
        &h,
        &second.surface_level(0),
        [0, 0, EDGE, EDGE],
        WHITE,
    );
    let mut painted = Painted::new(WHITE);
    small_scratch(
        &h,
        &second,
        false,
        &mut painted,
        &[CYAN, MAGENTA, YELLOW],
        3,
        "second texture",
    );
}

/// `Reset` with spares parked: the managed texture survives and keeps rewriting correctly.
#[test]
fn reset_with_spares_parked_keeps_rewrites_correct() {
    const EDGE: u32 = 64;
    let h = Harness::new();
    let tex = h.create_texture(EDGE, EDGE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    tex.lock_rect(0, 0).write_u32_rect(
        EDGE as usize,
        EDGE as usize,
        &[GRAY; (EDGE * EDGE) as usize],
    );
    let mut painted = Painted::new(GRAY);
    small_scratch(
        &h,
        &tex,
        true,
        &mut painted,
        &[RED, GREEN, BLUE],
        2,
        "before Reset",
    );
    assert_eq!(h.present(), 0, "park the last frame's versions");
    assert_eq!(h.reset(BACK.0, BACK.1), 0, "Reset");
    let colors = [YELLOW, CYAN, MAGENTA, WHITE];
    small_scratch(&h, &tex, true, &mut painted, &colors, 3, "after Reset");
}
