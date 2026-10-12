//! What the Metal HUD's `mtld3d` rows say, and the rules that keep them readable.
//!
//! On macOS 27 the HUD lists every state reporter under its "State Reporters"
//! caption: the domain and its label on one row, then one row per metadata
//! key, indented by two spaces. The layer reports its release identity as the
//! label and a handful of per-device figures as volatile metadata; this module
//! holds everything about those rows that needs no Objective-C: the counters
//! the encoder keeps, the row keys, the values and the width budget.
//!
//! The HUD draws a metadata value only when it is a number or a date: a
//! string value has no `stringValue`, so the HUD skips its row. Every value is
//! therefore an integer, and a unit goes into the key (`MiB`).
//!
//! Four rules keep the panel readable whatever the figures are, and the tests
//! hold each one:
//!
//! 1. Width. The HUD centre-truncates a row whose indent, key and value
//!    together exceed the panel's character columns. [`HUD_COLUMNS`] is the
//!    budget the default panel gives, and every key with its widest possible
//!    value leaves at least one column between them.
//! 2. Bounded values. Counts are `u32`, at most ten digits; byte figures are
//!    whole MiB, saturating at `u32::MAX`; the two 32-bit address-space rows
//!    never exceed the 4096 MiB such a space has.
//! 3. Order. The HUD sorts the keys with `compare:`, which orders ASCII by
//!    code point. Every key starts with its group word, and the groups read in
//!    the order `Frame`, `Live`, `Mem`, `Shaders`. Wherever two adjacent keys
//!    first differ, byte order and case-insensitive order agree, so a HUD that
//!    sorted either way would show the same order.
//! 4. Header. The domain row carries [`HUD_DOMAIN`] and the release identity,
//!    which [`hud_label`] shortens to fit beside the domain.

use crate::{LOG_TARGET, dxso::ShaderType};

/// The `StateReporting` domain the layer reports under, which the HUD prints as the header row.
pub const HUD_DOMAIN: &str = "mtld3d";

/// Character columns the HUD's default panel gives a row.
///
/// The HUD derives its budget from the panel's width in points divided by
/// the advance of its monospace "M"; the default panel holds 33.
pub const HUD_COLUMNS: usize = 33;

/// Columns the HUD puts before every metadata key.
pub const HUD_KEY_INDENT: usize = 2;

/// The longest release identity the domain row carries beside the domain and one space.
pub const HUD_LABEL_MAX: usize = HUD_COLUMNS - HUD_DOMAIN.len() - 1;

/// How many rows the layer can report under its domain.
pub const HUD_ROW_COUNT: usize = 19;

/// The metadata keys, in the order the HUD sorts them, indexed like [`hud_rows`]' answer.
///
/// `UP` draws come from user-pointer data (`DrawPrimitiveUP`,
/// `DrawIndexedPrimitiveUP`), `VB` draws from vertex buffers. `FF` counts the
/// fixed-function shaders the layer generated, `SM1` to `SM3` the game's
/// shader programs by shader model.
pub const HUD_ROW_KEYS: [&str; HUD_ROW_COUNT] = [
    "Frame draws UP",
    "Frame draws VB",
    "Frame passes",
    "Live pipelines",
    "Live samplers",
    "Live textures",
    "Mem 32-bit free MiB",
    "Mem 32-bit largest MiB",
    "Mem CPU heap MiB",
    "Mem in flight MiB",
    "Mem reuse pool MiB",
    "Shaders PS FF",
    "Shaders PS SM1",
    "Shaders PS SM2",
    "Shaders PS SM3",
    "Shaders VS FF",
    "Shaders VS SM1",
    "Shaders VS SM2",
    "Shaders VS SM3",
];

/// How many samples a second the encoder reports at most.
pub const HUD_SAMPLE_HZ: u64 = 2;

/// The size of a 32-bit address space in MiB, which bounds both address-space rows.
const ADDRESS_SPACE_32_MIB: u32 = 4096;

/// Index of the first address-space row in [`HUD_ROW_KEYS`].
const ROW_MEM_32_FREE: usize = 6;

/// Index of the first shader row in [`HUD_ROW_KEYS`]; four per stage, pixel stage first.
const ROW_SHADERS: usize = 11;

/// Shortest commit hash [`hud_label`] keeps when it shortens a development identity.
const MIN_LABEL_HASH: usize = 4;

/// Draw and pass counts of the frame being encoded, and of the last presented one.
///
/// The encoder counts every draw it replays and every render pass it
/// submits. A mid-frame flush submits part of a frame, so its passes add to
/// the frame's; a presenting submission freezes the frame's counts as the
/// ones the HUD shows and starts the next frame at zero.
#[derive(Default)]
pub struct HudFrameStats {
    draws: u32,
    draws_up: u32,
    passes: u32,
    shown: HudFrame,
}

impl HudFrameStats {
    /// Count one replayed draw; `up` when its vertices are user-pointer data.
    ///
    /// Two wrapping adds and no branch, since it runs once per draw.
    #[inline]
    pub fn count_draw(&mut self, up: bool) {
        self.draws = self.draws.wrapping_add(1);
        self.draws_up = self.draws_up.wrapping_add(u32::from(up));
    }

    /// Add one submission's render passes, and freeze the frame when the submission presents.
    pub fn end_submit(&mut self, passes: usize, presented: bool) {
        self.passes = self
            .passes
            .wrapping_add(u32::try_from(passes).unwrap_or(u32::MAX));
        if presented {
            self.shown = HudFrame {
                draws_up: self.draws_up,
                draws_vb: self.draws.wrapping_sub(self.draws_up),
                passes: self.passes,
            };
            self.draws = 0;
            self.draws_up = 0;
            self.passes = 0;
        }
    }

    /// The counts of the last presented frame.
    #[must_use]
    pub const fn shown(&self) -> &HudFrame {
        &self.shown
    }
}

/// The draw and pass counts of one presented frame.
#[derive(Default)]
pub struct HudFrame {
    draws_up: u32,
    draws_vb: u32,
    passes: u32,
}

impl HudFrame {
    /// Draws whose vertices were user-pointer data.
    #[must_use]
    pub const fn draws_up(&self) -> u32 {
        self.draws_up
    }

    /// Draws whose vertices came from vertex buffers.
    #[must_use]
    pub const fn draws_vb(&self) -> u32 {
        self.draws_vb
    }

    /// Render passes the frame's submissions carried.
    #[must_use]
    pub const fn passes(&self) -> u32 {
        self.passes
    }
}

/// The game's shader programs a device holds, by stage and shader model.
///
/// The encoder keeps every parsed program for the device's life, keyed by its
/// bytecode, so a program is counted once, when the device first sees it.
#[derive(Default)]
pub struct ShaderModels {
    /// Programs by stage (pixel, vertex) and shader model (1, 2, 3).
    counts: [[u32; 3]; 2],
}

impl ShaderModels {
    /// Count a program the device has not held before.
    ///
    /// A major version outside 1 to 3 is not counted; the parser accepts
    /// none, so one here is logged once.
    pub fn adopt(&mut self, stage: ShaderType, major: u8) {
        let Some(model) = usize::from(major).checked_sub(1).filter(|model| *model < 3) else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "hud stats: a shader program with major version {major} is not counted"
            );
            return;
        };
        let slot = &mut self.counts[stage_index(stage)][model];
        *slot = slot.saturating_add(1);
    }

    /// The programs of `stage`, by shader model 1, 2 and 3.
    #[must_use]
    pub const fn counts(&self, stage: ShaderType) -> [u32; 3] {
        self.counts[stage_index(stage)]
    }
}

/// Lets a sample through at most once per period of a monotonic clock.
#[derive(Default)]
pub struct SampleGate {
    last: Option<u64>,
}

impl SampleGate {
    /// Whether a sample is due at `now`: the first call, and then once `period` has passed.
    pub const fn due(&mut self, now: u64, period: u64) -> bool {
        if let Some(last) = self.last
            && now.wrapping_sub(last) < period
        {
            return false;
        }
        self.last = Some(now);
        true
    }
}

/// Every figure one sample reports, gathered by the encoder.
pub struct HudSample {
    /// The last presented frame's user-pointer draws.
    pub draws_up: u32,
    /// The last presented frame's vertex-buffer draws.
    pub draws_vb: u32,
    /// The last presented frame's render passes.
    pub passes: u32,
    /// Render pipeline states the device holds.
    pub pipelines: usize,
    /// Sampler states the device holds.
    pub samplers: usize,
    /// Textures alive on the device.
    pub textures: usize,
    /// The address-space watch's last walk, packed by [`pack_free_va`]; zero before one.
    pub free_va: u64,
    /// The native runtime's own heap.
    pub cpu_heap_bytes: u64,
    /// Vertex and index buffer copies held until the GPU has finished with them.
    pub in_flight_bytes: u64,
    /// Spare memory parked for reuse.
    pub reuse_pool_bytes: u64,
    /// Fixed-function shaders generated, pixel stage then vertex stage.
    pub fixed_function: [usize; 2],
    /// The game's programs by shader model 1 to 3, pixel stage then vertex stage.
    pub models: [[u32; 3]; 2],
}

/// The value of every row, indexed like [`HUD_ROW_KEYS`]; `None` leaves the row out.
///
/// The address-space rows are left out until the watch has walked once,
/// which a 64-bit process never does, and a shader row while it counts no
/// shader, so a game shows only the shader models it uses.
#[must_use]
pub fn hud_rows(sample: &HudSample) -> [Option<u32>; HUD_ROW_COUNT] {
    let mut rows = [None; HUD_ROW_COUNT];
    rows[0] = Some(sample.draws_up);
    rows[1] = Some(sample.draws_vb);
    rows[2] = Some(sample.passes);
    rows[3] = Some(count(sample.pipelines));
    rows[4] = Some(count(sample.samplers));
    rows[5] = Some(count(sample.textures));
    if sample.free_va != 0 {
        let (free, largest) = unpack_free_va(sample.free_va);
        rows[ROW_MEM_32_FREE] = Some(free.min(ADDRESS_SPACE_32_MIB));
        rows[ROW_MEM_32_FREE + 1] = Some(largest.min(ADDRESS_SPACE_32_MIB));
    }
    rows[8] = Some(mib(sample.cpu_heap_bytes));
    rows[9] = Some(mib(sample.in_flight_bytes));
    rows[10] = Some(mib(sample.reuse_pool_bytes));
    for (stage, (fixed, models)) in sample.fixed_function.iter().zip(&sample.models).enumerate() {
        let first = ROW_SHADERS + stage * 4;
        let values = [count(*fixed), models[0], models[1], models[2]];
        for (row, value) in rows[first..first + 4].iter_mut().zip(values) {
            *row = (value != 0).then_some(value);
        }
    }
    rows
}

/// Pack one address-space walk into the word the watch thread stores for the encoder.
///
/// Free MiB in the high half and the largest free block's MiB in the low
/// half, each saturating at `u32::MAX`. A walk that found nothing free packs
/// to zero, which reads as no walk yet.
#[must_use]
pub fn pack_free_va(free_mib: u64, largest_mib: u64) -> u64 {
    let half = |mib: u64| u64::from(u32::try_from(mib).unwrap_or(u32::MAX));
    (half(free_mib) << 32) | half(largest_mib)
}

/// The free and largest-free-block MiB of a word [`pack_free_va`] packed.
#[must_use]
pub const fn unpack_free_va(packed: u64) -> (u32, u32) {
    let [h0, h1, h2, h3, l0, l1, l2, l3] = packed.to_be_bytes();
    (
        u32::from_be_bytes([h0, h1, h2, h3]),
        u32::from_be_bytes([l0, l1, l2, l3]),
    )
}

/// The release identity as the domain row shows it, at most [`HUD_LABEL_MAX`] bytes.
///
/// An identity that fits is returned as it is. A longer development identity
/// (`<tag>-<n>-g<hash>`, with or without `-dirty`) keeps its tag, its commit
/// distance and `-dirty`, and gives up hash digits, down to four; with room
/// for fewer it drops the hash and its `-g` altogether. Anything longer still
/// keeps its first [`HUD_LABEL_MAX`] bytes.
#[must_use]
pub fn hud_label(build: &str) -> String {
    if build.len() <= HUD_LABEL_MAX {
        return build.to_owned();
    }
    let (body, dirty) = build
        .strip_suffix("-dirty")
        .map_or((build, ""), |body| (body, "-dirty"));
    if let Some(at) = body.rfind("-g") {
        let (head, hash) = body.split_at(at + 2);
        let room = HUD_LABEL_MAX.saturating_sub(head.len() + dirty.len());
        if hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            if room >= MIN_LABEL_HASH {
                return format!("{head}{}{dirty}", &hash[..room.min(hash.len())]);
            }
            // Too little room for a useful hash: drop it with its `-g`.
            let short = format!("{}{dirty}", &body[..at]);
            if short.len() <= HUD_LABEL_MAX {
                return short;
            }
        }
    }
    let mut end = HUD_LABEL_MAX;
    while !build.is_char_boundary(end) {
        end -= 1;
    }
    build[..end].to_owned()
}

/// Index of `stage` in the per-stage arrays: pixel first, as the keys sort.
const fn stage_index(stage: ShaderType) -> usize {
    match stage {
        ShaderType::Pixel => 0,
        ShaderType::Vertex => 1,
    }
}

/// A collection length as a row value, saturating at `u32::MAX`.
fn count(len: usize) -> u32 {
    u32::try_from(len).unwrap_or(u32::MAX)
}

/// Whole MiB of `bytes`, saturating at `u32::MAX`.
fn mib(bytes: u64) -> u32 {
    u32::try_from(bytes >> 20).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests;
