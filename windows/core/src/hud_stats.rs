//! What the Metal HUD's `mtld3d` rows say, and the rules that keep them readable.
//!
//! On macOS 27 the HUD lists every state reporter under its "State Reporters"
//! caption: the domain and its label on one row, then one row per metadata
//! key, indented by two spaces. The layer reports its release identity as the
//! label and a handful of figures as volatile metadata; this module holds
//! everything about those rows that needs no Objective-C: the counters the
//! encoder keeps, the row keys, the values and the width budget.
//!
//! The HUD draws a metadata value only when it is a number or a date: a
//! string value has no `stringValue`, so the HUD skips its row. Every value is
//! therefore an integer, and a unit goes into the key (`MiB`).
//!
//! Five rules keep the panel readable whatever the figures are, and the tests
//! hold each one:
//!
//! 1. Width. The HUD centre-truncates a row whose indent, key and value
//!    together exceed the panel's character columns. [`HUD_COLUMNS`] is the
//!    budget the default panel gives, and every key with its widest possible
//!    value leaves at least one column between them.
//! 2. Bounded values. Counts are `u32`, at most ten digits; byte figures are
//!    whole MiB, saturating at `u32::MAX`; the 32-bit address-space row never
//!    exceeds the 4096 MiB such a space has.
//! 3. Order. The HUD sorts the keys with `compare:`, which orders ASCII by
//!    code point. Every key starts with its group word, and the groups read in
//!    the order `Frame`, `Live`, `Mem`, `Shaders`. Wherever two adjacent keys
//!    first differ, byte order and case-insensitive order agree, so a HUD that
//!    sorted either way would show the same order.
//! 4. Header. The domain row carries [`HUD_DOMAIN`] and the release identity,
//!    which [`hud_label`] shortens to fit beside the domain.
//! 5. Stability. A row appears only once it carries information, its first
//!    nonzero sampled value on the device, and never disappears afterwards,
//!    so the panel does not jump as a figure falls back to 0 ([`HudRowLatch`]).

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
pub const HUD_ROW_COUNT: usize = 10;

/// The metadata keys, in the order the HUD sorts them, indexed like [`hud_rows`]' answer.
///
/// `UP` draws come from user-pointer data (`DrawPrimitiveUP`,
/// `DrawIndexedPrimitiveUP`), `VB` draws from vertex buffers. Both shader rows
/// count distinct shaders seen since the device was created, not live objects,
/// and neither drops an entry when the game releases a shader. `Shaders FF`
/// counts the fixed-function shader keys the encoder built from the game's
/// draw state: those of a stage with no shader bound, and the vertex shader of
/// a pre-transformed (`XYZRHW`, `POSITIONT`) draw. They are not D3D objects.
/// Fixed-function vertex keys whose build failed leave the count when the
/// encoder's reset cleanup runs (a `Reset` that changes the implicit surfaces
/// or the device window, or an automatic back-buffer resize), until a draw
/// that uses one records it again. `Shaders programmable` counts the game's
/// own shader programs by bytecode, so the same bytecode created twice counts
/// once, and only shrinks when the device is destroyed.
pub const HUD_ROW_KEYS: [&str; HUD_ROW_COUNT] = [
    "Frame draws UP",
    "Frame draws VB",
    "Frame passes",
    "Live textures",
    "Mem 32-bit largest MiB",
    "Mem guest pages MiB",
    "Mem in flight MiB",
    "Mem reuse pool MiB",
    "Shaders FF",
    "Shaders programmable",
];

/// How many samples a second the encoder reports at most.
pub const HUD_SAMPLE_HZ: u64 = 2;

/// The size of a 32-bit address space in MiB, which bounds the address-space row.
const ADDRESS_SPACE_32_MIB: u32 = 4096;

/// The index of the address-space row, `Mem 32-bit largest MiB`, in [`HUD_ROW_KEYS`].
const ADDRESS_SPACE_ROW: usize = 4;

const _: () = assert!(HUD_ROW_COUNT <= u16::BITS as usize, "one latch bit per row");

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

/// Which rows a device has shown, so each appears once it carries information and then stays.
///
/// A row is hidden until its first nonzero sampled value and shown from then
/// on, at 0 too. The latch sees only the samples its reporter takes: at most
/// [`HUD_SAMPLE_HZ`] a second, the frame counts from the last presented frame,
/// and only while this device's encoder holds the rows. A device that never
/// draws from user-pointer data shows no `Frame draws UP` row, and one that
/// does keeps the row through frames that have none. The address-space row
/// latches on any value, since a walk that found less than 1 MiB free shows 0
/// and that is the state the row exists to show. One latch lives as long as
/// its device's reporter.
#[derive(Default)]
pub struct HudRowLatch {
    /// Bit `i` set once row `i` of [`HUD_ROW_KEYS`] has carried information.
    seen: u16,
}

impl HudRowLatch {
    /// The rows the panel shows of `rows`: each one latched so far, with its current value.
    #[must_use]
    pub fn visible(
        &mut self,
        mut rows: [Option<u32>; HUD_ROW_COUNT],
    ) -> [Option<u32>; HUD_ROW_COUNT] {
        for (row, value) in rows.iter_mut().enumerate() {
            let bit = 1_u16 << row;
            if value.is_some_and(|value| value != 0 || row == ADDRESS_SPACE_ROW) {
                self.seen |= bit;
            }
            if self.seen & bit == 0 {
                *value = None;
            }
        }
        rows
    }
}

/// Every figure one sample reports, gathered by the encoder.
///
/// The frame counts, the textures, the in-flight bytes and both shader
/// counts belong to the reporting device; the largest free block, the guest
/// pages and the reuse pool are the process's, read from `d3d9.dll`.
pub struct HudSample {
    /// The last presented frame's user-pointer draws.
    pub draws_up: u32,
    /// The last presented frame's vertex-buffer draws.
    pub draws_vb: u32,
    /// The last presented frame's render passes.
    pub passes: u32,
    /// Textures alive on the device.
    pub textures: usize,
    /// The watch's last walk as [`largest_free_word`] stores it; zero before a walk.
    pub largest_free_word: u64,
    /// Bytes `d3d9.dll` holds in live page boxes, the parked ones of the reuse pool included.
    pub guest_page_bytes: u64,
    /// Vertex and index buffer copies held until the GPU has finished with them.
    pub in_flight_bytes: u64,
    /// Page boxes `d3d9.dll` keeps parked for reuse.
    pub reuse_pool_bytes: u64,
    /// Distinct fixed-function shader keys the game's draws used since device creation.
    pub fixed_function: usize,
    /// Distinct shader bytecode the game created since device creation, both stages.
    pub programs: usize,
}

/// The value of every row, indexed like [`HUD_ROW_KEYS`]; `None` leaves the row out.
///
/// The address-space row is left out until the watch has walked once, which
/// a 64-bit process never does; a walk that found less than 1 MiB free shows 0.
/// Every other row carries its value, 0 included; [`HudRowLatch`] decides
/// which rows the panel shows. The guest pages row leaves out the reuse pool,
/// whose parked boxes are live page boxes too, so the two rows add up to what
/// `d3d9.dll` holds, to within 1 MiB, since each rounds down to whole MiB.
#[must_use]
pub fn hud_rows(sample: &HudSample) -> [Option<u32>; HUD_ROW_COUNT] {
    let largest = sample.largest_free_word.checked_sub(1).map(|mib| {
        u32::try_from(mib)
            .unwrap_or(u32::MAX)
            .min(ADDRESS_SPACE_32_MIB)
    });
    // Two Relaxed loads of counters other threads move, not one snapshot: a
    // box parked between them can make the pool read larger than the live
    // total, which saturates to 0 rather than wrapping.
    let pages_in_use = sample
        .guest_page_bytes
        .saturating_sub(sample.reuse_pool_bytes);
    [
        Some(sample.draws_up),
        Some(sample.draws_vb),
        Some(sample.passes),
        Some(count(sample.textures)),
        largest,
        Some(mib(pages_in_use)),
        Some(mib(sample.in_flight_bytes)),
        Some(mib(sample.reuse_pool_bytes)),
        Some(count(sample.fixed_function)),
        Some(count(sample.programs)),
    ]
}

/// The word the address-space watch stores for a walk whose largest free block is `mib` MiB.
///
/// One more than the figure, saturating, so that 0 means only that no walk
/// has happened yet: a walk that found less than 1 MiB free, the state the
/// row exists to show, stores 1 and still shows, as 0.
#[must_use]
pub const fn largest_free_word(mib: u64) -> u64 {
    mib.saturating_add(1)
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
