//! The source-keyed shader-library indexes, and the draw-to-draw memo in front of them.
//!
//! A draw names each stage's library by its source record: the canonical
//! fixed-function key or the programmable identity, plus the pixel stage's
//! variant. Consecutive draws almost always name the same records, because a
//! snapshot re-sends a stage's record only when the application changed it,
//! so the memo answers a repeat from the record's address without hashing
//! or comparing the key again.
//!
//! A record's address names one immutable record only while the packet that
//! carries it replays: the packet retains its command storage until the
//! submission has read it, and the next packet may reuse the same bytes for
//! another record. [`StageLibraries::begin_packet`] therefore forgets the
//! memo before each packet, and every method that writes an index forgets it
//! as well, so the memo never answers with handles the index no longer
//! holds. It keeps built handles only: an unknown key or a recorded failure
//! goes to the index on every draw, as it did before the memo.

use mtld3d_core::{
    build_index::BuildIndex,
    draw_data::{FixedPsSource, FixedVsSource, ProgrammablePsSource, ProgrammableVsSource},
    dxso::{FfPsKey, FfVsKey, VariantKey, VsSamplerKinds},
    ids::ProgramId,
};
use mtld3d_shared::MetalHandle;
use rustc_hash::FxHashMap;

use crate::{
    draw::{PsSourceView, VsSourceView},
    encoder::StageLibHandles,
};

/// Per-draw shader-library lookup, keyed on the shader-identity struct.
///
/// `FxHash` + exact `Eq`, probed by borrow: no per-draw content hash, no
/// clone. One pair of maps per stage; VS keys exclude `variant` (variants
/// share one `MTLLibrary`) but carry the user clip plane count (a
/// programmable VS compiles one library per count), PS keys fold the variant
/// in. The Xxh3 `disk_key` is computed only on a miss here, to bridge the
/// encoder's `lib_cache` (warm-load) and address the on-disk cache. A key
/// whose build failed is recorded too: the same key yields the same source,
/// so its later draws are dropped on the probe instead of compiling again. A
/// library a worker is building is in the encoder's `pending_libs` until its
/// outcome lands here. A device reset forgets the failures (a Reset at
/// unchanged back-buffer dimensions never reaches it), shutdown forgets
/// everything. The indexes do not own the libraries: `lib_cache` does.
#[derive(Default)]
pub struct StageLibraries {
    ff_vs: BuildIndex<FfVsKey, StageLibHandles>,
    prog_vs: BuildIndex<(ProgramId, u16, u8, VsSamplerKinds), StageLibHandles>,
    ff_ps: FxHashMap<FfPsKey, BuildIndex<VariantKey, StageLibHandles>>,
    prog_ps: BuildIndex<(ProgramId, VariantKey), StageLibHandles>,
    memo: LibraryMemo,
}

impl StageLibraries {
    /// Both stages' built handles, or `None` when either stage is unknown or failed.
    ///
    /// A stage whose source record and variant are the ones the previous
    /// call answered for takes the memoised handles; any other goes to its
    /// index and, when built, replaces the memo's slot.
    #[inline]
    pub fn lookup_ready(
        &mut self,
        vs: VsSourceView<'_>,
        ps: PsSourceView<'_>,
        variant: VariantKey,
    ) -> Option<(StageLibHandles, StageLibHandles)> {
        let vs_record = vs_record(vs);
        let vs_handles = if self.memo.vs_record == vs_record {
            self.debug_assert_memo_vs(vs);
            self.memo.vs
        } else {
            self.memo_vs(vs, vs_record)?
        };
        let ps_record = ps_record(ps);
        let ps_handles = if self.memo.holds_ps(ps_record, &variant) {
            self.debug_assert_memo_ps(ps, variant);
            self.memo.ps
        } else {
            self.memo_ps(ps, variant, ps_record)?
        };
        Some((vs_handles, ps_handles))
    }

    /// Probe the VS source-key index without starting or waiting for a build.
    ///
    /// The borrowed outcome is one pointer: unknown, recorded failure, or
    /// ready handles.
    #[inline]
    pub fn lookup_vs(&self, source: VsSourceView<'_>) -> Option<&Option<StageLibHandles>> {
        match source {
            VsSourceView::FixedFunction(FixedVsSource { key, .. }) => self.ff_vs.lookup_entry(key),
            VsSourceView::Programmable(value) => {
                self.prog_vs.lookup_entry(&programmable_vs_key(value))
            }
        }
    }

    /// Probe the PS source-key and variant index without starting or waiting for a build.
    ///
    /// Preserves the recorded failure inside the borrowed outcome, so the
    /// full resolver distinguishes it from a key it still needs to build.
    #[inline]
    pub fn lookup_ps(
        &self,
        source: PsSourceView<'_>,
        variant: VariantKey,
    ) -> Option<&Option<StageLibHandles>> {
        match source {
            PsSourceView::FixedFunction(FixedPsSource { key, .. }) => self
                .ff_ps
                .get(key)
                .and_then(|variants| variants.lookup_entry(&variant)),
            PsSourceView::Programmable(ProgrammablePsSource { ps_id, .. }) => {
                self.prog_ps.lookup_entry(&(*ps_id, variant))
            }
        }
    }

    /// Record how building the VS library of a draw's source ended, `None` being a failure.
    pub fn record_vs(&mut self, source: VsSourceView<'_>, outcome: Option<StageLibHandles>) {
        match source {
            VsSourceView::FixedFunction(FixedVsSource { key, .. }) => {
                self.record_ff_vs(key.clone(), outcome);
            }
            VsSourceView::Programmable(value) => {
                self.record_programmable_vs(programmable_vs_key(value), outcome);
            }
        }
    }

    /// Record how building the PS library of a draw's source and variant ended.
    pub fn record_ps(
        &mut self,
        source: PsSourceView<'_>,
        variant: VariantKey,
        outcome: Option<StageLibHandles>,
    ) {
        match source {
            PsSourceView::FixedFunction(FixedPsSource { key, .. }) => {
                self.record_ff_ps(key.clone(), variant, outcome);
            }
            PsSourceView::Programmable(ProgrammablePsSource { ps_id, .. }) => {
                self.record_programmable_ps(*ps_id, variant, outcome);
            }
        }
    }

    /// Record a fixed-function VS build outcome under its owned key.
    pub fn record_ff_vs(&mut self, key: FfVsKey, outcome: Option<StageLibHandles>) {
        self.memo.forget();
        self.ff_vs.record(key, outcome);
    }

    /// Record a programmable VS build outcome under its program, inputs, clip planes and samplers.
    pub fn record_programmable_vs(
        &mut self,
        key: (ProgramId, u16, u8, VsSamplerKinds),
        outcome: Option<StageLibHandles>,
    ) {
        self.memo.forget();
        self.prog_vs.record(key, outcome);
    }

    /// Record a fixed-function PS build outcome under its owned key and variant.
    pub fn record_ff_ps(
        &mut self,
        key: FfPsKey,
        variant: VariantKey,
        outcome: Option<StageLibHandles>,
    ) {
        self.memo.forget();
        self.ff_ps.entry(key).or_default().record(variant, outcome);
    }

    /// Record a programmable PS build outcome under its program and variant.
    pub fn record_programmable_ps(
        &mut self,
        ps_id: ProgramId,
        variant: VariantKey,
        outcome: Option<StageLibHandles>,
    ) {
        self.memo.forget();
        self.prog_ps.record((ps_id, variant), outcome);
    }

    /// Forget the failed keys so each is built once more; built libraries stay.
    pub fn forget_failures(&mut self) {
        self.memo.forget();
        self.ff_vs.forget_failures();
        self.prog_vs.forget_failures();
        for variants in self.ff_ps.values_mut() {
            variants.forget_failures();
        }
        self.prog_ps.forget_failures();
    }

    /// Forget every key, for a teardown that destroys the libraries they name.
    pub fn clear(&mut self) {
        self.memo.forget();
        self.ff_vs.clear();
        self.prog_vs.clear();
        self.ff_ps.clear();
        self.prog_ps.clear();
    }

    /// Forget the memo before a packet whose records may reuse the previous packet's addresses.
    pub const fn begin_packet(&mut self) {
        self.memo.forget();
    }

    /// Look up a VS source whose record the memo does not hold, and memoise it when built.
    fn memo_vs(&mut self, vs: VsSourceView<'_>, record: usize) -> Option<StageLibHandles> {
        let handles = (*self.lookup_vs(vs)?)?;
        self.memo.vs_record = record;
        self.memo.vs = handles;
        Some(handles)
    }

    /// Look up a PS source and variant the memo does not hold, and memoise them when built.
    fn memo_ps(
        &mut self,
        ps: PsSourceView<'_>,
        variant: VariantKey,
        record: usize,
    ) -> Option<StageLibHandles> {
        let handles = (*self.lookup_ps(ps, variant)?)?;
        self.memo.ps_record = record;
        self.memo.ps_variant = variant;
        self.memo.ps = handles;
        Some(handles)
    }

    /// Check, in builds with debug assertions, that a memoised VS answer is the index's own.
    #[inline]
    fn debug_assert_memo_vs(&self, vs: VsSourceView<'_>) {
        debug_assert!(
            matches!(self.lookup_vs(vs), Some(Some(handles)) if same_handles(handles, &self.memo.vs)),
            "the VS library memo answers what the index holds for its record"
        );
    }

    /// Check, in builds with debug assertions, that a memoised PS answer is the index's own.
    #[inline]
    fn debug_assert_memo_ps(&self, ps: PsSourceView<'_>, variant: VariantKey) {
        debug_assert!(
            matches!(self.lookup_ps(ps, variant), Some(Some(handles)) if same_handles(handles, &self.memo.ps)),
            "the PS library memo answers what the index holds for its record and variant"
        );
    }
}

/// The built libraries of the previous draw, by the addresses of its source records.
///
/// A record address of zero is an empty slot: no source record is at
/// address zero.
struct LibraryMemo {
    vs_record: usize,
    vs: StageLibHandles,
    ps_record: usize,
    ps_variant: VariantKey,
    ps: StageLibHandles,
}

impl LibraryMemo {
    const EMPTY_HANDLES: StageLibHandles = StageLibHandles {
        library: MetalHandle::NULL,
        func: MetalHandle::NULL,
    };

    /// Whether the memo holds the pixel stage of this record and variant.
    ///
    /// The variant is compared as [`variant_words`]: the derived field-wise
    /// compare, inlined into the draw, assembled the caller's key byte by
    /// byte from registers.
    #[inline]
    fn holds_ps(&self, record: usize, variant: &VariantKey) -> bool {
        self.ps_record == record && variant_words(&self.ps_variant) == variant_words(variant)
    }

    const fn forget(&mut self) {
        self.vs_record = 0;
        self.ps_record = 0;
    }
}

impl Default for LibraryMemo {
    fn default() -> Self {
        Self {
            vs_record: 0,
            vs: Self::EMPTY_HANDLES,
            ps_record: 0,
            ps_variant: VariantKey::default(),
            ps: Self::EMPTY_HANDLES,
        }
    }
}

/// The identity of a VS source record: its address, with the low bit set for a fixed-function one.
///
/// Source records are 8-aligned, so the tag bit never collides with an
/// address bit.
fn vs_record(source: VsSourceView<'_>) -> usize {
    match source {
        VsSourceView::Programmable(value) => core::ptr::from_ref(value).addr(),
        VsSourceView::FixedFunction(value) => core::ptr::from_ref(value).addr() | 1,
    }
}

/// The identity of a PS source record, tagged like [`vs_record`].
fn ps_record(source: PsSourceView<'_>) -> usize {
    match source {
        PsSourceView::Programmable(value) => core::ptr::from_ref(value).addr(),
        PsSourceView::FixedFunction(value) => core::ptr::from_ref(value).addr() | 1,
    }
}

/// The 22 bytes of a variant key as three words; equal words mean equal keys.
///
/// Field order and widths follow the `repr(C)` layout, so the backend reads
/// each word with one load. The destructuring names every field, so a field
/// added to the key fails to compile here instead of escaping the compare.
fn variant_words(variant: &VariantKey) -> [u64; 3] {
    let VariantKey {
        alpha_func,
        fog_mode,
        fog_table_mode,
        reserved,
        depth_sampler_mask,
        depth_fetch_mask,
        fetch4_mask,
        fetch4_alpha_mask,
        raw_depth_red_mask,
        volume_sampler_mask,
        cube_sampler_mask,
        tt_projected_mask,
        color_out_mask,
        sample_mask,
        flags,
    } = variant;
    [
        u64::from(*alpha_func)
            | u64::from(*fog_mode) << 8
            | u64::from(*fog_table_mode) << 16
            | u64::from(*reserved) << 24
            | u64::from(*depth_sampler_mask) << 32
            | u64::from(*depth_fetch_mask) << 48,
        u64::from(*fetch4_mask)
            | u64::from(*fetch4_alpha_mask) << 16
            | u64::from(*raw_depth_red_mask) << 32
            | u64::from(*volume_sampler_mask) << 48,
        u64::from(*cube_sampler_mask)
            | u64::from(*tt_projected_mask) << 16
            | u64::from(*color_out_mask) << 24
            | u64::from(*sample_mask) << 32
            | u64::from(flags.bits()) << 40,
    ]
}

/// The programmable VS index key: program, provided inputs, clip planes and sampler kinds.
const fn programmable_vs_key(value: &ProgrammableVsSource) -> (ProgramId, u16, u8, VsSamplerKinds) {
    (
        value.vs_id,
        value.provided_input_mask,
        value.clip_plane_count,
        value.sampler_kinds,
    )
}

const fn same_handles(left: &StageLibHandles, right: &StageLibHandles) -> bool {
    left.library.raw() == right.library.raw() && left.func.raw() == right.func.raw()
}

#[cfg(test)]
mod tests;
