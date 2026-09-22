//! The gamma ramp a layer's present pass applies.
//!
//! D3D9's gamma ramp is the display's transfer function, so nothing upstream
//! of the drawable may carry it: the game's back buffer, every render target
//! and every readback stay exactly as the game drew them. The only place it
//! belongs is the present pass, as a table its fragment stage looks each
//! channel up in.
//!
//! The table is 256 entries of four `u16` lanes, which the PE side has already
//! validated and laid out ([`mtld3d_core::gamma`]); this module only keeps it
//! for the layer it belongs to and hands it to an encoder. At 2 KiB it fits
//! `setFragmentBytes:`, which copies it into the command buffer, so a layer
//! needs no Metal resource of its own and nothing has to outlive a present.
//!
//! One writer and one reader, both the encoder thread: the ramp arrives on the
//! frame the guest's `SetGammaRamp` rode out on, ahead of that frame's own
//! present. The mutex is what makes the map safely shared with the teardown
//! path, and it is taken per present only while a ramp is active.

use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};

use mtld3d_shared::SetGammaRampParams;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLRenderCommandEncoder;
use rustc_hash::FxHashMap;

use crate::{LOG_TARGET, metal::macdrv::attachment};

/// `u16` lanes of one table: 256 entries of R, G, B and a one.
///
/// Fixed rather than taken from the payload, so a short or long write is
/// rejected at the boundary instead of reaching a fragment stage that indexes
/// 256 entries unconditionally.
const LANES: usize = 256 * 4;

/// Fragment buffer index the present pipelines read the table from.
///
/// Slot 0 is the `BT.2446` uniform block; the gamma pipelines add this one.
pub const FRAGMENT_BUFFER_INDEX: usize = 1;

/// One layer's table and the revision that says when it last changed.
struct Table {
    /// Bumped by every ramp the layer takes, starting at 1.
    ///
    /// The software cursor renders its sprite once and caches the image, so it
    /// needs to know that the table it rendered against is not the current
    /// one. The frame pass rebuilds every present and ignores this.
    revision: u64,
    entries: [u16; LANES],
}

/// The live tables, keyed by the raw `CAMetalLayer*` of the layer they apply to.
///
/// An entry exists exactly while that layer's attachment record reports
/// `gamma_active`, and is removed when the ramp goes back to identity or the
/// layer is detached. At most a handful of entries, one per attached layer.
static TABLES: LazyLock<Mutex<FxHashMap<usize, Table>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

fn tables() -> MutexGuard<'static, FxHashMap<usize, Table>> {
    TABLES.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Take a validated ramp for a layer, or remove the one it has.
///
/// `entries` is `None` when the guest went back to an identity ramp or left
/// fullscreen: the table is removed and the layer's presents go back to the
/// route they would take with no ramp at all. Returns `false` for a layer no
/// attachment record names, which is a device that never attached one.
pub fn set_gamma_ramp(params: &SetGammaRampParams, entries: Option<&[u16]>) -> bool {
    let layer = usize::try_from(params.layer_handle.raw())
        .expect("a 64-bit host addresses every layer pointer");
    let Some(att) = attachment::find_by_layer(layer) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: SetGammaRamp for layer {layer:#x} with no attachment record; \
             the presented frame is unchanged",
        );
        return false;
    };
    if let Some(entries) = entries {
        {
            let mut tables = tables();
            let revision = tables.get(&layer).map_or(1, |table| table.revision + 1);
            let mut kept = Table {
                revision,
                entries: [0u16; LANES],
            };
            kept.entries.copy_from_slice(entries);
            tables.insert(layer, kept);
        }
        // Only once the entries are in place: the present path reads the flag
        // first and binds the table second.
        att.set_gamma_active(true);
    } else {
        // The other order for the same reason.
        att.set_gamma_active(false);
        tables().remove(&layer);
    }
    true
}

/// Which table this layer carries, `0` for none.
///
/// A caller that caches what it rendered against compares this: a different
/// number means the ramp moved and the cached image is stale.
#[must_use]
pub fn revision(layer: usize) -> u64 {
    tables().get(&layer).map_or(0, |table| table.revision)
}

/// Hand the layer's table to a fragment stage, and say whether one was bound.
///
/// The caller has already read `gamma_active` to pick a gamma pipeline, and
/// that flag is only set while the table exists, so `false` here means the
/// pass must not run a gamma pipeline.
pub fn bind(enc: &ProtocolObject<dyn MTLRenderCommandEncoder>, layer: usize) -> bool {
    let tables = tables();
    let Some(table) = tables.get(&layer) else {
        return false;
    };
    // SAFETY: objc2 typed binding. The pointer is the live entries', non-null
    // and valid for `size_of::<[u16; LANES]>()` bytes for the call, which is
    // all `setFragmentBytes:` needs: it copies into the command buffer.
    unsafe {
        enc.setFragmentBytes_length_atIndex(
            core::ptr::NonNull::from(&table.entries).cast(),
            core::mem::size_of_val(&table.entries),
            FRAGMENT_BUFFER_INDEX,
        );
    }
    // Released here rather than at the end of the function: the lock spans the
    // copy `setFragmentBytes:` makes and nothing more.
    drop(tables);
    true
}

/// Lanes a payload must carry, for the boundary check.
#[must_use]
pub const fn expected_lanes() -> usize {
    LANES
}

/// The layer of a retired attachment keeps no table.
///
/// Called from the detach path, the one place a record leaves the registry, so
/// a window the game destroyed does not hold 2 KiB until the process ends.
pub fn detach(att: &attachment::Attachment) {
    tables().remove(&att.layer());
}
