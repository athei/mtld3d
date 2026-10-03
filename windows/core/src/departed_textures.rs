//! Textures that moved to another device, waiting for the device they left to drop them.
//!
//! A `D3DPOOL_MANAGED` texture follows the device it is used with. The device
//! it leaves has Metal storage for it in its encoder's cache, and only that
//! device can queue the destroy: the op goes into its current frame, which
//! its own API lock guards. The move runs under the adopting device's lock,
//! and taking the other device's lock there as well would let two devices
//! that trade textures in opposite directions deadlock. So the move files the
//! texture's id here, and the device it left drains the list into its own
//! frame the next time it hands one to its encoder, under its own lock.
//!
//! A texture that moves back before that drain takes its id off the list
//! again: the storage is its own once more, and a destroy after the move
//! back would take it away from the draws that bind it.
//!
//! Pure bookkeeping: no Metal handles, no D3D9 objects, so the whole contract
//! is host-testable.

use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

use crate::ids::TextureId;

/// One device's departed textures.
#[derive(Default)]
pub struct DepartedTextures {
    /// Whether `ids` may hold anything, so the per-frame drain skips the lock.
    pending: AtomicBool,
    /// Ids filed by moves and not yet drained, in filing order.
    ids: Mutex<Vec<TextureId>>,
}

impl DepartedTextures {
    /// File a texture that just moved to another device.
    ///
    /// # Panics
    /// When a thread panicked while holding the list's lock.
    pub fn note(&self, id: TextureId) {
        let mut ids = self.ids.lock().expect("departed textures mutex poisoned");
        if !ids.contains(&id) {
            ids.push(id);
        }
        // Set while the lock is held, so no cancel or drain can see the id in
        // the list with the flag still clear and skip it.
        self.pending.store(true, Ordering::Release);
        drop(ids);
    }

    /// Take back a texture that moved back before its departure was drained.
    ///
    /// # Panics
    /// When a thread panicked while holding the list's lock.
    pub fn cancel(&self, id: TextureId) {
        if !self.pending.load(Ordering::Acquire) {
            return;
        }
        let mut ids = self.ids.lock().expect("departed textures mutex poisoned");
        ids.retain(|&filed| filed != id);
        if ids.is_empty() {
            self.pending.store(false, Ordering::Release);
        }
    }

    /// Every texture filed since the last drain, oldest first.
    ///
    /// Returns an empty list without taking the lock when nothing was filed.
    ///
    /// # Panics
    /// When a thread panicked while holding the list's lock.
    pub fn take(&self) -> Vec<TextureId> {
        if !self.pending.load(Ordering::Acquire) {
            return Vec::new();
        }
        let mut ids = self.ids.lock().expect("departed textures mutex poisoned");
        self.pending.store(false, Ordering::Release);
        core::mem::take(&mut *ids)
    }
}

#[cfg(test)]
mod tests;
