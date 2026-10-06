//! Page boxes charged to a live-bytes gauge for as long as their holder keeps them.
//!
//! The address-space watch walks the texture registry for texture staging and
//! reads the vertex/index backing gauge, but two more holders keep page boxes
//! in the 32-bit address space that neither sees, so their pages would be
//! counted only in the page-box total. [`HeldPages`] charges a box to its
//! holder's gauge while held:
//!
//! - [`PageHolder::Surface`]: the backing of a system-memory or scratch
//!   offscreen plain surface, the staging of a lockable render target, and the
//!   read-back page a back-buffer `LockRect` or `GetDC` holds until released.
//! - [`PageHolder::EncoderLease`]: a renamed vertex/index backing or an upload
//!   snapshot the PE side keeps until the encoder acknowledges its last read.

use core::{
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::page_box::PageBox;

/// Padded bytes of every live [`HeldPages`] a surface holds.
///
/// Process-wide, because the resource it measures is: the watch reads it
/// against the one 32-bit address space and the process-wide page-box
/// total, and a surface can outlive the device that created it, which would
/// leave a per-device count with no owner to give the bytes back to.
static SURFACE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Padded bytes of every live [`HeldPages`] an encoder lease holds.
///
/// Process-wide on the same argument as [`SURFACE_BYTES`]: the 32-bit
/// address space is one resource, and a lease retires on whichever thread
/// sees the encoder's acknowledgment, after the frame that recorded it, so
/// it has no device record to reach at that point.
static ENCODER_LEASE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Who holds a [`HeldPages`], and so which gauge it is charged to.
pub enum PageHolder {
    /// A surface's system-memory backing, lockable staging or read-back page.
    Surface,
    /// A retired backing or upload snapshot awaiting the encoder's last read.
    EncoderLease,
}

impl PageHolder {
    const fn gauge(&self) -> &'static AtomicU64 {
        match self {
            Self::Surface => &SURFACE_BYTES,
            Self::EncoderLease => &ENCODER_LEASE_BYTES,
        }
    }
}

/// Padded bytes the surfaces keep in page boxes now.
#[must_use]
pub fn live_surface_bytes() -> u64 {
    SURFACE_BYTES.load(Ordering::Relaxed)
}

/// Padded bytes the encoder leases keep in page boxes now.
#[must_use]
pub fn live_encoder_lease_bytes() -> u64 {
    ENCODER_LEASE_BYTES.load(Ordering::Relaxed)
}

/// A page box charged to its holder's gauge until it drops or is handed back.
pub struct HeldPages {
    /// The box; `None` only inside [`Self::into_page`], which consumes `self`.
    page: Option<PageBox>,
    /// The bytes charged at construction, returned exactly on release.
    ///
    /// Kept apart from the box because `DerefMut` lets a holder replace the
    /// box with one of another length, and returning that length would leave
    /// the gauge off by the difference.
    charged: u64,
    holder: PageHolder,
}

impl HeldPages {
    /// Take `page` for `holder` and charge its padded length to that holder's gauge.
    #[must_use]
    pub fn new(page: PageBox, holder: PageHolder) -> Self {
        let charged = page.len() as u64;
        holder.gauge().fetch_add(charged, Ordering::Relaxed);
        Self {
            page: Some(page),
            charged,
            holder,
        }
    }

    /// Give the box back, returning its charge to the gauge.
    ///
    /// # Panics
    ///
    /// Never: the box is present until this call takes it.
    #[must_use]
    pub fn into_page(mut self) -> PageBox {
        self.page
            .take()
            .expect("a held box is present until into_page")
    }

    const fn page(&self) -> &PageBox {
        self.page
            .as_ref()
            .expect("a held box is present until into_page")
    }
}

impl Deref for HeldPages {
    type Target = PageBox;

    fn deref(&self) -> &PageBox {
        self.page()
    }
}

impl DerefMut for HeldPages {
    fn deref_mut(&mut self) -> &mut PageBox {
        self.page
            .as_mut()
            .expect("a held box is present until into_page")
    }
}

impl Drop for HeldPages {
    fn drop(&mut self) {
        self.holder
            .gauge()
            .fetch_sub(self.charged, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests;
