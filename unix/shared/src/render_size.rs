//! The sizes a scaled display mode decides, computed the same way on both sides.
//!
//! In a scaled display mode a screen's backing store has more pixels than
//! its panel, and the display scales every composited frame down to the
//! panel. Present caps the drawable at the panel's share of the layer
//! ([`panel_drawable_size`]); the API side rasterizes a back buffer larger
//! than the panel at the same share (`render.panelFit`), so the frame fits
//! the capped drawable and present is a 1:1 copy. Both round through
//! [`scale_rounded`], which is what makes the two sizes agree to the pixel.

const _: () = {
    assert!(size_of::<RenderScaleWire>() == 16);
    assert!(align_of::<RenderScaleWire>() == 4);
};

/// A render scale on the wire: render pixels per logical pixel, an exact ratio per axis.
///
/// The API side builds it from `mtld3d_core::render_scale::RenderScale`,
/// which keeps each ratio non-zero and at most one; a reader checks that
/// again before it scales anything with it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderScaleWire {
    pub width_num: u32,
    pub width_den: u32,
    pub height_num: u32,
    pub height_den: u32,
}

impl RenderScaleWire {
    /// Render and logical resolutions are the same in both axes.
    pub const IDENTITY: Self = Self {
        width_num: 1,
        width_den: 1,
        height_num: 1,
        height_den: 1,
    };
}

/// The panel and backing pixel sizes of a window's screen, `None` for a size not known.
///
/// The unix side publishes them into a word the device owns
/// (`AttachMetalLayerParams::screen_sizes_ptr`) at attach and whenever the
/// window's screen changes; the device reads the word when it sizes its back
/// buffer again. [`Self::pack`] folds both into that word.
#[derive(Debug, PartialEq, Eq)]
pub struct ScreenSizes {
    /// The native pixel size of the screen's panel.
    pub panel: Option<(u32, u32)>,
    /// The screen's frame in the window layer's pixels.
    pub backing: Option<(u32, u32)>,
}

impl ScreenSizes {
    /// Nothing known: the word reads zero.
    pub const UNKNOWN: Self = Self {
        panel: None,
        backing: None,
    };

    /// Fold both sizes into one word, 16 bits per dimension, the panel high.
    ///
    /// A size that does not fit 16 bits, or has a zero dimension, packs as
    /// unknown, so a word of zero is nothing published.
    #[must_use]
    pub fn pack(&self) -> u64 {
        let half = |size: Option<(u32, u32)>| {
            size.and_then(|(width, height)| {
                let width = u16::try_from(width).ok().filter(|&w| w != 0)?;
                let height = u16::try_from(height).ok().filter(|&h| h != 0)?;
                Some((u64::from(width) << 16) | u64::from(height))
            })
            .unwrap_or(0)
        };
        (half(self.panel) << 32) | half(self.backing)
    }

    /// Read back what [`Self::pack`] wrote.
    #[must_use]
    pub fn unpack(bits: u64) -> Self {
        let half = |half: u64| {
            let [_, _, _, _, w1, w0, h1, h0] = half.to_be_bytes();
            let width = u32::from(u16::from_be_bytes([w1, w0]));
            let height = u32::from(u16::from_be_bytes([h1, h0]));
            (width != 0 && height != 0).then_some((width, height))
        };
        Self {
            panel: half(bits >> 32),
            backing: half(bits & 0xffff_ffff),
        }
    }
}

/// `size × num / den`, rounded to nearest with halves up, exact in integers.
///
/// `den` is non-zero. The product of two `u32`s fits a `u64`, and so does
/// twice a remainder below a `u32` denominator, so the result is exact for
/// every input: an exact ratio of a whole screen lands on the panel size with
/// no float error.
///
/// # Panics
///
/// When `den` is zero.
#[must_use]
pub fn scale_rounded(size: u32, num: u32, den: u32) -> u64 {
    let product = u64::from(size) * u64::from(num);
    let den = u64::from(den);
    let (quotient, remainder) = (product / den, product % den);
    quotient + u64::from(remainder * 2 >= den)
}

/// Whether the display scales the backing store down to the panel, in either axis.
///
/// That is when the drawable can be capped below the layer's size and the
/// compositor magnifies it back, so the layer's magnification filter is
/// linear then and nearest otherwise, and when a back buffer larger than the
/// panel rasterizes at the panel's pixels.
#[must_use]
pub const fn scaled_down_to_panel(backing: Option<(u32, u32)>, panel: Option<(u32, u32)>) -> bool {
    matches!(
        (backing, panel),
        (Some((backing_w, backing_h)), Some((panel_w, panel_h)))
            if backing_w > panel_w || backing_h > panel_h
    )
}

/// The drawable size for a layer whose own size is `natural`, capped at the panel's pixels.
///
/// The cap scales `natural` per axis by panel over backing, so a layer
/// covering the whole screen comes out at exactly the panel size, and the
/// compositor takes the rest of the way. It applies only when it shrinks the
/// drawable and the source (the frame being presented) still fits in the
/// capped size: a source that does not would lose pixels to the cap that the
/// panel can show. Any unknown input leaves `natural`, and nothing is ever
/// enlarged.
#[must_use]
pub fn panel_drawable_size(
    natural: (u32, u32),
    backing: Option<(u32, u32)>,
    panel: Option<(u32, u32)>,
    source: Option<(u32, u32)>,
) -> (u32, u32) {
    let (Some((backing_w, backing_h)), Some((panel_w, panel_h)), Some((source_w, source_h))) =
        (backing, panel, source)
    else {
        return natural;
    };
    if backing_w == 0 || backing_h == 0 {
        return natural;
    }
    let scale = |size: u32, num: u32, den: u32| {
        // Clamped to `size`, so the conversion cannot fail.
        u32::try_from(scale_rounded(size, num, den).min(u64::from(size)))
            .unwrap_or(size)
            .max(1)
    };
    let (natural_w, natural_h) = natural;
    let capped_w = scale(natural_w, panel_w, backing_w);
    let capped_h = scale(natural_h, panel_h, backing_h);
    let shrinks = natural_w > capped_w || natural_h > capped_h;
    let source_fits = source_w <= capped_w && source_h <= capped_h;
    if shrinks && source_fits {
        (capped_w, capped_h)
    } else {
        natural
    }
}

#[cfg(test)]
mod tests;
