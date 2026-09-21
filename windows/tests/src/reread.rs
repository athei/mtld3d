//! Follow-up reads for a GPU read-back that came back wrong.
//!
//! A test that renders, copies the result somewhere readable and reads it on
//! the CPU learns one thing from a wrong value: some stage between the pass and
//! the read lost the data. [`assert_or_reread`] asks the layer twice more
//! before it panics, so the failure names the stage. A read that is accepted
//! returns before either follow-up is built, so a passing test does no GPU
//! work it did not do before.

use core::fmt::Debug;
use std::{thread, time::Duration};

use crate::Harness;

/// How long the frame that carried the pass gets to finish before the plain re-read.
///
/// Nothing in D3D9 waits for one command buffer by name, and a wait that
/// trusts the queue's order is worth nothing in the one case the re-read is
/// for. The frames these tests submit run in a few milliseconds.
const SETTLE: Duration = Duration::from_millis(100);

/// One CPU observation of a GPU result, judged by the test that made it.
pub struct Reading {
    accepted: bool,
    shown: String,
}

impl Reading {
    /// `value` as the panic message shows it, integers in hex, and the test's verdict on it.
    #[must_use]
    pub fn new<T: Debug + ?Sized>(value: &T, accepted: bool) -> Self {
        Self {
            accepted,
            shown: format!("{value:08x?}"),
        }
    }

    /// A reading whose rendering the test wrote itself.
    #[must_use]
    pub const fn described(shown: String, accepted: bool) -> Self {
        Self { accepted, shown }
    }
}

/// Accept `first`, or read twice more and panic with all three readings.
///
/// `plain` reads the destination of the copy again and must not repeat the
/// copy. It has to reach the GPU: a second `GetRenderTargetData` does, since
/// every call flushes the open frame and blits the texture in a command buffer
/// it waits for, while a second `LockRect` of a texture the first lock read
/// back does not, because the first lock moved authority to the CPU staging.
/// `recopy` repeats the copy out of the same source without drawing, then
/// reads. Both run [`SETTLE`] after the first read at the earliest.
///
/// What the three readings say, with `A` the command buffer that carried the
/// copy and `B` the first read-back's own, committed after `A` on one queue:
///
/// - `plain` accepted: the destination holds the pass output now and did not
///   when `B` ran, so `B` ran ahead of `A`.
/// - `plain` rejected, `recopy` accepted: the destination never received the
///   pass output although the source of the copy holds it, so the first copy
///   ran ahead of the pass that feeds it, or was dropped. The two are one
///   outcome here.
/// - both rejected: the source of the copy does not hold the pass output. For
///   a `StretchRect` out of a multisampled surface that source is the
///   single-sampled resolve twin, because a repeated `StretchRect` finds no
///   pass of its own frame to hang a resolve on and copies the twin as the
///   earlier frame left it, so a lost pass and a lost resolve are one outcome.
///   For a RESZ write it is the multisampled depth attachment itself, which
///   the second transfer reads again without any draw.
///
/// # Panics
/// Panics when `first` was rejected, after both follow-up reads.
#[track_caller]
pub fn assert_or_reread(
    h: &Harness,
    context: &str,
    expected: &str,
    first: Reading,
    plain: impl FnOnce() -> Reading,
    recopy: impl FnOnce() -> Reading,
) {
    if first.accepted {
        return;
    }
    thread::sleep(SETTLE);
    let plain = plain();
    let recopy = recopy();
    let stage = match (plain.accepted, recopy.accepted) {
        (true, true) => {
            "the destination is right when read again with no new copy, so the first read-back \
             ran ahead of the command buffer that carried the copy"
        }
        (true, false) => {
            "the destination is right when read again with no new copy, so the first read-back \
             ran ahead of the command buffer that carried the copy; the repeated copy then \
             delivered a wrong value, which is a second fault"
        }
        (false, true) => {
            "the destination stayed wrong and a repeated copy is right, so the source holds the \
             pass output and the first copy ran ahead of the pass or was dropped"
        }
        (false, false) => {
            "wrong throughout, so the source of the copy does not hold the pass output: the \
             pass, or the resolve that belongs to it, was lost"
        }
    };
    panic!(
        "{context}\n  expected:            {expected}\n  first read:          {}\n  \
         re-read, no copy:    {}\n  after a second copy: {}\n  reading: {stage}\n  device: {}",
        first.shown,
        plain.shown,
        recopy.shown,
        h.adapter_description(),
    );
}
