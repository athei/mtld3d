//! Which devices share a window's subclass, and the procedure it replaced.
//!
//! D3D9 allows several devices on one window, and each device hooks the window
//! procedure of the window it presents into. Only the first device on a window
//! installs the hook; a later one joins it, because replacing the procedure a
//! second time would store the hook itself as the procedure to forward to, and
//! every message would then call the hook again until the stack overflows.
//! Messages go to the first device still registered, and the window gets its
//! own procedure back when the last device leaves, in whichever order they go.

use rustc_hash::FxHashMap;

/// The subclassed windows, by `HWND`, each with the devices registered on it.
///
/// Windows and devices are addresses widened to `usize`: the map only compares
/// them, and the caller owns what they point at.
#[derive(Default)]
pub struct WindowSubclasses {
    windows: FxHashMap<usize, Subclass>,
}

/// One window's hook: the procedure it replaced and the devices sharing it.
struct Subclass {
    /// The procedure the hook forwards to, which the last device puts back.
    original: usize,
    /// Registered devices, oldest first; the first one receives the messages.
    devices: Vec<usize>,
    /// Calls of the hook on this window that have not returned yet.
    ///
    /// A message the game's procedure handles can send the window further
    /// messages before it returns, each of which runs the hook again inside
    /// the first call; zero means no message is being dispatched.
    depth: u32,
    /// The device to tell once the outermost hook call on this window has returned.
    held_until_dispatch_ends: Option<usize>,
}

impl WindowSubclasses {
    /// Register `device` on `window`, installing the hook when no device holds the window yet.
    ///
    /// `install` runs only for the first device on the window. It replaces the
    /// window procedure and returns the one it replaced, which the hook then
    /// forwards to. Returns `true` when it ran. A device already registered
    /// on the window is left where it is.
    pub fn register(
        &mut self,
        window: usize,
        device: usize,
        install: impl FnOnce() -> usize,
    ) -> bool {
        if let Some(subclass) = self.windows.get_mut(&window) {
            if !subclass.devices.contains(&device) {
                subclass.devices.push(device);
            }
            return false;
        }
        let original = install();
        self.windows.insert(
            window,
            Subclass {
                original,
                devices: vec![device],
                depth: 0,
                held_until_dispatch_ends: None,
            },
        );
        true
    }

    /// Unregister `device` from `window`, restoring the procedure when it was the last device.
    ///
    /// `restore` runs only when no device is left on the window, with the
    /// procedure the first device's hook replaced. Returns `true` when it ran.
    /// A device not registered on the window changes nothing.
    pub fn unregister(
        &mut self,
        window: usize,
        device: usize,
        restore: impl FnOnce(usize),
    ) -> bool {
        let Some(subclass) = self.windows.get_mut(&window) else {
            return false;
        };
        subclass.devices.retain(|&registered| registered != device);
        if !subclass.devices.is_empty() {
            return false;
        }
        let original = subclass.original;
        self.windows.remove(&window);
        restore(original);
        true
    }

    /// The device a message on `window` goes to, and the procedure the hook forwards to.
    ///
    /// `None` for a window no device holds.
    #[must_use]
    pub fn route(&self, window: usize) -> Option<(usize, usize)> {
        let subclass = self.windows.get(&window)?;
        let device = *subclass.devices.first()?;
        Some((device, subclass.original))
    }

    /// [`Self::route`] for a hook call that starts on `window`, counted until [`Self::leave`].
    ///
    /// `None` for a window no device holds, and then nothing is counted and
    /// the call has no [`Self::leave`] to make.
    pub fn enter(&mut self, window: usize) -> Option<(usize, usize)> {
        let route = self.route(window)?;
        let subclass = self.windows.get_mut(&window)?;
        subclass.depth = subclass.depth.saturating_add(1);
        Some(route)
    }

    /// Have the running outermost hook call on `window` name `device` when it returns.
    ///
    /// A window no device holds keeps nothing. A later hold in the same
    /// dispatch replaces an earlier one.
    pub fn hold_until_dispatch_ends(&mut self, window: usize, device: usize) {
        if let Some(subclass) = self.windows.get_mut(&window) {
            subclass.held_until_dispatch_ends = Some(device);
        }
    }

    /// Count a hook call on `window` that [`Self::enter`] counted as returned.
    ///
    /// When it was the outermost one, returns the device the dispatch was
    /// held for, if that device is still registered on the window. `None`
    /// for a nested call, a dispatch nothing held, and a window no device
    /// holds any more: a window whose last device left drops its count with
    /// its registration.
    pub fn leave(&mut self, window: usize) -> Option<usize> {
        let subclass = self.windows.get_mut(&window)?;
        subclass.depth = subclass.depth.saturating_sub(1);
        if subclass.depth > 0 {
            return None;
        }
        let device = subclass.held_until_dispatch_ends.take()?;
        subclass.devices.contains(&device).then_some(device)
    }

    /// Every device registered on `window`, oldest first; empty for a window no device holds.
    ///
    /// For what concerns each device on the window rather than the one that
    /// receives its messages.
    #[must_use]
    pub fn devices_on(&self, window: usize) -> &[usize] {
        self.windows
            .get(&window)
            .map_or(&[], |subclass| subclass.devices.as_slice())
    }
}

#[cfg(test)]
mod tests;
