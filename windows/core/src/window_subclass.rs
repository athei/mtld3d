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
}

#[cfg(test)]
mod tests;
