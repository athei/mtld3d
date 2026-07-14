// Fire a `warn!` at most once per call site. Backed by a `static
// AtomicBool` declared inside the macro expansion, so each textual
// invocation gets its own latch. Intended for stubs, unimplemented
// branches, default match arms, and any silent early-return that would
// otherwise hide a bug behind a clean `RUST_LOG=info` log.
//
// Takes the log target explicitly so the macro can be shared across
// crates. Callers pass `target: LOG_TARGET` (whichever const is in
// scope) so `RUST_LOG=<target>=...` filtering keeps working.
#[macro_export]
macro_rules! log_once_warn {
    (target: $target:expr, $($arg:tt)*) => {{
        static FIRED: ::core::sync::atomic::AtomicBool =
            ::core::sync::atomic::AtomicBool::new(false);
        if !FIRED.swap(true, ::core::sync::atomic::Ordering::Relaxed) {
            ::log::warn!(target: $target, $($arg)*);
        }
    }};
}

// Like `log_once_warn!`, but the latch is keyed on a caller-supplied `u64`
// so each distinct input produces its own one-shot warn. Use this where
// a single format string legitimately receives many distinct values we
// want enumerated in the log — e.g. `D3DTS_{state} not honoured`,
// `usage={u} usage_index={i} has no attribute register`, `unhandled op={op}`.
// Per-call-site `log_once_warn!` would collapse all of those into one line
// and hide the tail from the warn surface.
//
// The first-seen check releases the lock *before* `log::warn!` runs, so a
// logger that re-enters the macro cannot deadlock.
#[macro_export]
macro_rules! log_once_warn_by {
    (target: $target:expr, key: $key:expr, $($arg:tt)*) => {{
        static SEEN: ::std::sync::LazyLock<
            ::std::sync::Mutex<::std::collections::BTreeSet<u64>>,
        > = ::std::sync::LazyLock::new(|| {
            ::std::sync::Mutex::new(::std::collections::BTreeSet::new())
        });
        let k: u64 = $key;
        let first = {
            let mut s = SEEN.lock().expect("log_once_warn_by set poisoned");
            s.insert(k)
        };
        if first {
            ::log::warn!(target: $target, $($arg)*);
        }
    }};
}

// Like `log_once_warn_by!`, but fires at `trace!` level and wraps the
// mutex + btree-insert in a `log_enabled!(Trace)` short-circuit. At
// the default `RUST_LOG=info` level the entire body compiles down to
// a single relaxed atomic load — no lock, no hashing, no allocation,
// the `$key` expression is not even evaluated. The user pays the
// per-call mutex only after explicitly opting into
// `RUST_LOG=...=trace`.
//
// Design principle: trace is the only log level allowed to add
// per-call overhead. Any per-key dedup macro must gate on
// `log_enabled!` for its level.
//
// Use for silent-drop paths that can fire per draw per frame in a
// degraded build, where we want per-identity fan-out (one trace line
// per distinct failing shader/buffer) but cannot afford per-call
// cost at default log level. A paired `log_once_warn!` at the same
// site still provides the default-level existence signal.
#[macro_export]
macro_rules! log_once_trace_by {
    (target: $target:expr, key: $key:expr, $($arg:tt)*) => {{
        if ::log::log_enabled!(target: $target, ::log::Level::Trace) {
            static SEEN: ::std::sync::LazyLock<
                ::std::sync::Mutex<::std::collections::BTreeSet<u64>>,
            > = ::std::sync::LazyLock::new(|| {
                ::std::sync::Mutex::new(::std::collections::BTreeSet::new())
            });
            let k: u64 = $key;
            let first = {
                let mut s = SEEN.lock().expect("log_once_trace_by set poisoned");
                s.insert(k)
            };
            if first {
                ::log::trace!(target: $target, $($arg)*);
            }
        }
    }};
}

// Mirror of `log_once_trace_by!` at `log::Level::Debug`. Same `log_enabled!`
// short-circuit: at the default `RUST_LOG=info` level the body compiles down
// to a single relaxed atomic load, and `$key` is not evaluated. Use for
// per-identity enumerations that are too chatty for `info` but cheaper than
// trace — e.g. one-shot D3D9 spec probes (`CheckDeviceFormat → OK`) that
// games sweep at startup.
#[macro_export]
macro_rules! log_once_debug_by {
    (target: $target:expr, key: $key:expr, $($arg:tt)*) => {{
        if ::log::log_enabled!(target: $target, ::log::Level::Debug) {
            static SEEN: ::std::sync::LazyLock<
                ::std::sync::Mutex<::std::collections::BTreeSet<u64>>,
            > = ::std::sync::LazyLock::new(|| {
                ::std::sync::Mutex::new(::std::collections::BTreeSet::new())
            });
            let k: u64 = $key;
            let first = {
                let mut s = SEEN.lock().expect("log_once_debug_by set poisoned");
                s.insert(k)
            };
            if first {
                ::log::debug!(target: $target, $($arg)*);
            }
        }
    }};
}

// Mirror of `log_once_warn!` at `log::Level::Info`. Use for D3D9 calls
// where the no-op we implement IS the complete and correct Metal-world
// behaviour — e.g. VRAM-residency hints (`PreLoad`, `SetPriority`,
// `EvictManagedResources`) and obsolete features that every modern
// Windows driver also ignores (`GetSoftwareVertexProcessing`,
// `GetNPatchMode`). Logged once so the first call is still visible for
// triage; info keeps them off the `RUST_LOG=warn` port-candidate surface.
//
// The cut line: no-op / non-portable D3D9 methods log once at info
// (first call stays visible for triage) but never at warn, which is
// reserved for port candidates.
#[macro_export]
macro_rules! log_once_info {
    (target: $target:expr, $($arg:tt)*) => {{
        static FIRED: ::core::sync::atomic::AtomicBool =
            ::core::sync::atomic::AtomicBool::new(false);
        if !FIRED.swap(true, ::core::sync::atomic::Ordering::Relaxed) {
            ::log::info!(target: $target, $($arg)*);
        }
    }};
}

// Mirror of `log_once_warn_by!` at `log::Level::Info`. Use for done-by-
// design calls whose format string carries a value that varies across
// legitimate inputs — e.g. `SetSoftwareVertexProcessing({software})`
// where both `0` and `1` are legitimate and we want both visible.
#[macro_export]
macro_rules! log_once_info_by {
    (target: $target:expr, key: $key:expr, $($arg:tt)*) => {{
        static SEEN: ::std::sync::LazyLock<
            ::std::sync::Mutex<::std::collections::BTreeSet<u64>>,
        > = ::std::sync::LazyLock::new(|| {
            ::std::sync::Mutex::new(::std::collections::BTreeSet::new())
        });
        let k: u64 = $key;
        let first = {
            let mut s = SEEN.lock().expect("log_once_info_by set poisoned");
            s.insert(k)
        };
        if first {
            ::log::info!(target: $target, $($arg)*);
        }
    }};
}

/// Test helper.
///
/// Returns `true` the first time a given `key` is seen by the shared
/// `seen` set, `false` on subsequent calls. Mirrors the first-seen check
/// inside `log_once_warn_by!` so the dedup semantics can be asserted
/// without installing a `log` subscriber (which would require the crate
/// to opt into `log`'s `std` feature just for tests).
#[cfg(test)]
pub fn first_seen(seen: &std::sync::Mutex<std::collections::BTreeSet<u64>>, key: u64) -> bool {
    seen.lock().unwrap().insert(key)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, sync::Mutex};

    use super::first_seen;

    #[test]
    fn first_seen_dedups_repeat_keys_and_separates_distinct() {
        let seen: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());
        assert!(first_seen(&seen, 42));
        assert!(!first_seen(&seen, 42));
        assert!(first_seen(&seen, 43));
        assert!(!first_seen(&seen, 43));
        // Packed two-u8 key, matching the resolve_attrs_for_ff site.
        let pack = |u: u8, i: u8| (u64::from(u) << 8) | u64::from(i);
        assert!(first_seen(&seen, pack(1, 0))); // BLENDWEIGHT
        assert!(first_seen(&seen, pack(2, 0))); // BLENDINDICES
        assert!(!first_seen(&seen, pack(1, 0)));
    }

    // Compile-only: the macros must accept the call shapes the codebase
    // uses. If macro arg parsing regresses, this fails to build.
    #[test]
    fn macros_compile_with_typical_shapes() {
        crate::log_once_warn!(target: "t", "no args");
        crate::log_once_warn!(target: "t", "fmt {} {}", 1, 2);
        crate::log_once_warn_by!(target: "t", key: 0u64, "no args");
        crate::log_once_warn_by!(target: "t", key: 7u64, "fmt {} {}", 1, 2);
        let state: u32 = 257;
        crate::log_once_warn_by!(
            target: "t",
            key: u64::from(state),
            "SetTransform: D3DTS_{state} not honoured — value dropped"
        );
        crate::log_once_trace_by!(target: "t", key: 0u64, "no args");
        crate::log_once_trace_by!(target: "t", key: 7u64, "fmt {} {}", 1, 2);
        crate::log_once_trace_by!(
            target: "t",
            key: u64::from(state),
            "drop: VS {state:#x} did not resolve"
        );
        crate::log_once_debug_by!(target: "t", key: 0u64, "no args");
        crate::log_once_debug_by!(target: "t", key: 7u64, "fmt {} {}", 1, 2);
        crate::log_once_debug_by!(
            target: "t",
            key: u64::from(state),
            "CheckDeviceFormat OK {state:#x}"
        );
        crate::log_once_info!(target: "t", "no args");
        crate::log_once_info!(target: "t", "fmt {} {}", 1, 2);
        crate::log_once_info_by!(target: "t", key: 0u64, "no args");
        crate::log_once_info_by!(
            target: "t",
            key: u64::from(state),
            "SetSoftwareVertexProcessing({state}): obsolete"
        );
    }
}
