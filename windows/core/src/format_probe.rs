//! Dedup keys for the `CheckDeviceFormat` diagnostic lines.
//!
//! `CheckDeviceFormat` is a four-field query, and the COM layer logs one line
//! per distinct query, so the log reads as the map of the render path a title
//! is choosing. Two of those fields span the whole `u32` (a `D3DFORMAT` can be
//! a fourcc) and `D3DUSAGE` reaches bit 23, so the fields do not fit side by
//! side in the `u64` the `log_once_*_by!` macros key on. A packing that
//! overlaps two of them makes one query swallow another's line, and the query
//! whose line went missing then reads as one the title never made.

use std::hash::Hasher;

use xxhash_rust::xxh3::Xxh3;

/// Dedup key for one `CheckDeviceFormat` query shape.
///
/// Folds the query fields with `xxh3`, so distinct field tuples give distinct
/// keys up to the width of the hash. Each log site owns its own seen-set, so a
/// `from_probe` key and a `from_query` key never meet.
pub struct FormatProbeKey(u64);

impl FormatProbeKey {
    /// Mint from the fields the entry probe reports.
    ///
    /// The probe fires ahead of any validation, and names the resource being
    /// asked about rather than the display mode it is asked under.
    #[must_use]
    pub fn from_probe(usage: u32, rtype: u32, check_format: u32) -> Self {
        Self(fold(&[usage, rtype, check_format]))
    }

    /// Mint from the full four-field query.
    ///
    /// Used by the accepted-query line, which reports `adapter_format` too.
    #[must_use]
    pub fn from_query(adapter_format: u32, usage: u32, rtype: u32, check_format: u32) -> Self {
        Self(fold(&[adapter_format, usage, rtype, check_format]))
    }

    /// Inner u64.
    ///
    /// The `log_once_*_by!` macros take their dedup key as a bare `u64`.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

fn fold(fields: &[u32]) -> u64 {
    let mut h = Xxh3::new();
    for &field in fields {
        h.write_u32(field);
    }
    h.finish()
}

#[cfg(test)]
mod tests;
