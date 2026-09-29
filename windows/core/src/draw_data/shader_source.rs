//! Canonical shader-source records and lease-bound borrowed views.

use core::ptr::NonNull;

use super::{PsKey, VsKey, ps_source_disk_key_ff, vs_source_disk_key_ff};
use crate::{
    dxso::{FfPsKey, FfVsKey, VariantKey, VsSamplerKinds},
    ids::ProgramId,
    shader_key::{ps_source_disk_key_programmable, vs_source_disk_key_programmable},
};

bitflags::bitflags! {
    /// Shader constant-file dependencies captured once by the API.
    #[repr(transparent)]
    pub struct ShaderSourceFlags: u8 {
        const RELATIVE = 1;
        const INTEGER = 2;
        const BOOLEAN = 4;
        const BUMP_ENV = 8;
    }
}

/// Canonical programmable vertex source, with no implicit padding.
#[repr(C, align(8))]
pub struct ProgrammableVsSource {
    pub vs_id: ProgramId,
    pub max_const_used: u16,
    pub provided_input_mask: u16,
    pub flags: ShaderSourceFlags,
    pub clip_plane_count: u8,
    pub sampler_kinds: VsSamplerKinds,
}

impl ProgrammableVsSource {
    #[must_use]
    pub const fn uses_rel_const(&self) -> bool {
        self.flags.contains(ShaderSourceFlags::RELATIVE)
    }
    #[must_use]
    pub const fn uses_int_const(&self) -> bool {
        self.flags.contains(ShaderSourceFlags::INTEGER)
    }
    #[must_use]
    pub const fn uses_bool_const(&self) -> bool {
        self.flags.contains(ShaderSourceFlags::BOOLEAN)
    }
}

/// Canonical programmable pixel source, with initialized reserved bytes.
#[repr(C, align(8))]
pub struct ProgrammablePsSource {
    pub ps_id: ProgramId,
    pub max_const_used: u16,
    pub flags: ShaderSourceFlags,
    pub color_out_mask: u8,
    pub reserved: [u8; 4],
}

impl ProgrammablePsSource {
    #[must_use]
    pub const fn uses_bump_env(&self) -> bool {
        self.flags.contains(ShaderSourceFlags::BUMP_ENV)
    }
    #[must_use]
    pub const fn uses_int_const(&self) -> bool {
        self.flags.contains(ShaderSourceFlags::INTEGER)
    }
    #[must_use]
    pub const fn uses_bool_const(&self) -> bool {
        self.flags.contains(ShaderSourceFlags::BOOLEAN)
    }
}

/// Canonical fixed-function vertex source.
#[repr(C, align(8))]
pub struct FixedVsSource {
    pub key: FfVsKey,
    pub max_row_count: u16,
    pub reserved: [u8; 6],
}

/// Canonical fixed-function pixel source.
#[repr(C, align(8))]
pub struct FixedPsSource {
    pub key: FfPsKey,
    pub sampled_stage_mask: u16,
    pub constant_rows: u8,
    pub reserved: [u8; 3],
}

/// API builder ownership; only the selected canonical record is captured.
pub enum VsSource {
    Programmable(ProgrammableVsSource),
    FixedFunction(FixedVsSource),
}

/// API builder ownership; only the selected canonical record is captured.
pub enum PsSource {
    Programmable(ProgrammablePsSource),
    FixedFunction(FixedPsSource),
}

/// Borrowed source shared between cache probes and draw emission.
///
/// Copy duplicates a reference and tag, never source storage or ownership.
#[derive(Clone, Copy)]
pub enum VsSourceView<'a> {
    Programmable(&'a ProgrammableVsSource),
    FixedFunction(&'a FixedVsSource),
}

/// Borrowed source shared between cache probes and draw emission.
///
/// Copy duplicates a reference and tag, never source storage or ownership.
#[derive(Clone, Copy)]
pub enum PsSourceView<'a> {
    Programmable(&'a ProgrammablePsSource),
    FixedFunction(&'a FixedPsSource),
}

impl VsSource {
    #[must_use]
    pub const fn as_view(&self) -> VsSourceView<'_> {
        match self {
            Self::Programmable(v) => VsSourceView::Programmable(v),
            Self::FixedFunction(v) => VsSourceView::FixedFunction(v),
        }
    }
}
impl PsSource {
    #[must_use]
    pub const fn as_view(&self) -> PsSourceView<'_> {
        match self {
            Self::Programmable(v) => PsSourceView::Programmable(v),
            Self::FixedFunction(v) => PsSourceView::FixedFunction(v),
        }
    }
}

impl VsSourceView<'_> {
    #[must_use]
    pub fn key(self, variant: VariantKey) -> VsKey {
        match self {
            Self::Programmable(v) => VsKey::Programmable {
                vs_id: v.vs_id,
                variant,
                provided_input_mask: v.provided_input_mask,
                clip_plane_count: v.clip_plane_count,
                sampler_kinds: v.sampler_kinds,
            },
            Self::FixedFunction(v) => VsKey::FixedFunction {
                ff: v.key.clone(),
                variant,
            },
        }
    }
    #[must_use]
    pub fn disk_key(self) -> u64 {
        match self {
            Self::Programmable(v) => vs_source_disk_key_programmable(
                v.vs_id,
                v.provided_input_mask,
                v.clip_plane_count,
                v.sampler_kinds,
            ),
            Self::FixedFunction(v) => vs_source_disk_key_ff(&v.key),
        }
    }
}
impl PsSourceView<'_> {
    #[must_use]
    pub fn key(self, variant: VariantKey) -> PsKey {
        match self {
            Self::Programmable(v) => PsKey::Programmable {
                ps_id: v.ps_id,
                variant,
            },
            Self::FixedFunction(v) => PsKey::FixedFunction {
                ff: v.key.clone(),
                variant,
            },
        }
    }
    #[must_use]
    pub fn disk_key(self, variant: VariantKey) -> u64 {
        match self {
            Self::Programmable(v) => ps_source_disk_key_programmable(v.ps_id, variant),
            Self::FixedFunction(v) => ps_source_disk_key_ff(&v.key, variant),
        }
    }
}

/// Lease-bound pointer to one canonical vs source.
///
/// Copy duplicates the token without extending the enclosing frame lease.
#[derive(Clone, Copy)]
pub struct VsSourcePtr {
    pointer: NonNull<u8>,
    fixed: bool,
}
// SAFETY: construction requires immutable retained storage through all consumers.
unsafe impl Send for VsSourcePtr {}
impl VsSourcePtr {
    /// Borrow an API builder's selected record.
    ///
    /// # Safety
    /// Keep the builder immutable and allocated through every token copy.
    #[must_use]
    pub unsafe fn new(pointer: NonNull<VsSource>) -> Self {
        // SAFETY: the caller retains the initialized builder through all consumers.
        let source = unsafe { pointer.as_ref() };
        // SAFETY: its selected record has the same retained lifetime.
        unsafe { Self::from_view(source.as_view()) }
    }
    /// Retain a pointer to a validated canonical source record.
    ///
    /// # Safety
    /// Keep the record immutable and allocated through every token copy.
    #[must_use]
    pub unsafe fn from_view(source: VsSourceView<'_>) -> Self {
        match source {
            VsSourceView::Programmable(value) => Self {
                pointer: NonNull::from(value).cast(),
                fixed: false,
            },
            VsSourceView::FixedFunction(value) => Self {
                pointer: NonNull::from(value).cast(),
                fixed: true,
            },
        }
    }
    #[must_use]
    pub fn as_ref(&self) -> VsSourceView<'_> {
        if self.fixed {
            let pointer = self.pointer.as_ptr() as usize as *const FixedVsSource;
            // SAFETY: construction binds this token to an aligned initialized fixed source.
            VsSourceView::FixedFunction(unsafe { &*pointer })
        } else {
            let pointer = self.pointer.as_ptr() as usize as *const ProgrammableVsSource;
            // SAFETY: construction binds this token to an aligned initialized programmable source.
            VsSourceView::Programmable(unsafe { &*pointer })
        }
    }
}
/// Lease-bound pointer to one canonical ps source.
///
/// Copy duplicates the token without extending the enclosing frame lease.
#[derive(Clone, Copy)]
pub struct PsSourcePtr {
    pointer: NonNull<u8>,
    fixed: bool,
}
// SAFETY: construction requires immutable retained storage through all consumers.
unsafe impl Send for PsSourcePtr {}
impl PsSourcePtr {
    /// Borrow an API builder's selected record.
    ///
    /// # Safety
    /// Keep the builder immutable and allocated through every token copy.
    #[must_use]
    pub unsafe fn new(pointer: NonNull<PsSource>) -> Self {
        // SAFETY: the caller retains the initialized builder through all consumers.
        let source = unsafe { pointer.as_ref() };
        // SAFETY: its selected record has the same retained lifetime.
        unsafe { Self::from_view(source.as_view()) }
    }
    /// Retain a pointer to a validated canonical source record.
    ///
    /// # Safety
    /// Keep the record immutable and allocated through every token copy.
    #[must_use]
    pub unsafe fn from_view(source: PsSourceView<'_>) -> Self {
        match source {
            PsSourceView::Programmable(value) => Self {
                pointer: NonNull::from(value).cast(),
                fixed: false,
            },
            PsSourceView::FixedFunction(value) => Self {
                pointer: NonNull::from(value).cast(),
                fixed: true,
            },
        }
    }
    #[must_use]
    pub fn as_ref(&self) -> PsSourceView<'_> {
        if self.fixed {
            let pointer = self.pointer.as_ptr() as usize as *const FixedPsSource;
            // SAFETY: construction binds this token to an aligned initialized fixed source.
            PsSourceView::FixedFunction(unsafe { &*pointer })
        } else {
            let pointer = self.pointer.as_ptr() as usize as *const ProgrammablePsSource;
            // SAFETY: construction binds this token to an aligned initialized programmable source.
            PsSourceView::Programmable(unsafe { &*pointer })
        }
    }
}
