//! Boolean attributes a texture carries from its creation call.
//!
//! One packed field on the COM-side container, and the input every predicate
//! that classifies a texture by the call that made it reads. Kept here rather
//! than beside the container so the classifiers themselves stay host-testable.

bitflags::bitflags! {
    /// Boolean attributes of a texture, packed into one field.
    ///
    /// The packing keeps `TextureInner`/`TextureCreateInfo` under the bool-bag
    /// lint and tightens the surrounding structs' tail padding.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TextureFlags: u8 {
        /// `D3DUSAGE_AUTOGENMIPMAP` requested AND the format supports it.
        ///
        /// Metal can't auto-generate compressed BC/DXT, so the flag is dropped
        /// in `device_create_texture` for those. When set, mip-0 uploads append
        /// a `BlitCommand::generate_mipmaps` to the frame's leading-blit list
        /// right after the mip-0 `CopyBufferToTexture`, and the COM
        /// `IDirect3DBaseTexture9::GenerateMipSubLevels` call pushes the same op
        /// explicitly. Also collapses the app-visible level count to 1.
        const AUTOGEN_MIPMAP = 1 << 1;
        /// Sampleable shadow-map texture.
        ///
        /// Created via
        /// `CreateTexture(format=D24X8, usage=D3DUSAGE_DEPTHSTENCIL)`.
        /// `LockRect` bails with INVALIDCALL, no staging is wired up, and
        /// `SetDepthStencilSurface` resolves through to this texture's Metal
        /// handle when one of its mip surfaces is bound.
        const DEPTH_FORMAT = 1 << 2;
        /// Cube-map texture.
        ///
        /// Backed by six Metal array slices when the pool is GPU-visible.
        const CUBE = 1 << 3;
        /// The texture behind a `CreateOffscreenPlainSurface` surface.
        ///
        /// D3D9 lets a game lock such a surface even in the default pool,
        /// so its staging is never released after an upload.
        const OFFSCREEN_PLAIN = 1 << 4;
        /// Created through `CreateVolumeTexture`, whatever its depth.
        ///
        /// Distinct from `Direct3DTexture9::is_volume` (`depth > 1`), which asks
        /// whether the *Metal* texture is 3D: a single-slice volume texture is
        /// created 2D on both sides yet still hands out `IDirect3DVolume9`
        /// sub-resources rather than surfaces. The cached sub-resource slots have
        /// to be freed as the kind they hold, so the container records which one
        /// that is.
        const VOLUME_TEXTURE = 1 << 5;
        /// System-memory resource: no Metal texture exists for it yet.
        ///
        /// Set at creation for the two CPU-only pools (`D3DPOOL_SYSTEMMEM` and
        /// `D3DPOOL_SCRATCH`). Nothing warms up an `MTLTexture` and nothing
        /// uploads a mip while it is set, so a texture the application only
        /// locks, copies from, or reads back into costs system memory alone.
        /// Binding one for sampling clears it (`promote_to_gpu` on the texture
        /// wrapper), because D3D9 does sample a bound system-memory texture.
        const CPU_ONLY = 1 << 6;
    }
}
