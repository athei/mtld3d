# Status

What mtld3d implements, what it does not yet, what it never will, and the
divergences from D3D9 it keeps on purpose. The tested games are in
[`GAMES.md`](GAMES.md); the end-to-end suite's coverage is in
[`COVERAGE.md`](../windows/tests/COVERAGE.md).

## Supported

- Shader models 1.x through 3.0, translated from DXSO to MSL and cached on
  disk by content hash.
- The fixed-function pipeline: lighting, texture-coordinate generation
  (camera-space normal, position and reflection vector, and sphere map), the
  texture-stage cascade (CURRENT/TEMP registers, per-stage constants,
  DOTPRODUCT3 color and alpha, MODULATEALPHA_ADDCOLOR,
  MODULATECOLOR_ADDALPHA, MODULATEINVALPHA_ADDCOLOR,
  MODULATEINVCOLOR_ADDALPHA and premultiplied texture-alpha blending),
  vertex blending and range-based
  vertex fog. Table fog supports Z and W sources with LINEAR,
  EXP and EXP2; SM3 shaders own their fog.
- Every draw call and primitive type, point sprites, user clip planes, all
  sixteen vertex streams, hardware instancing.
- State blocks, occlusion and event queries.
- `D3DUSAGE_DYNAMIC` with `D3DPOOL_MANAGED` or `D3DPOOL_SCRATCH` is rejected
  with `D3DERR_INVALIDCALL` at 2D, cube and volume texture creation, for every
  format: the flag asks to drive the copy the device reads, which the managed
  pool keeps for the runtime and the scratch pool does not have at all.
- Signed Q16W16V16U16 textures use native RGBA16Snorm storage for 2D, cube
  and volume resources. All four lanes retain signed 16-bit samples without
  conversion. Render-target and sRGB usages are unavailable, and texture/cube
  AUTOGEN uses one actual level.
- A2R10G10B10 and A2B10G10R10 textures use native BGR10A2Unorm and
  RGB10A2Unorm storage for 2D, cube and volume resources, including all ten
  RGB bits and two alpha bits; blue is the low lane of the first and red of
  the second. Explicit mip chains and DEFAULT offscreen ColorFill are
  supported. Render-target, display/backbuffer, sRGB and BUMP usages remain
  unavailable, and AUTOGEN uses one actual level.
- Signed Q8W8V8U8 textures use native RGBA8Snorm storage across 2D, cube and
  volume resources. All four channels, including alpha, retain signed values.
  Render targets and sRGB are unavailable; AUTOGEN texture/cube requests use
  a single-level fallback.
- Signed V16U16 textures use native 16-bit U/V storage across 2D, cube and
  volume resources. AUTOGEN requests on 2D/cube use the one-level NOAUTOGEN
  fallback; render-target and sRGB usages remain unavailable.
- `ColorFill` of a DEFAULT offscreen plain surface in V8U8, V16U16, Q8W8V8U8
  or Q16W16V16U16 writes each colour channel as the nearest nonnegative signed
  code, R, G, B, A into U, V, W, Q.
- DXT1 to DXT5 volume textures use native BC1, BC2 and BC3 3D storage in
  every pool, with short mips, slice filtering and sRGB sampling. DXT2 and
  DXT4 keep their identities and sample the stored blocks unchanged. ATI1
  and packed-YUV volumes stay SCRATCH-only, and render-target usage and mip
  autogeneration stay unavailable on volumes.
- Compressed (DXT1 to DXT5, ATI1), integer and float formats, cube and volume
  textures, auto-generated mipmaps, `StretchRect` with format conversion and
  packed and planar YUV decoding, `GetDC`.
- Planar YV12 and NV12 as DEFAULT-pool offscreen plain surfaces: they lock at
  the 4-byte-aligned width with the chroma planes after the luma rows, and
  `StretchRect` decodes them (reduced-range BT.601) into any render target, at
  any size, and 1:1 into a colour offscreen plain. They are no texture, cube,
  volume or render-target format.
- Managed 2D texture publication through `AddDirtyRect`, including scaled
  mip regions. `NO_DIRTY_UPDATE` adds no publication after initialization;
  initial uploads and eviction retain their CPU source. Overlapping partial
  locks may still change bytes an earlier queued upload reads.
- Raw sampleable depth (INTZ, DF16, DF24), hardware shadow comparisons on
  standard depth formats, depth bias, the full two-sided stencil test, and
  GPU-only plain depth textures for RESZ destinations. Dynamic DEFAULT-pool
  D16, D24X8 and D24S8 textures support packed CPU locks, explicit mips and
  RESZ readback.
- Fetch4 gathers on 2D L8, L16, A8, R16F, R32F and raw-depth textures.
- Anisotropic filtering, LOD bias, sRGB read on the formats whose Metal
  counterpart has an sRGB twin and sRGB write into any render target (through
  that twin where it exists, through the pixel shader otherwise, which is
  what `D3DUSAGE_QUERY_SRGBWRITE` answers), alpha test, scissor,
  separate alpha blend, blend factor, write masks, native wireframe fill.
- Four render targets with independent formats and blending.
- Multisampling at 2x and 4x, 8x where the device offers it, and the ATOC
  alpha-to-coverage extension through `D3DRS_ADAPTIVETESS_Y`, plus AMD's
  `A2M1`/`A2M0` controls through `D3DRS_POINTSIZE`. `ALPHATESTENABLE` gates
  ATOC; A2M is an independent latch. Either request replaces alpha testing
  only on multisampled render targets. Both controls work independently
  of reported vendor; control writes preserve numeric point size.
- Windowed and fullscreen presentation of the implicit swap chain, mode
  enumeration, hardware and software cursors, MetalFX upscaling, HDR output.
- The gamma ramp of a fullscreen device. `SetGammaRamp` validates the ramp,
  keeps it for `GetGammaRamp`, and the present pass looks each channel up in
  it on the way to the drawable, the software cursor's sprite included. A ramp
  that changes nothing costs nothing, and `D3DSGR_CALIBRATE` is accepted and
  ignored, which is what a device without `D3DCAPS2_CANCALIBRATEGAMMA`
  answers. Readback is unaffected: the ramp is the display's transfer
  function, not part of the image the game drew.
- Every presentation interval: `DEFAULT` and `ONE` pace at the display rate,
  `TWO`, `THREE` and `FOUR` at the reported mode's refresh rate over two,
  three and four, and `IMMEDIATE` runs free. `present.maxFps` lowers any of
  them.
- `D3DCREATE_MULTITHREADED`: a device created with it, and every object it
  creates, may be called from any thread; each entry point holds a reentrant
  per-device lock, and a device created without the flag pays nothing.
- Several devices alive at once, each on its own window, each owning its
  display state on the unix side.

## Not implemented yet

Each fails cleanly, with an absent cap bit or a documented error return,
unless its entry says otherwise.

- Point polygon fill: Metal has no point-fill mode, so the state is warned
  once and drawn solid.
- Dynamic depth textures outside DEFAULT-pool 2D D16, D24X8 and D24S8,
  including dynamic depth attachments, remain unavailable. Depth textures
  have no vertex sampling. Automatic mip-generation requests use the
  single-level `D3DOK_NOAUTOGEN` fallback.
- A colour target smaller than its depth surface under `render.scale`: a
  whole-surface depth or stencil `Clear` reaches the whole depth surface at
  any scale, but two cases keep the colour target's extent, each warned
  once. One is the draws through a 1x1 render target 0 left unwritten over a
  depth surface that `render.scale` reduces; the other is a depth or stencil
  `Clear` by rect or by viewport that reaches past the colour target while
  either surface is scaled. At the identity scale both reach the depth
  surface, as D3D9 does.
- Additional swap chains: `CreateAdditionalSwapChain` succeeds and its
  object carries its own present parameters, but its `GetBackBuffer` hands
  back the device's back buffer and its `Present` presents the device frame
  into the device window, warned once. There is no back buffer or window per
  chain, so a title that renders a second view through one sees it alias the
  first.
- `Present` arguments: a source rect, a destination rect, a destination
  window override and a dirty region are accepted on
  `IDirect3DDevice9::Present` and `IDirect3DSwapChain9::Present` and ignored,
  each warned once; the whole back buffer is presented across the device
  window.
- SM3 linkage limits. A `ps_3_0` links at most `MAX_LINKED_INPUTS` (eight)
  input semantics outside the fixed-function varyings (NORMAL, TANGENT,
  COLOR2 and the like); a ninth and later read zero, warned once. A
  pretransformed (XYZRHW, POSITIONT) draw feeds a `ps_3_0` the declaration's
  elements by semantic, at most eight of them besides the position, colours,
  texture coordinates and point size, the rest reading zero, warned once.
  Two of its inputs keep the fixed-function value instead, both unverified
  against Windows: TEXCOORDn is the set stage n's `D3DTSS_TEXCOORDINDEX`
  routes, which is the declaration's TEXCOORDn unless the index was changed,
  and FOG reads the specular alpha while fog is enabled with both fog modes
  NONE, where the fixed-function pixel stage reads its fog factor.
- Timestamp, timestamp frequency, timestamp disjoint and other niche query
  types: capability probes and creation report `D3DERR_NOTAVAILABLE`.
- Fixed-function bump-environment mapping: `D3DTOP_BUMPENVMAP` and
  `D3DTOP_BUMPENVMAPLUMINANCE` are absent from `TextureOpCaps`, and
  `D3DUSAGE_QUERY_LEGACYBUMPMAP` answers `D3DERR_NOTAVAILABLE` for every
  format to match. The SM1 `texbem` instruction and the `D3DTSS_BUMPENVMAT*`
  matrix it reads are a separate surface and do work.
- DEFAULT offscreen cross-format `StretchRect` outside the narrow normalized
  codecs and A16B16G16R16/A32B32G32R32F into A8R8G8B8 returns
  `D3DERR_INVALIDCALL`. Wide-to-wide conversion and offscreen scaling remain
  unsupported. Render-target conversion uses its separate GPU path.
- Planar YUV (YV12, NV12) outside the DEFAULT pool: SYSTEMMEM, SCRATCH and
  MANAGED offscreen plains are rejected with `D3DERR_INVALIDCALL`, and so is
  `UpdateSurface` with a planar endpoint. A YV12 surface of odd height is
  rejected as well, because the origin of its U plane is not pinned by any
  reference. Planar chroma is never filtered: `D3DTEXF_LINEAR` filters luma
  and replicates each chroma sample over its 2x2 block. `ColorFill` of a
  packed or planar YUV surface succeeds and leaves it unfilled.
- Packed YUV (YUY2, UYVY) textures: nothing decodes them when they are
  sampled, so `CheckDeviceFormat` answers `D3DERR_NOTAVAILABLE` for every
  texture query and only a `D3DPOOL_SCRATCH` texture creates. They remain
  offscreen plain surfaces, and a DEFAULT-pool one is a `StretchRect` source
  that decodes into any render target and 1:1 into a colour offscreen plain.
- Scaled, sub-rect or converting depth-to-depth `StretchRect`: only the
  whole-surface 1:1 copy between same-format DEFAULT-pool depth surfaces
  works, multisample resolve included.
- D3D9Ex: `Direct3DCreate9Ex` resolves and answers `D3DERR_NOTAVAILABLE`, so
  a runtime probe sees a d3d9 without 9Ex rather than a broken DLL. There is
  no `IDirect3D9Ex` and no `IDirect3DDevice9Ex`, every create rejects a
  non-null `pSharedHandle` with `E_NOTIMPL`, and `Caps2` leaves
  `CANSHARERESOURCE` off. 9Ex is the same device created with an extended
  flag rather than a separate contract: the flag refuses `D3DPOOL_MANAGED`,
  changes which pools a caller may lock, reports `WHQLLevel` 1 on the adapter
  identifier, and puts the extended entry points on the objects that already
  exist, `CreateDeviceEx`, `PresentEx`, `ResetEx`, `CheckDeviceState`,
  `GetDisplayModeEx`, `ComposeRects`, the frame-latency pair, the SYSTEMMEM
  user-memory create that the same `pSharedHandle` parameter carries, and
  shared resources. It is wanted eventually and waits for a title that needs
  it: both World of Warcraft targets create a plain device, and nothing else
  in the tested set asks for 9Ex. Issue #789 is the record of the decision
  and of what an implementation would cover.
- SM3 relative addressing of anything but the float constants: an input
  read through the loop counter (`v[aL]` in `ps_3_0` and `vs_3_0`) and a
  `vs_3_0` output written through it (`o[aL]`). A shader that writes
  `o[aL]` fails `CreateVertexShader` with `D3DERR_INVALIDCALL`; one that
  reads `v[aL]` is created, but its translation fails with a logged error
  and its draws are left out, so this one does not fail cleanly. No known
  title needs either.
- A draw that samples the colour render target it is drawing into (a
  feedback loop) has no hazard handling, and nothing detects the bind, so
  this one does not fail cleanly. Metal leaves the result undefined, and an
  Apple GPU returns the target's contents from before the pass began rather
  than the pixels the pass has written. Depth has this handled: a draw that
  samples the bound depth attachment reads a snapshot copy
  (`depth_snapshot_for_sampling` in `unix/unix/src/encoder.rs`). Colour
  has no equivalent. DXVK detects the bind and resolves it; it is not built
  here because no known title needs it.

## Deliberately not implemented

- D3D9On12: `Direct3DCreate9On12` is not exported and there is no bridge
  behind it. It maps a D3D9 device onto a D3D12 device, which has no
  counterpart here.
- Physical display-mode switching: the mode is meant to stay virtual, see the
  [Fullscreen](../INSTALL.md#fullscreen) section of `INSTALL.md`.
- Device loss on focus changes: no exclusive mode is taken, so a focus
  change loses nothing, and `TestCooperativeLevel` reports `D3D_OK` across
  it. A device is lost only when the layer itself fails: a frame that
  cannot be encoded or submitted to Metal, such as a depth transfer whose
  private planes cannot be allocated, latches `D3DERR_DEVICELOST`. A frame
  whose commands cannot be recorded because the layer's own command buffer
  could not grow latches `E_OUTOFMEMORY` instead. Every later call that
  reports device state, `TestCooperativeLevel` and `Reset` included,
  returns the latched code until the device is released, and the log names
  the step that failed first. A failed device drops its open frame at each
  `Present`, and its final `Release` destroys the render targets and depth
  surfaces released after the failure. There is no recovery short of
  creating a new device.
- Software paths: no reference rasterizer, no software vertex processing, no
  `RegisterSoftwareDevice`; the default Metal device is the only adapter.
- Legacy remnants: N-patch and RT-patch tessellation, vertex tweening,
  palettized textures. Accepted or rejected per spec, non-functional.
- The depth formats `D15S1` and `D24X4S4`, whose one- and four-bit stencil no
  Metal format has: answered `D3DERR_NOTAVAILABLE` and refused by every
  create, as the reference implementations do.

## Kept divergences

Divergences from D3D9 kept on purpose because closing them costs frame time,
memory, or a game that relies on the looser behaviour. The rationale for each
is in [`CONFORMANCE.md`](../unix/conformance/CONFORMANCE.md#kept-divergences).

- `LockRect` serves a level of a non-dynamic DEFAULT-pool 2D texture, which
  D3D9 rejects. No knob.
- `GetData(D3DGETDATA_FLUSH)` can answer a pending occlusion query at once
  instead of waiting for the GPU. `query.flushImmediate`, off by default.
- EVENT query polls queue the open frame even without `D3DGETDATA_FLUSH`,
  so a caller polling before Present can make progress. No knob.
- An EVENT query poll can answer completed at once instead of waiting for
  the GPU. `query.eventImmediate`, off by default.
- Depth and stencil are discarded at every `Present` on a surface nothing
  samples. D3D9 keeps them unless the game sets
  `D3DPRESENTFLAG_DISCARD_DEPTHSTENCIL` or creates the surface with
  `Discard = TRUE`, and neither is consulted. Keeping them costs a store and
  a load of the depth surface every frame on a tile-based GPU. A game that
  tests against an earlier frame's depth or stencil without clearing reads
  undefined values. No knob.
- A partial `Lock` of a dynamic vertex or index buffer without
  `D3DLOCK_DISCARD` returns a pointer a queued draw may still read. No knob.
- A partial `LockRect` of a texture level without `D3DLOCK_NOOVERWRITE` or
  `D3DLOCK_READONLY` returns a pointer an upload may still read. Under an
  upload from an earlier frame, or one a GPU operation has seen, only two
  writes land in place: this partial lock, and any write into a level the game
  holds mapped by a lock or a device context. Every other CPU writer renames
  the level's staging first. Under an upload of the frame being recorded that
  no draw or other GPU operation has seen, it writes in place, which no draw
  can tell apart. Writes into a render-target or depth texture
  and read-backs from the GPU always rename when an upload still reads the
  level. No knob.
- A DEFAULT-pool `D3DUSAGE_WRITEONLY` static buffer keeps no CPU copy once
  uploaded, so a read through the lock pointer sees zeros.
  `buffer.ignoreLockBounds` keeps the copy.
- The window procedure carrying cursor realization and the windowed
  auto-resize is the device window's, and follows a `Reset` that names another
  window; D3D9 subclasses the focus window instead. No knob.
- A rejected `Reset` leaves render target 0 on the back buffer and the
  depth stencil on the implicit surface, as a successful one does, where
  the reference implementations leave both unbound; what D3D9 itself
  leaves there is not observable. No knob.
- A texture whose Metal allocation is refused at an extent within the
  reported limits still creates: its Metal texture is made on the encoder
  thread after `CreateTexture`, `CreateCubeTexture` or `CreateVolumeTexture`
  has returned `D3D_OK`, so the refusal is logged and the texture samples as
  opaque black, where D3D9 answers the create with an out-of-memory error.
  No knob.
- `D3DRS_MULTISAMPLEANTIALIAS = FALSE` is ignored. No knob.
- A `D3DSAMP_BORDERCOLOR` other than transparent black, opaque black or
  opaque white reads as opaque black under `D3DTADDRESS_BORDER`; D3D9
  returns the colour the game set. No knob.
- `ATI1` creates as a 2D texture in every pool and as an offscreen plain
  surface in every pool a plain surface takes, while every format query
  answers `D3DERR_NOTAVAILABLE` for it, and its lock reports the BC4 block
  pitch rather than the pitch D3D9 reports for the format. No knob.
- `D16_LOCKABLE` and `D32F_LOCKABLE` create as the auto depth-stencil and as
  a depth-stencil texture, on a 32-bit float depth, while every depth query
  answers `D3DERR_NOTAVAILABLE` for them: a `LockRect` of the depth surface
  fails, warned once. They are single-sampled; a multisampled swap chain
  refuses them, as `CheckDeviceMultiSampleType` does. A standalone
  depth-stencil surface in either format is refused, as before. No knob.
- `CreateDevice` and `Reset` accept any back-buffer format and present a
  BGRA8 back buffer for every one but A8R8G8B8 and X8R8G8B8, which
  `GetDesc` then reports as X8R8G8B8, where D3D9 refuses a format outside
  its back-buffer set and keeps a 16-bit one. `CheckDeviceType` offers that
  set only. No knob.
- The adapter mode list leaves out every display size win32u cannot scale
  the monitor to in its 16-bit ratio, on every Wine, and a fullscreen request
  for one follows the window instead of setting the mode. The list describes
  the primary display only and serves first the sizes that fill it (a bar
  of less than one physical pixel once win32u has scaled them onto it), then
  the standard sizes of another shape that user32 lists for it (2560x1440
  down to 640x480), which win32u letterboxes in fullscreen, and nothing else
  of user32's list. No knob.
- A windowed device's `SetGammaRamp` is stored and reported back but changes
  nothing on screen, where D3D9 ramps the whole desktop. Only the implicit
  swap chain carries a ramp. No knob.
- A draw whose shader library or render pipeline is still building is left
  out of its frame when everything it depends on is rebuilt every frame
  (the discard-effect back buffer, or targets and depth or stencil planes
  cleared in this frame and the one before), none of it was copied or
  sampled into kept content in the last 600 frames, and no occlusion
  query counts,
  and appears once the build lands; D3D9 draws every call in its frame.
  Building inline stalls the frame for the length of a Metal compile.
  `shader.asyncCompile`, on by default.
