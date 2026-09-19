# Status

What mtld3d implements, what it does not yet, what it never will, and the
divergences from D3D9 it keeps on purpose. The tested games are in the
[README](../README.md#tested-games); the end-to-end suite's coverage is in
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
- Signed Q16W16V16U16 textures use native RGBA16Snorm storage for 2D, cube
  and volume resources. All four lanes retain signed 16-bit samples without
  conversion. Render-target and sRGB usages are unavailable; texture/cube
  AUTOGEN uses one actual level, and MANAGED+DYNAMIC creation is rejected.
- A2R10G10B10 textures use native BGR10A2Unorm storage for 2D, cube and
  volume resources, including all ten RGB bits and two alpha bits. Explicit
  mip chains and DEFAULT offscreen ColorFill are supported. Render-target,
  display/backbuffer, sRGB and BUMP usages remain unavailable; AUTOGEN uses
  one actual level and MANAGED+DYNAMIC creation is rejected.
- Signed Q8W8V8U8 textures use native RGBA8Snorm storage across 2D, cube and
  volume resources. All four channels, including alpha, retain signed values.
  Render targets and sRGB are unavailable; AUTOGEN texture/cube requests use
  a single-level fallback. MANAGED+DYNAMIC creation is rejected.
- Signed V16U16 textures use native 16-bit U/V storage across 2D, cube and
  volume resources. AUTOGEN requests on 2D/cube use the one-level NOAUTOGEN
  fallback; render-target and sRGB usages remain unavailable.
- Compressed (DXT1 to DXT5, ATI1), integer and float formats, cube and volume
  textures, auto-generated mipmaps, `StretchRect` with format conversion and
  YUV decoding, `GetDC`.
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
- Anisotropic filtering, LOD bias, sRGB read and write, alpha test, scissor,
  separate alpha blend, blend factor, write masks, native wireframe fill.
- Four render targets with independent formats and blending.
- Multisampling at 2x and 4x, 8x where the device offers it, and the ATOC
  alpha-to-coverage extension through `D3DRS_ADAPTIVETESS_Y`, plus AMD's
  `A2M1`/`A2M0` controls through `D3DRS_POINTSIZE`. `ALPHATESTENABLE` gates
  ATOC; A2M is an independent latch. Either request replaces alpha testing
  only on multisampled render targets. Both controls work independently
  of reported vendor; control writes preserve numeric point size.
- Windowed and fullscreen swap chains, mode enumeration, hardware and software
  cursors, MetalFX upscaling, HDR output.
- `D3DCREATE_MULTITHREADED`: a device created with it, and every object it
  creates, may be called from any thread; each entry point holds a reentrant
  per-device lock, and a device created without the flag pays nothing.
- Several devices alive at once, each on its own window, each owning its
  display state on the unix side.

## Not implemented yet

Each fails cleanly, with an absent cap bit or a documented error return.

- Point polygon fill: Metal has no point-fill mode, so the state is warned
  once and drawn solid.
- Dynamic depth textures outside DEFAULT-pool 2D D16, D24X8 and D24S8,
  including dynamic depth attachments, remain unavailable. Depth textures
  have no vertex sampling. Automatic mip-generation requests use the
  single-level `D3DOK_NOAUTOGEN` fallback.
- Timestamp, timestamp frequency, timestamp disjoint and other niche query
  types: capability probes and creation report `D3DERR_NOTAVAILABLE`.
- DEFAULT offscreen cross-format `StretchRect` outside the narrow normalized
  codecs and A16B16G16R16/A32B32G32R32F into A8R8G8B8 returns
  `D3DERR_INVALIDCALL`. Wide-to-wide conversion and offscreen scaling remain
  unsupported. Render-target conversion uses its separate GPU path.
- Scaled, sub-rect or converting depth-to-depth `StretchRect`: only the
  whole-surface 1:1 copy between same-format DEFAULT-pool depth surfaces
  works, multisample resolve included.

## Deliberately not implemented

- D3D9Ex: `Direct3DCreate9Ex` resolves and answers `D3DERR_NOTAVAILABLE`, so
  a runtime probe sees a d3d9 without 9Ex rather than a broken DLL; no
  `IDirect3D9Ex`, shared handles or D3D9On12. A different contract, built for
  the Vista compositor.
- Physical display-mode switching: the mode is meant to stay virtual, see the
  README's [Fullscreen](../README.md#fullscreen) section.
- Device loss: no exclusive mode is taken, so nothing is ever lost, and
  `TestCooperativeLevel` reports `D3D_OK` across focus changes.
- Software paths: no reference rasterizer, no software vertex processing, no
  `RegisterSoftwareDevice`; the default Metal device is the only adapter.
- Legacy remnants: N-patch and RT-patch tessellation, vertex tweening,
  palettized textures, gamma ramp. Accepted or rejected per spec,
  non-functional.

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
- Depth stores are elided where nothing reads the buffer back. No knob.
- A partial `Lock` of a dynamic vertex or index buffer without
  `D3DLOCK_DISCARD` returns a pointer a queued draw may still read. No knob.
- A partial `LockRect` of a texture level without `D3DLOCK_NOOVERWRITE` or
  `D3DLOCK_READONLY` returns a pointer an upload may still read. No knob.
- A DEFAULT-pool `D3DUSAGE_WRITEONLY` static buffer keeps no CPU copy once
  uploaded, so a read through the lock pointer sees zeros.
  `buffer.ignoreLockBounds` keeps the copy.
- The window procedure carrying cursor realization and the windowed
  auto-resize is the device window's, and follows a `Reset` that names another
  window; D3D9 subclasses the focus window instead. No knob.
- `D3DRS_MULTISAMPLEANTIALIAS = FALSE` is ignored. No knob.
