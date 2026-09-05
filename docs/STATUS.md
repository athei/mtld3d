# Status

What mtld3d implements, what it does not yet, what it never will, and the
divergences from D3D9 it keeps on purpose. The tested games are in the
[README](../README.md#tested-games); the end-to-end suite's coverage is in
[`COVERAGE.md`](../windows/tests/COVERAGE.md).

## Supported

- Shader models 1.x through 3.0, translated from DXSO to MSL and cached on
  disk by content hash.
- The fixed-function pipeline: lighting, texture-coordinate generation, the
  texture-stage cascade, vertex blending, vertex and table fog.
- Every draw call and primitive type, point sprites, user clip planes, all
  sixteen vertex streams, hardware instancing.
- State blocks, occlusion and event queries.
- Compressed (DXT1 to DXT5, ATI1), integer and float formats, cube and volume
  textures, auto-generated mipmaps, `StretchRect` with format conversion and
  YUV decoding, `GetDC`.
- Sampleable depth (INTZ, DF16, DF24) with shadow compare, depth bias, the
  full two-sided stencil test.
- Anisotropic filtering, LOD bias, sRGB read and write, alpha test, scissor,
  separate alpha blend, blend factor, write masks.
- Four render targets with independent formats and blending.
- Multisampling at 2x and 4x, 8x where the device offers it.
- Windowed and fullscreen swap chains, mode enumeration, hardware and software
  cursors, MetalFX upscaling, HDR output.
- `D3DCREATE_MULTITHREADED`: a device created with it, and every object it
  creates, may be called from any thread; each entry point holds a reentrant
  per-device lock, and a device created without the flag pays nothing.

## Not implemented yet

Each fails cleanly, with an absent cap bit or a documented error return.

- Non-solid fill modes: Metal has no wireframe, so the state is warned once
  and drawn solid.
- Timestamp and other niche query types: creation reports
  `D3DERR_NOTAVAILABLE`.
- Scaled, sub-rect or converting depth-to-depth `StretchRect`: only the
  whole-surface 1:1 copy between same-format DEFAULT-pool depth surfaces
  works, multisample resolve included.

## Deliberately not implemented

- D3D9Ex: no `Direct3DCreate9Ex`, shared handles or D3D9On12. A different
  contract, built for the Vista compositor.
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
- Depth stores are elided where nothing reads the buffer back. No knob.
- A partial `Lock` of a dynamic vertex or index buffer without
  `D3DLOCK_DISCARD` returns a pointer a queued draw may still read. No knob.
- A partial `LockRect` of a texture level without `D3DLOCK_NOOVERWRITE` or
  `D3DLOCK_READONLY` returns a pointer an upload may still read. No knob.
- A DEFAULT-pool `D3DUSAGE_WRITEONLY` static buffer keeps no CPU copy once
  uploaded, so a read through the lock pointer sees zeros.
  `buffer.ignoreLockBounds` keeps the copy.
- `D3DRS_MULTISAMPLEANTIALIAS = FALSE` is ignored. No knob.
