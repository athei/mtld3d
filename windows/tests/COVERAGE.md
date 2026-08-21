# mtld3d end-to-end test coverage

Every test is one isolated `#[test]` driving the real `d3d9.dll` through the
shared [`Harness`](src/harness.rs), verified by pixel readback, `HRESULT`, or a
getter round-trip — no manual inspection. Run with `make test` (nextest runs
each test in its own Wine process, in parallel, on both `i686`/`x86_64`
windows-msvc; the host-native `mtld3d-core`/`mtld3d-shared` unit tests run too).

## Covered behaviour by file

| File | Coverage |
| --- | --- |
| `smoke.rs` | Clear-to-colour fill; `DrawPrimitiveUP` triangle with interpolated diffuse. |
| `device.rs` | `Direct3DCreate9`; adapter count/identifier/display-mode; `GetAdapterModeCount`/`EnumAdapterModes` (valid + out-of-range); `CheckDeviceType`/`CheckDeviceFormat`/`CheckDeviceFormatConversion` (accept + reject); `GetDeviceCaps` sanity (SM2 sub-structs at the ps_2_0 floor, cube/volume filter and address caps, no VTF and `QUERY_VERTEXTEXTURE` rejected to match); `TestCooperativeLevel`; `Reset` (0×0 reject, same-size state-default restore, resize, fullscreen monitor-rect + style adoption and restore, fullscreen ignores the requested resolution, undrawn post-resize `Present` reads back black). |
| `clear_present.rs` | (folded into smoke/device — clear flags exercised via `clear`). |
| `draw.rs` | XYZRHW screen-space quad; every accepted primitive type (point/line/linestrip/tristrip); triangle fans through `DrawPrimitiveUP`, `DrawPrimitive` (bound stream, `StartVertex`, and a 300 triangle fan that outgrows the encoder's shared index pattern mid-frame) and `DrawIndexedPrimitive` (bound 16-bit indices, `StartIndex` + `BaseVertexIndex`), all rewritten as triangle lists; `DrawIndexedPrimitiveUP`; `ProcessVertices` stubs. |
| `points.rs` | `D3DRS_POINTSIZE` sizing the square; an XYZ-only point under an ortho projection (Wine's test_pointsize shape); several sizes in one scene with a Clear between scenes and no Present; `POINTSIZE_MIN`/`MAX` clamp; `D3DFVF_PSIZE` per-vertex size; `POINTSCALEENABLE` with the viewport-height factor and eye-distance attenuation; programmable VS `oPts` vs the render-state default; point sprites through fixed-function and `ps_2_0` texcoords (quadrant texture), and the sprite state leaving triangles alone. |
| `clip_planes.rs` | Fixed-function plane keeping the positive side and released when disabled; world-space semantics under a translated view; `D3DRS_CLIPPING` gating; two sparse-index planes intersecting; RHW geometry ignoring the planes; clip-space semantics under a programmable VS; `D3DSBT_ALL` capture/apply round trip that renders. |
| `buffers.rs` | `CreateVertexBuffer`/`CreateIndexBuffer`; `DrawPrimitive`/`DrawIndexedPrimitive` from bound streams; DYNAMIC+DISCARD refill; `GetDesc` round-trips; `GetStreamSource`/`GetIndices` round-trips including higher streams and the NULL-bind offset/stride retention. |
| `streams.rs` | A two-stream declaration through a programmable VS; the `SetStreamSourceFreq` contract (defaults, rejections leaving state untouched, flag round-trip); instanced indexed draws (count from stream 0, per-instance step rate, non-indexed draws never instance, no per-instance stream means one instance); recorded and `D3DSBT_ALL` state blocks restoring stream bindings and frequencies. |
| `render_states.rs` | Alpha + additive blend; COLORWRITEENABLE mask; scissor; cull-mode winding; defaults vs `render_state_defaults()`; set/get round-trip; stencil round-trip; stencil test gating a draw; stencil clear preserving depth; combined depth+stencil mid-frame clear resetting both planes; stencil reference compared through the mask; stencil clear and test against a depth-only surface; wireframe no-op (pinned).; a Clear after a draw in the same pass (clear quad) surviving the draw's cull mode. |
| `textures.rs` | Lock/sample A8R8G8B8/X8R8G8B8/R5G6B5/A1R5G5B5/A4R4G4B4/L8; DXT1 block decode; mip chain levels/dims; AUTOGENMIPMAP; SetLOD no-op; cube creation in all pools; CPU-only extension-format cubes; managed DXT face isolation; cube face upload, sampling, state blocks, render targets, and AUTOGENMIPMAP; volume creates. |
| `samplers.rs` | State round-trip; CLAMP≠WRAP past the unit square; POINT≠LINEAR; BORDER → Metal black preset (pinned). |
| `texture_stages.rs` | COLOROP round-trip; MODULATE/ADD/SELECTARG2; TFACTOR arg source. |
| `shaders.rs` | hand-assembled VS/PS; PS-constant colour; VS-constant translation; float-constant setters (in-range accept + out-of-range/`-1` → `INVALIDCALL`); integer/bool + Get*Constant* stubs. |
| `vertex_decl.rs` | `CreateVertexDeclaration` drives an FF draw; `GetVertexDeclaration` round-trip; a two-stream declaration through the FF pipeline; a declared stream with nothing bound reads zeros (bound and UP draws). |
| `transforms_ff.rs` | Set/Get/MultiplyTransform; FF diffuse passthrough; alpha test; Set/Get material + light + LightEnable. |
| `render_target.rs` | Render-to-texture + sample; depth occlusion; auto depth-stencil Get/Set; CreateDepthStencilSurface; backbuffer desc; StretchRect 1:1 accept; INTZ sampleable-depth dual-use (render-as-depth → sample) via both the FF and a programmable PS; `GetRenderTargetData` read-back into a SYSTEMMEM offscreen surface (`Surface::GetDevice` + `CreateOffscreenPlainSurface` + `LockRect`, pixels matched against the private export); surface-op contracts (ColorFill/CreateRenderTarget stubs, DEFAULT-pool rejection). |
| `mrt.rs` | Multiple render targets: `ps_3_0` writing `oC0`/`oC1` into two bound targets; `Clear` reaching every bound target and an unwritten target keeping its contents; `D3DRS_COLORWRITEENABLE1` masking slot 1 alone; the slot contract (four slots, slot 0 never null, `NOTFOUND` on an unbound slot, `SetRenderTarget(1, NULL)`); `Reset` unbinding slots 1..3; a target sized unlike slot 0 cleared but left out of draws; a mid-pass `Clear` and a rect `Clear` reaching both targets through the in-pass quad; the MRT caps bits. |
| `state_block.rs` | Capture/Apply (ALL); VERTEXSTATE restores FVF; PIXELSTATE restores sampler; Begin/EndStateBlock recording. |
| `query.rs` | EVENT fence; OCCLUSION sample count; TIMESTAMP contract. |
| `resource_misc.rs` | Factory refcount; `QueryInterface` → E_NOINTERFACE; `GetType`; no-op PreLoad/SetPriority; `GetAvailableTextureMem`; `EvictManagedResources`; `GetDevice`/`SetClipPlane` stubs; `ValidateDevice` → S_OK (single-pass valid); `SetGammaRamp` no-op. |
| `unload.rs` | `LoadLibrary` → `Direct3DCreate9` → `Release` → `FreeLibrary`, then a continuable exception raised with a resuming handler appended at the end of the chain: proves the unloaded image left no vectored exception handler behind (no harness: its `raw-dylib` import would keep the module mapped). |

## Documented limitations / stubs pinned by tests

These return `D3DERR_INVALIDCALL` (or are no-ops) by design — the target
workload does not need them, or Metal cannot represent them. Tests pin the
contract so a future implementation flips a known assertion.

- **Draw:** `DrawIndexedPrimitiveUP`, `ProcessVertices`.
- **Textures:** ATI1 and YUV cubes are CPU-only `D3DPOOL_SCRATCH` resources;
  GPU-backed cubes support mapped color and DXT formats. `SetLOD` is a
  managed-pool-only no-op.
- **Samplers:** arbitrary `D3DSAMP_BORDERCOLOR` (Metal has 3 preset borders).
- **Shaders:** integer/bool constant setters; `Get{Vertex,Pixel}ShaderConstantF`.
- **Surfaces:** `CreateRenderTarget` (use `CreateTexture(D3DUSAGE_RENDERTARGET)`)
  remains a stub. `ColorFill` fills DEFAULT-pool render-target *texture* surfaces
  (A8R8G8B8/X8R8G8B8/R32F); standalone surfaces + other formats are not yet
  covered. `CreateOffscreenPlainSurface` is implemented for
  `D3DPOOL_SYSTEMMEM` only (DEFAULT/MANAGED rejected); `GetRenderTargetData` /
  `GetFrontBufferData` read a backbuffer / standalone-color RT back into a
  SYSTEMMEM surface (texture-backed RT sources not yet resolved).
- **Resources:** `GetDevice` is implemented on surfaces but still stubbed on the
  other resource types (VB/IB/textures/shaders/queries). `SetClipPlane` is a
  stub; `ValidateDevice` returns S_OK (single-pass valid); `SetGammaRamp` is a
  no-op.
- **Legacy:** `SetPrivateData`, `SetPaletteEntries`, `GetRasterStatus`,
  `GetClipStatus`, `SetDialogBoxMode`.

INTZ sampleable-depth (cascade shadows): covered synthetically — the depth
texture is rendered into, then sampled in a later pass (depth unbound first so it
is not a live attachment), through both the fixed-function pixel pipeline and a
hand-assembled programmable PS. A `Depth32Float` slot emits `depth2d<float>` +
`sample_compare`; `make test` runs with Metal validation on, so a
depth/`texture2d` or attachment/format mismatch fails the test.
