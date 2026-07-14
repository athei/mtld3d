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
| `device.rs` | `Direct3DCreate9`; adapter count/identifier/display-mode; `GetAdapterModeCount`/`EnumAdapterModes` (valid + out-of-range); `CheckDeviceType`/`CheckDeviceFormat`/`CheckDeviceFormatConversion` (accept + reject); `GetDeviceCaps` sanity; `TestCooperativeLevel`; `Reset` (0×0 reject, same-size state-default restore, resize). |
| `clear_present.rs` | (folded into smoke/device — clear flags exercised via `clear`). |
| `draw.rs` | XYZRHW screen-space quad; every accepted primitive type (point/line/linestrip/tristrip); triangle-fan + `DrawIndexedPrimitiveUP` + `ProcessVertices` stubs. |
| `buffers.rs` | `CreateVertexBuffer`/`CreateIndexBuffer`; `DrawPrimitive`/`DrawIndexedPrimitive` from bound streams; DYNAMIC+DISCARD refill; `GetDesc` round-trips; `GetStreamSource`/`GetIndices`/`SetStreamSourceFreq` + non-zero stream stubs. |
| `render_states.rs` | Alpha + additive blend; COLORWRITEENABLE mask; scissor; cull-mode winding; defaults vs `render_state_defaults()`; set/get round-trip; stencil round-trip + wireframe no-op (pinned). |
| `textures.rs` | Lock/sample A8R8G8B8/X8R8G8B8/R5G6B5/A1R5G5B5/A4R4G4B4/L8; DXT1 block decode; mip chain levels/dims; AUTOGENMIPMAP; SetLOD no-op; SCRATCH-pool cube creates (other pools reject); volume creates. |
| `samplers.rs` | State round-trip; CLAMP≠WRAP past the unit square; POINT≠LINEAR; BORDER → Metal black preset (pinned). |
| `texture_stages.rs` | COLOROP round-trip; MODULATE/ADD/SELECTARG2; TFACTOR arg source. |
| `shaders.rs` | hand-assembled VS/PS; PS-constant colour; VS-constant translation; float-constant setters (in-range accept + out-of-range/`-1` → `INVALIDCALL`); integer/bool + Get*Constant* stubs. |
| `vertex_decl.rs` | `CreateVertexDeclaration` drives an FF draw; `GetVertexDeclaration` round-trip. |
| `transforms_ff.rs` | Set/Get/MultiplyTransform; FF diffuse passthrough; alpha test; Set/Get material + light + LightEnable. |
| `render_target.rs` | Render-to-texture + sample; depth occlusion; auto depth-stencil Get/Set; CreateDepthStencilSurface; backbuffer desc; StretchRect 1:1 accept; INTZ sampleable-depth dual-use (render-as-depth → sample) via both the FF and a programmable PS; `GetRenderTargetData` read-back into a SYSTEMMEM offscreen surface (`Surface::GetDevice` + `CreateOffscreenPlainSurface` + `LockRect`, pixels matched against the private export); surface-op contracts (ColorFill/CreateRenderTarget stubs, DEFAULT-pool rejection). |
| `state_block.rs` | Capture/Apply (ALL); VERTEXSTATE restores FVF; PIXELSTATE restores sampler; Begin/EndStateBlock recording. |
| `query.rs` | EVENT fence; OCCLUSION sample count; TIMESTAMP contract. |
| `resource_misc.rs` | Factory refcount; `QueryInterface` → E_NOINTERFACE; `GetType`; no-op PreLoad/SetPriority; `GetAvailableTextureMem`; `EvictManagedResources`; `GetDevice`/`SetClipPlane` stubs; `ValidateDevice` → S_OK (single-pass valid); `SetGammaRamp` no-op. |

## Documented limitations / stubs pinned by tests

These return `D3DERR_INVALIDCALL` (or are no-ops) by design — the target
workload does not need them, or Metal cannot represent them. Tests pin the
contract so a future implementation flips a known assertion.

- **Draw:** `DrawIndexedPrimitiveUP`, `ProcessVertices`; `D3DPT_TRIANGLEFAN`
  (no Metal fan primitive).
- **Buffers:** `GetStreamSource`, `GetIndices`, `SetStreamSourceFreq`;
  `SetStreamSource` on stream ≠ 0.
- **Render states:** stencil func/ops not plumbed to Metal (states round-trip
  but do not gate rendering); `D3DFILL_WIREFRAME` renders solid.
- **Textures:** `CreateCubeTexture` creates `D3DPOOL_SCRATCH` cubes only (a
  CPU-only, creatable/releasable shell — no `MTLTexture`, per-face lock/upload
  and sampling unwired); other pools reject (no `D3DPTEXTURECAPS_CUBEMAP`).
  `SetLOD` is a managed-pool-only no-op.
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
