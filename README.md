# mtld3d

Direct3D 9 translation layer for Wine on macOS, backed by Metal.

mtld3d replaces Wine's built-in `d3d9.dll` with an implementation that
translates D3D9 calls through Wine's PE/Unix boundary into Metal command
buffers on the host. The pure-Rust core (`mtld3d-core`) handles DXSO → MSL
shader translation, render-pass scheduling, and fixed-function state.

## Goal

mtld3d aims to be the **fastest** Direct3D 9 implementation for Wine on macOS.
Direct3D 8, on the same core, is planned and part of that goal. Every other
Direct3D version is a non-goal; D3D10/11/12 are already well served on macOS by
Apple's D3DMetal and by DXMT.

Conformance serves that goal rather than defining it: the implementation is
developed against Wine's d3d9 test suite and every divergence it reports is
triaged and written down (see [Testing](#testing)). But where matching D3D9
exactly would cost frame time, speed wins, as long as the divergence does not
keep a game from running. Those trades are listed under
[Faster than conformant](#faster-than-conformant), and where a knob makes sense
they are revertible from `mtld3d.conf`.

Apple Silicon is what is developed and tested today. Intel Macs are a goal, not
a claim: the non-uniform-memory paths are written but unverified on real
hardware.

## Installation

Download the release bundle (`mtld3d.tar.xz`) from
[GitHub Releases](https://github.com/athei/mtld3d/releases).

[`INSTALL.md`](INSTALL.md), which also ships inside the bundle, has the
requirements and the walkthroughs for both routes: **builtin**, which drops
into a Wine installation you own and replaces the stock d3d9 for that whole
tree, and **native override**, loaded per game through `d3d9=native` while the
Wine installation stays untouched, which is what
[CrossOver](https://www.codeweavers.com/crossover) bottles use. It also covers
[x87sidecar](https://github.com/athei/x87sidecar), worth running under an
x86_64 Wine because Rosetta 2 is slow at the x87 math D3D9-era games do.

## Status

### Implemented

- **Programmable pipeline**: vertex and pixel shader models 1.x through 3.0
  (DXSO bytecode → MSL translation), including vPos/vFace, flat shading, and
  the D3D9 half-pixel rasterization convention.
- **Fixed-function pipeline**, vertex and pixel: lighting (directional, point,
  spot), texture-coordinate generation, the full texture-stage cascade,
  material color sources, hardware vertex blending.
- **Fog**: vertex fog and per-pixel table fog, across the fixed-function,
  pre-transformed (RHW), and programmable paths.
- **All four draw paths**: DrawPrimitive / DrawIndexedPrimitive and both UP
  variants, every primitive type including triangle fans (rewritten as
  triangle lists, which Metal lacks).
- **Points**: `D3DRS_POINTSIZE`, per-vertex PSIZE and `oPts`, the min/max
  clamp, eye-distance scaling, and point sprites through `[[point_coord]]`.
- **User clip planes**: six planes through `[[clip_distance]]`, world-space
  for the fixed-function pipeline and clip-space for vertex shaders, gated by
  `D3DRS_CLIPPING` and carried by `D3DSBT_ALL` state blocks.
- **Vertex streams**: all sixteen `SetStreamSource` streams feed a draw, with
  per-stream offsets and strides; a declared stream with nothing bound reads
  zeros, as on hardware.
- **Hardware instancing**: `SetStreamSourceFreq` with `INDEXEDDATA` /
  `INSTANCEDATA`, including per-instance step rates, on the indexed draws
  (D3D9 never instances a non-indexed draw).
- **State blocks**: recorded (Begin/End) and D3DSBT_* snapshots.
- **Queries**: occlusion queries backed by real Metal visibility results; event
  queries.
- **Resources**: DXT1–5 and ATI1 compressed textures, the common uncompressed
  integer and float formats, cube and volume textures, mipmap auto-generation,
  managed-pool dirty-region uploads, StretchRect (including cross-format blits
  via a conversion pass and YUY2/UYVY to RGB decoding), GetDC read-back. The
  two system-memory pools, `D3DPOOL_SYSTEMMEM` and `D3DPOOL_SCRATCH`, hold
  their bytes CPU-side and allocate no Metal texture, so a resource that
  only Locks, feeds `UpdateTexture` / `UpdateSurface` or receives
  `GetRenderTargetData` costs no video memory; binding one for sampling,
  which D3D9 permits, allocates one then.
- **Depth**: sampleable depth textures (INTZ, DF16, DF24) with hardware
  shadow-compare PCF, depth bias and slope-scale bias, depth clamp for
  pre-transformed geometry.
- **Stencil**: the full test, both faces, and every stencil operation, with
  `Clear(D3DCLEAR_STENCIL)` bounded to the viewport independently of depth.
- **Sampling and output**: anisotropic filtering, per-sampler mipmap LOD bias
  (Metal has none, so it is applied at the sample site), sRGB read (32-bit and
  compressed formats) and sRGB write, alpha test, scissor, separate alpha
  blend, blend factor, color write masks. `D3DRS_SRGBWRITEENABLE` binds the
  render target's sRGB view, back buffer included, so the encode happens after
  the blender as the `D3DPMISCCAPS_POSTBLENDSRGBCONVERT` cap promises; a target
  whose format has no sRGB Metal counterpart keeps a pixel-shader encode, which
  is exact only for opaque draws.
- **Multiple render targets**: four simultaneous targets with independent
  formats and write masks, post-pixel-shader blending on each, and Clear
  reaching every bound target.
- **Multisampling**: `CheckDeviceMultiSampleType` reports the sample counts the
  Metal device accepts (2x and 4x everywhere, 8x where the device offers it),
  with `D3DMULTISAMPLE_NONMASKABLE` mapped onto the same ladder. A multisampled
  swap chain, `CreateRenderTarget` or `CreateDepthStencilSurface` renders into a
  multisampled attachment and resolves into the single-sample surface every
  other operation reads, so Present, `StretchRect`, `GetRenderTargetData` and
  sampling all see the resolved image. `D3DRS_MULTISAMPLEMASK` narrows the
  samples a draw writes.
- **Presentation**: windowed and fullscreen swap chains, adapter mode
  enumeration, hardware color cursors, and MetalFX upscaling on the way to the
  screen with `render.scale` choosing the render resolution. A render target or
  depth-stencil the game creates at the reported back-buffer size rasterizes at
  that same scale, so a depth resolve into an INTZ texture stays a same-size
  copy. Fullscreen never changes the display mode; what it does instead is
  under [Display-mode switching](#deliberately-not-implemented).
- **HDR**: on an EDR-capable display the layer is upgraded to
  `extendedDynamicRange` and present routes through a BT.2446-A
  inverse-tone-mapping pass in ICtCp, its peak following the display's live
  headroom. On by default; `color.hdr.enable` and `color.space` control it.

### Not implemented yet

Missing features a D3D9 application can reasonably want. Each fails cleanly,
with an absent cap bit or a documented error return, so applications take their
own fallback paths instead of breaking:

- **Non-solid fill modes** (Metal has no native wireframe).
- **TIMESTAMP and the other niche query types**: creation reports NOTAVAILABLE,
  as the spec allows.
- **Scaled, sub-rect and format-converting depth→depth StretchRect**: the
  whole-surface 1:1 copy between two same-format DEFAULT-pool depth-stencil
  surfaces is implemented, including the resolve out of a multisampled source;
  a source or destination rect short of the full surface, a size mismatch, a
  differing depth format, or a multisampled destination returns INVALIDCALL.

### Faster than conformant

Divergences from D3D9 kept on purpose, because closing them costs frame time,
memory headroom, or the games that rely on the looser behaviour:

- **`IDirect3DTexture9::LockRect` serves a level of a `D3DPOOL_DEFAULT` 2D
  texture created without `D3DUSAGE_DYNAMIC`**, which D3D9 rejects through that
  entry point as well as through the level's `IDirect3DSurface9::LockRect`
  (here only the surface entry point rejects it; cube and volume locks reject
  it as D3D9 does), because a game that streams into a DEFAULT texture it never
  marked `D3DUSAGE_DYNAMIC` would otherwise lose every upload it makes that
  way. The cost is system memory: that texture class is the one whose per-level
  staging is released once its upload retires, so serving the lock re-creates
  the buffer, and a *partial* lock taken after such a release leaves the pixels
  outside its rect no longer matching the GPU copy (warned once per texture).
- **`GetData(D3DGETDATA_FLUSH)` returns `S_OK` immediately** for a pending
  occlusion query instead of blocking until the GPU has the count. Games use
  that poll loop as a fence against 2004-era drivers without hazard tracking,
  which Metal does explicitly, so the wait buys no correctness and stalls the
  API thread, the frame-time bottleneck. `query.flushImmediate = false`
  restores the spec-correct wait.
- **Depth stores are elided** where nothing reads the buffer back, so content
  that relies on depth surviving a pass it never cleared can read stale depth.
  Preserving it unconditionally would cost the optimization on every frame of
  every game that does clear, which is all of the tested ones.
- **A partial `Lock` of a dynamic vertex or index buffer hands back a pointer
  into memory a queued draw may still be reading**, unless the game passed
  `D3DLOCK_DISCARD` or locked the whole buffer. On D3D9 the runtime keeps the
  game's writes from landing under a draw the GPU has not reached yet, and
  `D3DLOCK_NOOVERWRITE` is how a game opts out of that protection; here it is
  the other way round, so a game that locks a sub-range and expects the
  runtime to manage the timing can get corrupted geometry for a frame, with
  nothing in the log. Matching D3D9 means either stalling until the draw
  retires or renaming the backing on every such `Lock`, and a dynamic buffer is
  precisely the one a UI or particle batcher locks dozens of times per frame.
  The rename-and-retain path those extra locks would run through has already
  been observed peaking around 1.4 GB of retained backings, which is why
  `memory.vbibRetentionCapMB` exists and why hitting that cap forces a
  mid-frame GPU sync. So this one trades memory headroom rather than frame
  time, and it has no knob: turning it on is the memory growth. Wine's d3d9
  test suite probes this and reports it in some runs, so the rationale is
  written up in [`CONFORMANCE.md`](unix/conformance/CONFORMANCE.md) like every
  other kept divergence.
- **A partial `LockRect` of a texture level hands back a pointer into staging
  an upload may still be reading**, unless the game passed
  `D3DLOCK_NOOVERWRITE` or `D3DLOCK_READONLY`. The same trade as the buffer
  entry above, on the same kind of caller: a font atlas or a lightmap page
  written a few rectangles at a time, each rectangle uploaded by the next
  draw. D3D9 would stall or rename until that upload retires; here the write
  lands in place, and a game whose rectangle overlaps an upload the GPU has not
  reached yet can see a frame of wrong texels, with nothing in the log. The
  `in-place` row of the perf grid's texture section counts the arm firing. A
  whole-level lock is not part of this: it is renamed, and its contents are
  preserved whatever the texture's usage says, because a game that locks a
  whole dynamic page to rewrite a few blocks of it relies on that (Half-Life
  2's lightmap pages under animated light styles).
- **A `D3DPOOL_DEFAULT` `D3DUSAGE_WRITEONLY` static vertex or index buffer keeps
  no CPU copy of its contents** once an upload has carried every byte to the
  GPU. D3D9 preserves a buffer's contents across a plain `Lock` whatever its
  usage said, so a title that locks such a buffer and reads back through the
  pointer sees zeros rather than what it wrote, and one that writes past the
  window it announced loses the bytes outside it; the log carries a warning the
  first time a backing is released and again the first time a lock lands on a
  re-created one. What it buys is the reason the divergence is here: inside a
  large-address-aware i386 title those copies have been measured near a gigabyte
  of the same 4 GiB the title needs for its own data, and running out of it
  crashes the process. `buffer.ignoreLockBounds` keeps the copy for a title that
  provably writes outside its announced windows. `DrawIndexedPrimitive` on a
  triangle fan is the one path that still needs an index buffer's bytes on the
  CPU, because Metal has no fan primitive and the fan is rewritten as a triangle
  list; a released index buffer has them copied back off the GPU, which costs
  one mid-frame submit and one GPU wait, and the copy is then held for the
  buffer's life so no buffer stalls twice.

- **`D3DRS_MULTISAMPLEANTIALIAS = FALSE` is ignored.** The state asks the
  rasterizer to drop to a single sample for one draw on a multisampled target.
  Metal ties a pipeline's `rasterSampleCount` to the sample count of the pass's
  attachments and offers no per-draw override, so honouring it would mean
  rendering the draw into a separate single-sampled target and compositing it
  back. `D3DPRASTERCAPS_MULTISAMPLE_TOGGLE` is not advertised, which is how
  D3D9 tells an application the toggle is unavailable, and the first write is
  logged.

### Deliberately not implemented

- **D3D9Ex**: no Direct3DCreate9Ex, no shared resource handles, no D3D9On12.
  The extended interface is a different contract (device removal, OS-managed
  memory) built for the Vista+ compositor; the games this project targets are
  plain D3D9.
- **Display-mode switching**: a fullscreen device owns its window but never the
  desktop mode, which Wine's mac driver would set through
  `CGDisplaySetDisplayMode`, rearranging every other window on the screen. The
  window goes borderless over the monitor instead, while the back buffer keeps
  the display mode the game picked, exactly as a real mode-set would leave it,
  and present scales the frame to the display (MetalFX when enlarging). A
  request matching no enumerable mode, which a real mode-set would reject,
  follows the window instead: games that ask for such sizes derived them from
  their window and keep sizing their rendering and input from it.
- **The fullscreen focus lifecycle**: no device loss on deactivation, no
  focus-window subclassing, no synthesized activation messages. Presentation is
  a composited Metal layer and no exclusive mode is ever taken, so a lost
  display is never a lost device: `TestCooperativeLevel` reports `D3D_OK`
  across a focus change and only ever reports `D3DERR_DEVICENOTRESET`, which is
  real, after a failed `Reset`. Reporting a loss that did not happen would make
  every fullscreen game release and rebuild its whole `D3DPOOL_DEFAULT` working
  set on each activation change, for a device that lost nothing.
- **Software paths**: no reference or software rasterizer, no software vertex
  processing, no RegisterSoftwareDevice. HAL on the default Metal device is the
  only device type; multi-adapter setups are not enumerated. ProcessVertices
  transforms through the current FVF; an explicit vertex-declaration source is
  rejected.
- **Legacy remnants**: N-patch/RT-patch tessellation, vertex tweening,
  palettized textures, gamma ramp. Dead features in real-world content,
  accepted or rejected per spec but non-functional.

### Testing

mtld3d is developed and tested against **World of Warcraft 1.12 and 3.3.5a**
under Wine and CrossOver. No other games have been exercised yet; reports are
welcome.

Beyond the game workloads it is hardened against **Wine's d3d9 test suite**,
the de-facto D3D9 conformance suite. `make conformance` runs it against the
installed builtin, one runner process per PE architecture, and gates on a
per-site baseline; every remaining divergence is classified with a written
rationale, the ones kept for speed included, in
[`unix/conformance/CONFORMANCE.md`](unix/conformance/CONFORMANCE.md). The unit
and end-to-end suites (`make test`) run the pure-Rust core natively on the host
and the full stack under Wine.

## Building from source

mtld3d builds and runs on **Apple Silicon macOS**. `mtld3d.so` ships as both an
x86_64 and an arm64 Mach-O, since Wine loads it from the `lib/wine/<cpu>-unix`
directory matching its own build; the PE side is x86 either way, translated by
**Rosetta 2** under an x86_64 Wine and by FEX under an arm64 one. The Metal
backend targets **macOS 15** or newer (`unix/.cargo/config.toml` pins
`MACOSX_DEPLOYMENT_TARGET = 15.0`).

Two things have to exist before a build: a **Wine** build or install providing
`wine`, `winebuild` and `wineserver` plus its development tree
(`lib/wine/{i386,x86_64}-windows/` and `libwinecrt0.a`), and **rustup**.
Everything else is `make setup`'s job.

`WINE_SDK` points at that Wine tree and must be exported before any target,
including `make setup`: the Makefile takes the Wine binaries from it by
absolute path, `windows/shim/build.rs` reads `libwinecrt0.a` and `ntdll.a`
there for linking, and `make conformance` finds Wine's d3d9 test binaries in
it. `WINE_INSTALL_DIR` is a second tree `make install` copies into, alongside
`WINE_SDK` itself.

```sh
make setup              # one-time toolchain bootstrap
make                    # build all (windows i386+x64 + unix)
make install            # install into both Wine trees
make bundle             # pack the distributable tarball + its debug symbols (PROD=1 by default)
make test               # unit tests on the host, e2e under Wine, one leg per PE arch
make conformance        # Wine's d3d9 suite vs the checked-in baseline, one leg per PE arch
make check              # the pre-commit gate: fmt + clippy + audit + doc
make fmt                # format all workspaces (requires nightly)
make clippy             # run clippy on all workspaces
make audit              # the conventions clippy can't express (see docs/CONVENTIONS.md)
make doc                # build the docs with rustdoc warnings denied
make clean              # cargo clean both workspaces
make upgrade            # cargo update (semver-compatible) in both workspaces
make upgrade-incompat   # cargo upgrade --incompatible + cargo update
```

Every leg is also its own CI job, with the toolchains and tools the Makefile
leaves floating for a developer pinned in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

`make setup` is the one-time bootstrap: both rustup toolchains (stable, 1.97 or
newer per `rust-version` in the manifests, for everything; nightly only for
`make fmt`, whose `rustfmt.toml` uses nightly-only options) with the four
cross-compilation targets, the cargo tools, the PE
linker and archiver as symlinks onto the toolchain's own `rust-lld` and
`llvm-ar`, Rosetta 2 if it is missing, and the pinned Windows SDK splatted to
`/opt/xwin` (~3 GB, and one `sudo` prompt to create that root-owned directory).
It does not install Wine. The internal crates are path dependencies and are not
published to crates.io.

Frame pointers are off by default; `FP=1 make` forces them on for the guest-pc
sampling profiler, whose stack walks follow the guest frame-pointer chain.

`make install` copies the PE DLLs into `lib/wine/{i386,x86_64}-windows/` and
the `.so` into `lib/wine/{x86_64,aarch64}-unix/`, stamping the Wine-builtin
signature onto the `d3d9.dll` copies, because the loader ignores unsigned PEs
on the builtin search path. The build outputs themselves stay unsigned so they
can also serve as a native DLL override, and `make bundle` packs both flavors
into `windows/target/mtld3d.tar.xz`. Debug symbols travel with every binary, a
`.pdb` beside each PE and a `.dSYM` beside the `.so` (`make` runs `dsymutil`,
since Mach-O DWARF otherwise stays behind in the object files); `make bundle`
writes them as `windows/target/mtld3d-debug.tar.xz`, and every DLL logs its
release and linker-assigned image ID on load, so a crash report names the
archive that symbolicates it.

## Architecture

```
game.exe → d3d9.dll → mtld3d.dll → mtld3d.so
(PE, i386 or x64: one chain per arch)  (Mach-O, Wine's own arch)
```

- `d3d9.dll`: D3D9 API implementation, COM vtables, caps, state management.
- `mtld3d.dll`: PE shim that owns Wine's unix-call globals and exports `mtld3d_unix_call`.
- `mtld3d.so`: native macOS side, a pure Metal abstraction layer.
- `mtld3d-core`: host-testable pure-Rust rlib linked into `d3d9.dll`.

At runtime the frame flows through a three-thread pipeline:

```
API thread (the game's)     Encoder thread            Submit thread
───────────────────────     ──────────────            ─────────────
record frame N+1        →   encode frame N        →   submit + present frame N−1
```

- The **API thread** is the game's own render thread and the frame-time
  bottleneck, so it never waits on translation, Metal, or the GPU: each call
  only snapshots the state it needs into a closure on the frame's op list.
- The **encoder thread** (one per device) runs those closures: D3D9 → Metal
  translation, render-pass scheduling and load/store optimization, pipeline and
  sampler caches, lazy resource creation and texture uploads.
- The **submit thread** crosses the PE/Unix boundary to replay the finished
  command stream, waits for the drawable, presents and commits.

Each hand-off has capacity one, so the pipeline never runs more than one frame
ahead per stage: backpressure, not queueing, bounds latency.

For the boundary contract, the threading details, perf instrumentation, and the
shader/heap debugging toolkits, see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Workspaces

Two Cargo workspaces, one per target platform: `windows/` builds the PE side
for `i686-pc-windows-msvc` and `x86_64-pc-windows-msvc`, `unix/` the Mach-O
side for `x86_64-apple-darwin` and `aarch64-apple-darwin` (the latter is also
the native test target). Open each in its own editor window for rust-analyzer
to work. The shipped crates (both workspaces also hold test, types, and
conformance support crates):

| Crate         | Workspace  | Output                                                  |
|---------------|------------|---------------------------------------------------------|
| `d3d9`        | `windows/` | `d3d9.dll`                                              |
| `mtld3d`      | `windows/` | `mtld3d.dll`                                            |
| `mtld3d-core` | `windows/` | rlib (linked into `d3d9.dll`)                           |
| `mtld3d-unix` | `unix/`    | `mtld3d.so`                                             |
| `shared`      | `unix/`    | rlib (shared by `d3d9.dll`, `mtld3d.dll`, `mtld3d.so`)  |

`mtld3d-core` holds every platform-independent helper (DXSO → MSL emission, the
render-pass state machine, the slab allocator, format / FVF / vertex-decl /
dirty-rect math, fixed-function state) and compiles for the macOS host as well
as PE, so `cargo test -p mtld3d-core --target aarch64-apple-darwin` runs its
unit tests natively instead of through Wine.

`unix/shared` is the crate every linkage unit depends on, primarily for the
PE↔Unix wire format (the `Command` enum, the `Thunks` enum, param structs,
typed `mtl::` wire values). Pure data and pure-Rust helpers only, no FFI and no
`#[link]`, so both workspaces can depend on it cleanly.

## Configuration and logging

User-facing runtime options live in the optional `mtld3d.conf`, read once at
`Direct3DCreate9` from the directory of the running `.exe`, so a change needs a
restart. The sample in the repo root documents every option, its default, and
the `MTLD3D_CONFIG` environment override that beats the file.

Below both sits a third layer: a handful of games need options nobody should
have to discover, so mtld3d ships built-in profiles for them. A profile is
matched on the executable name plus the version resource its vendor linked in,
so it follows the game wherever it is installed and never fires on an unrelated
program of the same name. It only supplies starting values, and both the file
and the environment override it key by key.
`RUST_LOG=mtld3d::d3d9=info` names the profile that matched.

| Profile | Application | What it sets and why |
|---|---|---|
| `gta-iv` | Grand Theft Auto IV | `adapter.spoof=amd`, because the renderer branches on the reported vendor and stalls in its own identifier parsing on the NVIDIA identity. `caps.dfFormats=false`, because with the DF fourccs advertised it picks a mixed DF24 plus INTZ depth path that no hardware of its era offered. `query.flushImmediate=false`, because its occlusion culling needs real pixel counts rather than an immediate answer. `depth.aliasSameSize=true`, because its late alpha, sky and glow passes z-test one INTZ depth texture against scene depth rendered into a same-size sibling. |

Every crate logs via `log` + `env_logger`. All targets sit under `mtld3d::*`
and `env_logger` matches by `::`-separated prefix, so `RUST_LOG=mtld3d=warn` is
the single switch for the whole project; narrow it per target, for example
`RUST_LOG=mtld3d=warn,mtld3d::unix=trace`.

| Target                  | Scope                                                                 |
|-------------------------|-----------------------------------------------------------------------|
| `mtld3d::d3d9`          | `windows/d3d9/` + `windows/core/` (everything except `dxso` and `perf`) |
| `mtld3d::d3d9::cursor`  | hardware cursor (HCURSOR) lifecycle, bitmap cache, wndproc            |
| `mtld3d::dxso`          | DXSO → MSL emitter                                                    |
| `mtld3d::perf`          | 5-second averaged performance summary (`PERF=1` builds only)          |
| `mtld3d::shim`          | Wine unix-call PE shim DLL                                            |
| `mtld3d::unix`          | Metal-side `.so`                                                      |

Levels: `info!` for one-shot milestones, `warn!` for unimplemented stubs and
fallback paths, `error!` for unexpected internal failures, `trace!` for
per-call breadcrumbs, `debug!` for routine per-call noise useful in deep
debugging.

Each cdylib initializes the logger independently and idempotently; `mtld3d.so`
has no owning entry point, so `d3d9.dll` dispatches a one-shot `InitLogger`
thunk from its init path.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) is the operating manual: what to read
before changing anything, the gates and how to read their output, how
conformance work is organised, and what a pull request is expected to contain.
The development conventions themselves live in
[`docs/CONVENTIONS.md`](docs/CONVENTIONS.md).

The short version: **`make check`** and **`make test`** are green before every
commit, and one pull request is one clearly defined change.

## License

[zlib](LICENSE).
