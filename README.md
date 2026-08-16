# mtld3d

Direct3D 9 translation layer for Wine on macOS, backed by Metal.

mtld3d replaces Wine's built-in `d3d9.dll` with an implementation that translates D3D9 calls through Wine's PE/Unix boundary into Metal command buffers on the host. The pure-Rust core (`mtld3d-core`) handles DXSO → MSL shader translation, render-pass scheduling, and fixed-function state.

## Installation

Download the release bundle (`mtld3d.tar.xz`) from
[GitHub Releases](https://github.com/athei/mtld3d/releases). It runs on Apple
Silicon macOS 15 or newer, inside Wine 8.0+ or CrossOver 24+ (64-bit prefix);
an x86_64 Wine additionally needs Rosetta 2. The bundle supports two install
routes:

- **Builtin**: the bundle's `wine/` tree mirrors `lib/wine/` with every PE
  builtin-marked. Dropped into a Wine installation you own, it replaces the
  stock d3d9 for that whole tree.
- **Native override**: the bundle's `native/` tree holds an unmarked
  `d3d9.dll`, loaded per game (or per prefix) through a `d3d9=native` DLL
  override while the Wine installation stays untouched. This is the route for
  [CrossOver](https://www.codeweavers.com/crossover) bottles, where it is
  self-contained in the bottle and survives CrossOver updates.

Step-by-step walkthroughs for stock Wine and CrossOver, and the tradeoffs
between the routes, live in [`INSTALL.md`](INSTALL.md), which also ships
inside the bundle.

### x87 performance

D3D9-era games do their floating-point math in x87 instructions, which
Rosetta 2 translates slowly. Under an x86_64 Wine, run the game together with
[x87sidecar](https://github.com/athei/x87sidecar), a JIT that replaces
Rosetta's x87 handling. Its cooperative attach mode requires a Wine that
performs the sidecar handshake at startup: the Wine builds from
[wine-build](https://github.com/athei/wine-build) carry that patch, which
lets the x87sidecar binary work without any entitlements.

## Status

mtld3d aims to be the most faithful Direct3D 9 implementation for Wine on
Apple Silicon: correctness against native D3D9 behaviour first, performance
second. Direct3D 8 support on the same core is planned. Every other Direct3D
version is a non-goal; D3D10/11/12 are already well served on macOS by
Apple's D3DMetal and by DXMT.

Whatever is not implemented is reported honestly, through capability bits or
the documented error returns, so applications take their own fallback paths
instead of breaking.

### Implemented

- **Programmable pipeline**: vertex and pixel shader models 1.x through 3.0
  (DXSO bytecode → MSL translation), including vPos/vFace, flat shading, and
  the D3D9 half-pixel rasterization convention.
- **Fixed-function pipeline**, vertex and pixel: lighting (directional,
  point, spot), texture-coordinate generation, the full texture-stage
  cascade, material color sources, hardware vertex blending.
- **Fog**: vertex fog and per-pixel table fog, across the fixed-function,
  pre-transformed (RHW), and programmable paths.
- **All four draw paths**: DrawPrimitive / DrawIndexedPrimitive and both UP
  variants.
- **State blocks**: recorded (Begin/End) and D3DSBT_* snapshots.
- **Queries**: occlusion queries backed by real Metal visibility results;
  event queries.
- **Resources**: DXT1–5 and ATI1 compressed textures, the common
  uncompressed integer and float formats, volume textures, mipmap
  auto-generation, managed-pool dirty-region uploads, StretchRect (including
  cross-format blits via a conversion pass), GetDC read-back.
- **Depth**: sampleable depth textures (INTZ, DF16, DF24) with hardware
  shadow-compare PCF, depth bias and slope-scale bias, depth clamp for
  pre-transformed geometry.
- **Sampling and output**: anisotropic filtering, sRGB read (compressed
  formats) and sRGB write, alpha test, scissor, separate alpha blend,
  blend factor, color write masks.
- **Presentation**: windowed and fullscreen swap chains, adapter mode
  enumeration, hardware color cursors. A fullscreen device takes its window
  borderless over the monitor without ever changing the display mode; the back
  buffer follows the window, and `render.scale` decides how many pixels are
  actually rendered before MetalFX upscales the result to the screen.

### Not implemented yet

Missing features a D3D9 application can reasonably want. Each fails cleanly,
with an absent cap bit or a documented error return, so applications fall
back:

- **Stencil**: the caps report no stencil support; stencil render states are
  ignored and stencil clears are skipped.
- **MSAA**: multisampled creates are rejected; CheckDeviceMultiSampleType
  only accepts D3DMULTISAMPLE_NONE.
- **Multiple render targets**: a single simultaneous render target is
  advertised.
- **Cube-map sampling**: the cap is off and cube textures are never
  GPU-backed. CreateCubeTexture in the CPU pools still returns a real,
  lockable cube texture (faces work via GetCubeMapSurface/LockRect), so
  applications that probe cube support degrade gracefully; DEFAULT-pool
  creates are rejected.
- **Point sprites** and non-solid fill modes (Metal has no native wireframe).
- **TIMESTAMP and the other niche query types**: creation reports
  NOTAVAILABLE, as the spec allows.
- **YUV conversion**: YUY2/UYVY surfaces can be created and locked, but no
  YUV→RGB blit is performed.
- Depth→depth StretchRect.

### Deliberately not implemented

- **D3D9Ex**: no Direct3DCreate9Ex, no shared resource handles, no D3D9On12.
  The extended interface is a different contract (device removal, OS-managed
  memory) built for the Vista+ compositor; the games this project targets are
  plain D3D9.
- **Display-mode switching**: a fullscreen device owns its window but never
  the desktop mode. Wine's mac driver would hand the request to
  `CGDisplaySetDisplayMode`, which reconfigures the user's screen and
  rearranges every other window, and avoiding that would depend on a Wine
  registry setting. A monitor-covering borderless window keeps `GetClientRect`,
  mouse input and the back buffer in agreement without any of it, so the
  resolution a game picks in fullscreen is ignored and `render.scale` controls
  the render resolution instead.
- **The fullscreen focus lifecycle**: the focus dance around a fullscreen
  device is not emulated — no device loss on deactivation, no focus-window
  subclassing, no synthesized activation messages. Presentation is a composited
  Metal layer, so a lost display is never a lost device.
- **Software paths**: no reference or software rasterizer, no software
  vertex processing, no ProcessVertices, no RegisterSoftwareDevice. HAL on
  the default Metal device is the only device type; multi-adapter setups are
  not enumerated.
- **Hardware instancing** (SetStreamSourceFreq): the renderer is
  single-stream by architecture.
- **Clip-plane application**: SetClipPlane state round-trips but planes are
  not applied on the GPU.
- **Legacy remnants**: N-patch/RT-patch tessellation, vertex tweening,
  palettized textures, gamma ramp. Dead features in real-world content,
  accepted or rejected per spec but non-functional.

### Testing

mtld3d is developed and tested against **World of Warcraft 1.12 and 3.3.5a**
under Wine and CrossOver. No other games have been exercised yet; reports are
welcome.

Beyond the game workloads, the implementation is hardened against **Wine's
d3d9 test suite**, the de-facto D3D9 conformance suite. A dedicated runner
(`make conformance`) executes it against the installed builtin on both PE
architectures and gates on a per-site tracked baseline in which every
remaining divergence is triaged and classified with a documented rationale.
The methodology, classification scheme, and current audit live in
[`unix/conformance/CONFORMANCE.md`](unix/conformance/CONFORMANCE.md). The
unit and end-to-end suites (`make test`) run the pure-Rust core natively on
the host and the full stack under Wine.

## Building from source

mtld3d builds and runs on **Apple Silicon macOS**. `mtld3d.so` ships as both an
x86_64 and an arm64 Mach-O, since Wine loads it from the `lib/wine/<cpu>-unix`
directory matching its own build. The PE side is x86 either way, translated by
**Rosetta 2** (install it with `softwareupdate --install-rosetta`) under an
x86_64 Wine and by FEX under an arm64 one. The Metal backend targets
**macOS 15** or newer (`unix/.cargo/config.toml` pins
`MACOSX_DEPLOYMENT_TARGET = 15.0`).

The following must be available on `PATH`:

- A **Wine** build or install providing `winebuild`, `wine`, `wineserver`, and
  `wineboot`, plus its development tree (`lib/wine/{i386,x86_64}-windows/` and
  `libwinecrt0.a`).
- **Homebrew**, which `make setup` uses to install LLVM and lld.
- A **rustup** toolchain: stable (1.97 or newer, per `rust-version` in the
  Cargo manifests) for builds, nightly for `make fmt` (`rustfmt.toml` uses
  nightly-only options).

Two environment variables drive the Makefile. `WINE_SDK` points at the Wine
development tree consumed by `windows/shim/build.rs` for `libwinecrt0.a` /
`ntdll.a` linking, and must be exported before running any target, including
`make setup`. `WINE_INSTALL_DIR` is where `make install` copies the built
binaries (typically the same path as `WINE_SDK`).

```sh
make setup              # install cross-compilation toolchain (first time only)
make                    # build all (windows i386+x64 + unix)
make install            # install to Wine distribution
make bundle             # pack the distributable tarball + its debug symbols (PROD=1 by default)
make test               # build, install, run i386 + x64 test binaries
make check              # the pre-commit gate: fmt + clippy + audit + doc
make fmt                # format all workspaces (requires nightly)
make clippy             # run clippy on all workspaces
make audit              # the conventions clippy can't express (see docs/CONVENTIONS.md)
make doc                # build the docs with rustdoc warnings denied
make clean              # cargo clean both workspaces
make upgrade            # cargo update (semver-compatible) in both workspaces
make upgrade-incompat   # cargo upgrade --incompatible + cargo update; requires cargo-edit
```

`make setup` installs LLVM and lld via Homebrew, adds the
`i686-pc-windows-msvc`, `x86_64-pc-windows-msvc`, `x86_64-apple-darwin`, and
`aarch64-apple-darwin` rustup targets, installs xwin and cargo-edit, and splats
the Windows SDK (~3 GB) to `/opt/xwin`; creating that root-owned directory takes
a one-time `sudo` prompt. It does not install Wine or Rosetta. Windows targets
cross-compile from macOS via `lld-link` + xwin (see
`windows/.cargo/config.toml`); `unix/` builds both Apple targets explicitly,
one per Wine host ISA, since Wine picks the `.so` matching its own arch. The
internal crates use path dependencies and are not published to crates.io.

Frame pointers are off by default. `FP=1 make` forces them on for the guest-pc
sampling profiler, whose stack walks follow the guest frame-pointer chain.

`make install` copies the PE DLLs into `lib/wine/{i386,x86_64}-windows/` and
the unix `.so` into `lib/wine/{x86_64,aarch64}-unix/` under `WINE_INSTALL_DIR`,
stamping the Wine-builtin signature onto the `d3d9.dll` copies (the loader ignores
unsigned PEs on the builtin search path). The build outputs themselves stay
unsigned native PEs so they can also be deployed as a native DLL override;
`make bundle` packs both flavors into `windows/target/mtld3d.tar.xz`, the
release bundle described in [`INSTALL.md`](INSTALL.md).

Debug symbols travel with each binary: a `.pdb` beside every PE, and a `.dSYM`
beside the `.so` (on Mach-O the DWARF otherwise stays behind in the compiler's
object files, so `make` runs `dsymutil` to gather it). `make install` copies
both, and `make bundle` writes a second archive,
`windows/target/mtld3d-debug.tar.xz`, holding the symbols for exactly that
bundle. Every DLL logs its release and its linker-assigned image ID on load, so
a crash report from a release build names the debug archive that symbolicates
it.

## Architecture

```
test.exe → d3d9.dll → mtld3d.dll → mtld3d.so
(i386 PE)  (i386 PE)  (i386 PE)  (Mach-O, Wine's own arch)

test.exe → d3d9.dll → mtld3d.dll → mtld3d.so
(x64 PE)   (x64 PE)   (x64 PE)   (Mach-O, Wine's own arch)
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
  bottleneck, so it never waits on translation, Metal, or the GPU. Each D3D9
  call only snapshots the state it needs into a closure on the current
  frame's op list; `Present` hands the finished frame to the encoder and
  immediately starts recording the next.
- The **encoder thread** (one per device) runs the closures: D3D9 → Metal
  translation, render-pass scheduling and load/store optimization, pipeline
  and sampler caches, lazy resource creation and texture uploads.
- The **submit thread** crosses the PE/Unix boundary to replay the encoder's
  finished command stream into Metal encoders, waits for the drawable, and
  presents and commits, overlapping the encoder's build of the next frame.

Each hand-off has capacity one, so the pipeline never runs more than one
frame ahead per stage: backpressure, not queueing, bounds latency.

For the PE/Unix boundary contract, the threading details, perf instrumentation, and the shader/heap debugging toolkits, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Workspaces

Two Cargo workspaces, one per target platform: `windows/` builds the PE side
for `i686-pc-windows-msvc` and `x86_64-pc-windows-msvc`, `unix/` builds the
Mach-O side for `x86_64-apple-darwin` and `aarch64-apple-darwin` (one per Wine
host ISA; the latter is also the native test target). Open each in a separate
editor window for rust-analyzer to work correctly. The shipped crates and their
outputs (the workspaces also hold test, types, and conformance support crates):

| Crate         | Workspace  | Output                                                  |
|---------------|------------|---------------------------------------------------------|
| `d3d9`        | `windows/` | `d3d9.dll`                                              |
| `mtld3d`      | `windows/` | `mtld3d.dll`                                            |
| `mtld3d-core` | `windows/` | rlib (linked into `d3d9.dll`)                           |
| `mtld3d-unix` | `unix/`    | `mtld3d.so`                                             |
| `shared`      | `unix/`    | rlib (shared by `d3d9.dll`, `mtld3d.dll`, `mtld3d.so`)  |

`mtld3d-core` is a pure-Rust rlib holding every platform-independent helper:
DXSO → MSL emission, the render-pass state machine, the slab allocator,
format / FVF / vertex-decl / dirty-rect math, fixed-function state. It
compiles on both Windows PE and the macOS host, so
`cargo test -p mtld3d-core --target aarch64-apple-darwin` runs unit tests
natively instead of through Wine.

`unix/shared` is the crate every linkage unit depends on. Its primary role is
the PE↔Unix wire format (the `Command` enum, the `Thunks` enum, param
structs, typed `mtl::` wire values). Pure data and pure-Rust helpers only, no
FFI and no `#[link]`, so both workspaces can depend on it cleanly.

## Configuration and logging

User-facing runtime options live in `mtld3d.conf` (key = value, comments are
`#`-prefixed). mtld3d reads it once at `Direct3DCreate9` from the directory
of the running `.exe`; restart the game to pick up edits. A missing file is
fine, defaults apply. The repo root holds a sample with every option, its
default, and a short explanation.

Every key is also overridable at process launch via `MTLD3D_CONFIG`, a
semicolon-separated list of `key=value` entries using the same syntax as the
file. Env entries are merged on top of the file, and env wins on conflict.
Resolved options are logged once at startup under
`RUST_LOG=mtld3d::d3d9=info`; unknown keys, malformed lines, and unparseable
values warn once instead of silently falling back to the default.

Every crate logs via `log` + `env_logger`. All targets sit under `mtld3d::*`
and `env_logger` matches by `::`-separated prefix, so `RUST_LOG=mtld3d=warn`
is the single switch for the whole project. Examples: `RUST_LOG=mtld3d=warn`,
`RUST_LOG=mtld3d::d3d9=trace`, `RUST_LOG=mtld3d=warn,mtld3d::unix=trace`,
`RUST_LOG=mtld3d::perf=info`.

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

Each cdylib initializes the logger independently and idempotently;
`mtld3d.so` has no owning entry point, so `d3d9.dll` dispatches a one-shot
`InitLogger` thunk from its init path. The `mtld3d::perf` summary and its
counter aggregation rules are documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Contributing

The development conventions (module layout, visibility, encapsulation,
`unsafe` discipline, warning suppressions, the `log_once_warn!` rule,
doc-comment shape, typed `objc2-*` bindings) live in
[`docs/CONVENTIONS.md`](docs/CONVENTIONS.md).

Run **`make check`** before every commit: `cargo fmt --check`, clippy with
`nursery` and `pedantic` enabled, `make audit` (the rules clippy cannot
express), and `make doc` (rustdoc, so doc links have to resolve). The check
legs deny every warning via cargo's `build.warnings = "deny"`; normal builds
and a plain `cargo clippy` only warn. Each audit finding names the section of
`docs/CONVENTIONS.md` it comes from.

## License

[zlib](LICENSE).
