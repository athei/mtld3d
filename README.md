# mtld3d

Direct3D 9 translation layer for Wine on macOS, backed by Metal.

mtld3d replaces Wine's built-in `d3d9.dll` with an implementation that translates D3D9 calls through Wine's PE/Unix boundary into Metal command buffers on the host. The pure-Rust core (`mtld3d-core`) handles DXSO → MSL shader translation, render-pass scheduling, and fixed-function state.

## Status

mtld3d aims to be the most faithful Direct3D 9 implementation available for
Wine on Apple Silicon — correctness against native D3D9 behaviour first,
performance second. Direct3D 8 support on the same core is a planned goal.
Every other Direct3D version is an explicit non-goal: D3D10/11/12 are already
well served on macOS by Apple's D3DMetal and by DXMT, and there is nothing
useful for mtld3d to add there.

Whatever is not implemented is reported honestly — through capability bits or
the documented error returns — so applications take their own fallback paths
instead of breaking.

### Implemented

- **Programmable pipeline** — vertex and pixel shader models 1.x through 3.0
  (DXSO bytecode → MSL translation), including vPos/vFace, flat shading, and
  the D3D9 half-pixel rasterization convention.
- **Fixed-function pipeline**, vertex and pixel — lighting (directional,
  point, spot), texture-coordinate generation, the full texture-stage
  cascade, material color sources, hardware vertex blending.
- **Fog** — vertex fog and per-pixel table fog, across the fixed-function,
  pre-transformed (RHW), and programmable paths.
- **All four draw paths** — DrawPrimitive / DrawIndexedPrimitive and both UP
  variants.
- **State blocks** — recorded (Begin/End) and D3DSBT_* snapshots.
- **Queries** — occlusion queries backed by real Metal visibility results;
  event queries.
- **Resources** — DXT1–5 and ATI1 compressed textures, the common
  uncompressed integer and float formats, volume textures, mipmap
  auto-generation, managed-pool dirty-region uploads, StretchRect (including
  cross-format blits via a conversion pass), GetDC read-back.
- **Depth** — sampleable depth textures (INTZ, DF16, DF24) with hardware
  shadow-compare PCF, depth bias and slope-scale bias, depth clamp for
  pre-transformed geometry.
- **Sampling and output** — anisotropic filtering, sRGB read (compressed
  formats) and sRGB write, alpha test, scissor, separate alpha blend,
  blend factor, color write masks.
- **Presentation** — windowed and borderless-fullscreen swap chains, adapter
  mode enumeration, hardware color cursors.

### Not implemented yet

Missing features a D3D9 application can reasonably want; each fails cleanly
(absent cap bit or documented error return) so applications fall back:

- **Stencil** — the caps report no stencil support; stencil render states are
  ignored and stencil clears are skipped.
- **MSAA** — multisampled creates are rejected; CheckDeviceMultiSampleType
  only accepts D3DMULTISAMPLE_NONE.
- **Multiple render targets** — a single simultaneous render target is
  advertised.
- **Cube-map sampling** — the cap is off and cube textures are never
  GPU-backed. CreateCubeTexture in the CPU pools still returns a real,
  lockable cube texture (faces work via GetCubeMapSurface/LockRect), so
  applications that probe cube support degrade gracefully; DEFAULT-pool
  creates are rejected.
- **Point sprites** and non-solid fill modes (Metal has no native wireframe).
- **TIMESTAMP and the other niche query types** — creation reports
  NOTAVAILABLE, as the spec allows.
- **YUV conversion** — YUY2/UYVY surfaces can be created and locked, but no
  YUV→RGB blit is performed.
- Depth→depth StretchRect.

### Deliberately not implemented

- **D3D9Ex** — no Direct3DCreate9Ex, no shared resource handles, no
  D3D9On12. The extended interface is a different contract (device removal,
  OS-managed memory) built for the Vista+ compositor; the games this project
  targets are plain D3D9.
- **Exclusive fullscreen and display-mode switching** — presentation is a
  composited Metal layer; fullscreen means borderless at desktop resolution.
  The Win32 fullscreen lifecycle (mode changes, device-lost focus dance) is
  not emulated.
- **Software paths** — no reference or software rasterizer, no software
  vertex processing, no ProcessVertices, no RegisterSoftwareDevice. HAL on
  the default Metal device is the only device type; multi-adapter setups are
  not enumerated.
- **Hardware instancing** (SetStreamSourceFreq) — the renderer is
  single-stream by architecture.
- **Clip-plane application** — SetClipPlane state round-trips but planes are
  not applied on the GPU.
- **Legacy remnants** — N-patch/RT-patch tessellation, vertex tweening,
  palettized textures, gamma ramp: dead features in real-world content,
  accepted or rejected per spec but non-functional.

### Testing

mtld3d is developed and tested against **World of Warcraft 1.12 and 3.3.5a**
under Wine and CrossOver. No other games have been exercised yet — it may
well work, but expect rough edges, and reports are welcome.

Beyond the game workloads, the implementation is hardened against **Wine's
d3d9 test suite** — the de-facto D3D9 conformance suite. A dedicated runner
(`make conformance`) executes it against the installed builtin on both PE
architectures and gates on a per-site tracked baseline in which every
remaining divergence is triaged and classified with a documented rationale.
The methodology, classification scheme, and the current audit live in
[`unix/conformance/CONFORMANCE.md`](unix/conformance/CONFORMANCE.md). The
unit and end-to-end suites (`make test`) run the pure-Rust core natively on
the host and the full stack under Wine.

## Prerequisites

mtld3d builds and runs on **Apple Silicon macOS**. The shipped `mtld3d.so` is
an x86_64 Mach-O that runs under **Rosetta 2** inside Wine (install it with
`softwareupdate --install-rosetta`), and the Metal backend targets **macOS 15**
or newer (`unix/.cargo/config.toml` pins `MACOSX_DEPLOYMENT_TARGET = 15.0`).

The following must be available on `PATH`:

- A **Wine** build or install providing `winebuild`, `wine`, `wineserver`, and
  `wineboot`, plus its development tree (`lib/wine/{i386,x86_64}-windows/` and
  `libwinecrt0.a`). Point `WINE_SDK` at that tree — see [Wine paths](#wine-paths).
- **Homebrew** — `make setup` installs LLVM/lld and cargo tooling through it.
- A **rustup** toolchain (stable for builds; **nightly** is also required, for
  `make fmt`'s nightly-only rustfmt options).

`make setup` installs only the cross-compilation toolchain — it does **not**
install Wine or Rosetta. mtld3d is built and distributed as source via the
`Makefile`; the internal crates use path dependencies and are not published to
crates.io.

## Build

`WINE_SDK` must be exported before running **any** target (the Makefile requires
it even for `make setup`); `make install` additionally needs `WINE_INSTALL_DIR`.
See [Wine paths](#wine-paths).

```sh
make setup              # install cross-compilation toolchain (first time only)
make                    # build all (windows i386+x64 + unix)
make install            # install to Wine distribution
make bundle             # pack a distributable tarball (PROD=1 by default; see Installing below)
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

`make setup` installs LLVM and lld via Homebrew, adds the `i686-pc-windows-msvc`, `x86_64-pc-windows-msvc`, and `x86_64-apple-darwin` rustup targets, installs xwin and cargo-edit, and splats the Windows SDK (~3 GB) to `/opt/xwin` — creating and `chown`ing that root-owned directory takes a one-time `sudo` prompt. Windows targets cross-compile from macOS via `lld-link` + xwin (see `windows/.cargo/config.toml`); `unix/` targets `x86_64-apple-darwin` explicitly because Wine's unix `.so` must be x86_64 Mach-O. `rustfmt.toml` uses nightly-only options (`imports_granularity`, `group_imports`), so `make fmt` must be invoked explicitly.

## Architecture

```
test.exe → d3d9.dll → mtld3d.dll → mtld3d.so
(i386 PE)  (i386 PE)  (i386 PE)  (x86_64 Mach-O)

test.exe → d3d9.dll → mtld3d.dll → mtld3d.so
(x64 PE)   (x64 PE)   (x64 PE)   (x86_64 Mach-O)
```

- `d3d9.dll` — D3D9 API implementation: COM vtables, caps, state management.
- `mtld3d.dll` — PE shim that owns Wine's unix-call globals and exports `mtld3d_unix_call`.
- `mtld3d.so` — native macOS side, a pure Metal abstraction layer.
- `mtld3d-core` — host-testable pure-Rust rlib linked into `d3d9.dll`.

At runtime the frame flows through a three-thread pipeline:

```
API thread (the game's)     Encoder thread            Submit thread
───────────────────────     ──────────────            ─────────────
record frame N+1        →   encode frame N        →   submit + present frame N−1
```

- The **API thread** is the game's own render thread. The D3D9-era games this
  project targets drive the API from a single thread, so that thread is the
  frame-time bottleneck — it must never wait on translation, Metal, or the
  GPU. Every D3D9 call therefore only snapshots the state it needs into a
  closure and appends it to the current frame's op list; `Present` hands the
  finished frame to the encoder and immediately starts recording the next.
  No translation or encoding work runs on the API thread.
- The **encoder thread** (one per device) runs the closures: D3D9 → Metal
  translation, render-pass scheduling and load/store optimization, pipeline
  and sampler caches, lazy resource creation and texture uploads.
- The **submit thread** takes the encoder's finished frame payload, crosses
  the PE/Unix boundary to replay the command stream into Metal encoders,
  waits for the drawable, and presents and commits — overlapping the
  encoder's build of the following frame.

Each hand-off has capacity one, so the pipeline never runs more than one
frame ahead per stage — backpressure, not queueing, bounds latency.

For the PE/Unix boundary contract, the threading details, perf instrumentation, and the shader/heap debugging toolkits, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Workspaces

Two Cargo workspaces, one per target platform. Open each in a separate editor window for rust-analyzer to work correctly.

| Workspace  | Targets                                              | Members                                       |
|------------|------------------------------------------------------|-----------------------------------------------|
| `windows/` | `i686-pc-windows-msvc`, `x86_64-pc-windows-msvc`     | shim, d3d9, core, tests, types                |
| `unix/`    | `x86_64-apple-darwin`                                | unix, shared                                  |

`mtld3d-core` is a pure-Rust rlib holding every platform-independent helper — DXSO → MSL emission, render-pass state machine, slab allocator, format / FVF / vertex-decl / dirty-rect math, fixed-function state. Compiles on both Windows PE and macOS host, so `cargo test -p mtld3d-core --target aarch64-apple-darwin` runs unit tests natively instead of through Wine. `d3d9.dll` consumes it as an rlib dep.

`unix/shared` is the crate every linkage unit depends on. Primary role: PE↔Unix wire format (`Command` enum, `Thunks` enum, param structs, typed `mtl::` wire values). Pure data and pure-Rust helpers only — no FFI, no `#[link]` — so both workspaces can depend on it cleanly.

## Targets

| Crate         | Workspace  | Output                                                  |
|---------------|------------|---------------------------------------------------------|
| `mtld3d`      | `windows/` | `mtld3d.dll`                                            |
| `d3d9`        | `windows/` | `d3d9.dll`                                              |
| `mtld3d-core` | `windows/` | rlib (linked into `d3d9.dll`)                           |
| `mtld3d-unix` | `unix/`    | `mtld3d.so`                                             |
| `shared`      | `unix/`    | rlib (shared by `d3d9.dll`, `mtld3d.dll`, `mtld3d.so`)  |

## Wine paths

- PE DLLs (i386): `lib/wine/i386-windows/`
- PE DLLs (x64): `lib/wine/x86_64-windows/`
- Unix `.so`: `lib/wine/x86_64-unix/`
- `WINE_INSTALL_DIR`: where `make install` copies built binaries.
- `WINE_SDK`: Wine SDK consumed by `windows/shim/build.rs` for `libwinecrt0.a` / `ntdll.a` linking (typically the same path as `WINE_INSTALL_DIR`).

`make install` stamps the Wine-builtin signature onto the `d3d9.dll` copies it
places under `lib/wine` (the loader ignores unsigned PEs on the builtin search
path). The build outputs themselves stay unsigned native PEs so they can also
be deployed as a native DLL override — the bundle ships both flavors, see
[Installing the bundle](#installing-the-bundle).

## Installing the bundle

`make bundle` packs a distributable tarball into
`windows/target/mtld3d.tar.xz`. It supports two install routes:

- **Builtin** — the bundle's `wine/` tree mirrors `lib/wine/` with every PE
  builtin-marked (including `d3d9.dll`), so it drops verbatim into a Wine
  installation you own and replaces the stock d3d9 for that whole tree.
- **Native override** — the bundle's `native/` tree holds the unmarked
  `d3d9.dll`, loaded per game (or per prefix) through a `d3d9=native` DLL
  override while the Wine installation stays untouched. This is the route
  for [CrossOver](https://www.codeweavers.com/crossover) bottles, where it
  is self-contained in the bottle and survives CrossOver updates.

The step-by-step walkthroughs for stock Wine and CrossOver, and the
tradeoffs between the routes, live in [`INSTALL.md`](INSTALL.md) — it ships
inside the bundle.

## Logging

Every crate uses `log` + `env_logger`. Each call tags `target = LOG_TARGET` via a crate-level const. All targets sit under `mtld3d::*`; `env_logger` matches by `::`-separated prefix, so `RUST_LOG=mtld3d=…` is the single switch for the whole project.

Examples: `RUST_LOG=mtld3d=warn`, `RUST_LOG=mtld3d::d3d9=trace`, `RUST_LOG=mtld3d=warn,mtld3d::unix=trace`, `RUST_LOG=mtld3d::perf=debug`.

| Target                  | Scope                                                                 |
|-------------------------|-----------------------------------------------------------------------|
| `mtld3d::d3d9`          | `windows/d3d9/` + `windows/core/` (everything except `dxso` and `perf`) |
| `mtld3d::d3d9::cursor`  | hardware cursor (HCURSOR) lifecycle, bitmap cache, wndproc            |
| `mtld3d::dxso`          | DXSO → MSL emitter                                                    |
| `mtld3d::perf`          | 5-second averaged performance summary                                 |
| `mtld3d::shim`          | Wine unix-call PE shim DLL                                            |
| `mtld3d::unix`          | Metal-side `.so`                                                      |

Levels: `info!` one-shot milestones · `warn!` unimplemented stubs / fallback paths · `error!` unexpected internal failures · `trace!` per-call breadcrumbs for init debugging · `debug!` routine per-call noise useful for deep debugging.

Each cdylib (`d3d9.dll`, `mtld3d.dll`, `mtld3d.so`) calls `env_logger::try_init` independently. `d3d9.dll` and `mtld3d.dll` init from their own `DllMain`; `mtld3d.so` has no owning entry point, so `d3d9.dll` dispatches a one-shot `InitLogger` thunk from its init path. `try_init` is idempotent.

The `mtld3d::perf` summary, its counter aggregation rules, and the per-call cycle helpers are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Configuration

User-facing runtime options live in `mtld3d.conf` (key = value, comments are `#`-prefixed). mtld3d looks for it next to the running `.exe` (`std::env::current_exe()` → strip basename → join `mtld3d.conf`). A missing file is fine — defaults apply. The file is read once at `Direct3DCreate9`; restart the game to pick up edits. The repo root holds a sample with every option, its default, and a short explanation.

Resolved options are logged once at startup under `RUST_LOG=mtld3d::d3d9=info` (`config: <key> = <value>` lines). Unknown keys, malformed lines, and unparseable values fire `log_once_warn!` and otherwise no-op — a typo doesn't silently get the default.

Every key is also overridable at process launch via `MTLD3D_CONFIG`, a semicolon-separated list of `key=value` entries using the same syntax as the file (e.g. `MTLD3D_CONFIG="color.hdr.enable=true;cursor.scale=2"`). Env entries are merged on top of the file (env wins on conflict).

## Contributing

The development conventions (module layout, visibility, encapsulation, `unsafe` discipline, warning suppressions, the `log_once_warn!` rule, doc-comment shape, typed `objc2-*` bindings) live in [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md).

Run **`make check`** before every commit. It is the whole gate: `cargo fmt --check`, clippy at `-D warnings` with `nursery` + `pedantic` denied, `make audit` (the rules clippy cannot express — doc-comment shape, the `Clone`/`Copy` derive inventory, and the patterns confined to a known set of files), and `make doc` (rustdoc with warnings denied, so doc links have to actually resolve). Each audit finding names the section of `docs/CONVENTIONS.md` it comes from.

## License

[zlib](LICENSE).
