# mtld3d

Direct3D 9 for Wine on macOS, backed by Metal.

mtld3d replaces Wine's `d3d9.dll`. The PE side implements the D3D9 API and
translates it into Metal command buffers that a native library executes on the
host. The goal is the fastest Direct3D 9 implementation for Wine on macOS.
Direct3D 8 on the same core is planned. Every other Direct3D version is a
non-goal: D3D10 and later are already served on macOS by Apple's D3DMetal and
by DXMT.

Conformance serves speed rather than defining it. The implementation is
developed against Wine's d3d9 test suite and every divergence is triaged and
written down, but where matching D3D9 exactly would cost frame time, speed
wins as long as no game breaks. Those trades are listed under
[Kept divergences](#kept-divergences), each with its `mtld3d.conf` knob where
one makes sense.

Apple Silicon is what is developed and tested. Intel Macs are a goal, not a
claim: the paths for a GPU without unified memory exist but have not run on
real hardware.

## Requirements

| Requirement | Notes |
| --- | --- |
| macOS 15 or newer | Apple Silicon or Intel. |
| Wine 8.0 or newer, or CrossOver 24 or newer | Needs the current WoW64 loader. |
| A 64-bit prefix or bottle | 32-bit games run in it through WoW64. |
| Rosetta 2 | For an x86_64 Wine, which most builds are. An arm64 Wine translates x86 itself (FEX) and does not need it. |

D3D9-era games do their floating-point math in x87 instructions, which
Rosetta 2 translates slowly. Under an x86_64 Wine, run the game with
[x87sidecar](https://github.com/athei/x87sidecar), a JIT that replaces
Rosetta's x87 handling. The Wine builds from
[wine-build](https://github.com/athei/wine-build) carry the patch its
cooperative attach mode needs.

## Installation

Download `mtld3d.tar.xz` from
[GitHub Releases](https://github.com/athei/mtld3d/releases). It ships two
ways to load the same binaries:

| Route | Use it when | Cost |
| --- | --- | --- |
| Builtin | You own the Wine installation: your own build, a package, an app-bundled Wine. | Replaces the stock d3d9 for that whole Wine tree, and a Wine update wipes it. Not possible on CrossOver. |
| Native override | CrossOver, or a stock Wine whose own d3d9 should keep serving other applications. | A `d3d9=native` registry override per prefix plus a `d3d9.dll` copy per game or per prefix. |

[`INSTALL.md`](INSTALL.md), also inside the bundle, has the steps for stock
Wine and for CrossOver bottles, including the prefix markers a hand-installed
builtin needs.

To confirm mtld3d is loaded, open the newest file in `mtld3d-logs` next to the
game executable. Every run starts with

```
[mtld3d::shim] mtld3d.dll <version> <id>, unix call initialized
[mtld3d::d3d9] d3d9.dll <version> <id> loaded at <address>
```

Without them `d3d9.dll` never mapped, and the troubleshooting section of
`INSTALL.md` walks through the causes.

## Configuration

Runtime options live in `mtld3d.conf`, read once at `Direct3DCreate9` from
the directory of the running `.exe`, so a change needs a restart. The
[sample](mtld3d.conf) in the repository root documents every option. A
missing file is fine: defaults apply. Every key can also be set at launch
through the `MTLD3D_CONFIG` environment variable, a semicolon-separated list
of `key=value` entries that wins over the file.

| Key | Default | What it does |
| --- | --- | --- |
| `render.scale` | `1.0` | Render at a fraction of the presented size and let MetalFX upscale on the way to the screen. `0.75` is a good first try. |
| `present.maxFps` | `0` | Frame-rate ceiling in Hz, independent of vsync. `0` is uncapped. |
| `color.hdr.enable` | `true` | Use the HDR present path on a display with EDR headroom. |
| `color.space` | `passthrough` | Tag the layer with the display's own colorspace, or with sRGB (`accurate`). |
| `cursor.software` | `auto` | Draw the cursor in an overlay window instead of the hardware cursor. `auto` is on under HDR only. |
| `cursor.scale` | `auto` | Cursor bitmap enlargement. `auto` doubles it in Wine's retina mode. |
| `shaderCache.enable` | `true` | Persist compiled shaders in `mtld3d_shaders.bin` next to the `.exe`. |
| `log.dir` | `mtld3d-logs` next to the `.exe` | Where log files and GPU traces go. |
| `query.flushImmediate` | `false` | Answer `GetData(D3DGETDATA_FLUSH)` at once instead of waiting for the GPU. |
| `depth.aliasSameSize` | `false` | A newly bound depth texture of the same size inherits the previous one's contents. |
| `buffer.ignoreLockBounds` | `false` | Upload a whole static buffer on every Lock instead of the range it announced. |
| `memory.vbibRetentionCapMB` | `512` | Cap on vertex and index buffer backings retained while the GPU reads them. |
| `memory.vramBudgetMB` | `1024` (32-bit), `0` (64-bit) | Ceiling on the video memory `GetAvailableTextureMem` reports. |
| `memory.pageboxPoolCapMB` | `128` | Recycle pool for retired dynamic buffer backings. |
| `adapter.spoof` | `none` | Report the adapter as `nvidia` or `amd`. |
| `caps.dfFormats` | `true` | Advertise the DF16 and DF24 depth formats. |

The `debug.*` keys (`capsAll`, `expandPacked16`, `float32Filtering`,
`bytecodeDumpDir`, `skipShaders`) are diagnostics and documented in the sample
only.

Below the file and the environment sits a third layer. A few games need
options nobody should have to discover, so mtld3d ships profiles for them,
matched on the executable name plus the version resource its vendor linked in.
A profile only supplies starting values; the file and the environment override
it key by key. `RUST_LOG=mtld3d::d3d9=info` names the profile that matched.

| Profile | Application | What it sets and why |
| --- | --- | --- |
| `gta-iv` | Grand Theft Auto IV | `adapter.spoof=amd`: the renderer stalls in its own identifier parsing on the NVIDIA identity. `caps.dfFormats=false`: with DF advertised next to INTZ it picks a mixed depth path no hardware of its era offered. `depth.aliasSameSize=true`: its late alpha, sky and glow passes z-test an INTZ texture against depth rendered into a same-size sibling. |
| `wow` | World of Warcraft, 1.12 and 3.3.5 clients | `query.flushImmediate=true`: both clients poll `GetData(D3DGETDATA_FLUSH)` as a GPU fence after every loading-screen upload batch and never read the count, so the spec-correct wait would cost seconds per load. |

## Fullscreen and display modes

A fullscreen device sets the display mode the game picked, as native D3D9
does, and covers the monitor with a borderless window. That mode-set is meant
to stay virtual. Set Wine's `EmulateModeset` key in the prefix:

```sh
wine reg add 'HKCU\Software\Wine\X11 Driver' /v EmulateModeset /d Y /f
```

win32u reads the key whatever the driver, once per session, so set it before
launching the game and let the previous Wine session exit first. With it, the
physical display keeps its resolution, the game window is scaled onto it, and
mouse input is mapped into the mode, so clicks land where the game drew its
UI. A mode of another aspect than the display's is letterboxed. Without the
key, Wine's mac driver hands the mode-set to the display and the whole desktop
switches resolution.

The resolution list a game sees carries sizes of the display's own aspect
only, largest first and at most 15 per colour format, because Wine's full list
overflows menus built for a driver's short one. Any mode Wine accepts stays
settable whether listed or not. A request that matches no mode, such as a size
a game derived from its own window, follows the window instead.

`render.scale` multiplies on top of the mode: a 1600x900 setting at `0.75`
rasterizes 1200x675 and MetalFX upscales it to the screen in one pass. A
fullscreen game is never told it lost its device on a focus change: the
desktop mode comes back on deactivation and the game's mode is set again on
activation, with no `D3DERR_DEVICELOST` in between.

## HDR and the cursor

On a display with EDR headroom the layer is upgraded to extended dynamic range
and present routes through an inverse tone-mapping pass (BT.2446-A in ICtCp)
whose peak follows the display's live headroom. This is on by default;
`color.hdr.enable=false` forces the SDR path. A display without EDR headroom
runs the SDR path either way.

macOS has no HDR hardware cursor, so under HDR the game's cursor bitmap is
drawn through the same tone map in a transparent overlay window that follows
the pointer. That is what `cursor.software=auto` resolves to. The overlay also
never toggles the hardware cursor plane, which on stock Wine delays a present
per show or hide, so `cursor.software=true` is worth trying on SDR for a game
that hides the cursor while a mouse button is held. `cursor.scale` enlarges the
bitmap; `auto` doubles it in Wine's retina mode, where it would otherwise come
out at half size.

## Logging and diagnostics

Every line of both halves of mtld3d goes to a file, never to the process's
standard streams. Each process writes `<exe>-<pid>.log` into `mtld3d-logs`
beside the executable, `<pid>` being the macOS process id, so a launch never
overwrites the log of the one before it. The directory keeps the ten newest
logs and the ten newest GPU traces; `log.dir` moves it.

`RUST_LOG` filters the log. Unset, everything logs at `info`. All targets sit
under `mtld3d::*` and match by prefix, so `RUST_LOG=mtld3d=warn` is the single
switch for the whole project and `RUST_LOG=mtld3d=warn,mtld3d::unix=trace`
narrows one part.

| Target | Scope |
| --- | --- |
| `mtld3d::d3d9` | The D3D9 API implementation, `d3d9.dll`. |
| `mtld3d::dxso` | The DXSO to MSL shader translator. |
| `mtld3d::shim` | The unix-call shim, `mtld3d.dll`. |
| `mtld3d::unix` | The Metal side, `mtld3d.so`. |
| `mtld3d::perf` | 5-second performance summary, `PERF=1` builds only. |

`warn` covers unimplemented paths and fallbacks, `info` one-shot milestones,
`debug` and `trace` per-call detail. The narrower diagnostic targets are
listed in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Pressing F12 in a game records the next three frames: one `[dump]` log line
per D3D9 event a GPU trace cannot show (target and viewport changes, clears,
copies, query traffic, every draw with its shaders and textures), and a Metal
GPU trace of the same frames beside the log file as
`<exe>-<pid>-<n>.gputrace`. The trace needs `MTL_CAPTURE_ENABLED=1` in the
game's environment; without it the log says so and the dump still runs.

## Tested games

| Game | Status | Notes |
| --- | --- | --- |
| World of Warcraft 1.12 | Plays | The primary target. `wow` profile. |
| World of Warcraft 3.3.5a | Plays | `wow` profile. |
| Half-Life 2 | Plays | |
| Grand Theft Auto IV | Plays | `gta-iv` profile. |
| 3DMark05 | Runs end to end | |
| Unigine Tropics | Runs | |
| Gunmetal | Starts and benchmarks | |

Games that fail are tracked as `game-compat` issues in the
[tracker](https://github.com/athei/mtld3d/issues); reports are welcome.

## Status

### Supported

| Area | What works |
| --- | --- |
| Shaders | Vertex and pixel shader models 1.x through 3.0 through DXSO to MSL translation, including vPos, vFace, flat shading and the half-pixel rasterization fixup. Compiled shaders are cached on disk by content hash. |
| Fixed function | Vertex and pixel: lighting (directional, point, spot), texture-coordinate generation, the full texture-stage cascade, material colour sources, hardware vertex blending. Vertex and table fog on every path, pre-transformed geometry included. |
| Draws | All four draw calls and every primitive type; triangle fans are rewritten as lists, which Metal lacks. Points with `D3DRS_POINTSIZE`, per-vertex size, scaling and point sprites. Six user clip planes. All sixteen vertex streams with per-stream offsets and strides, and hardware instancing through `SetStreamSourceFreq`. |
| State | Recorded and `D3DSBT_*` state blocks; occlusion queries on Metal visibility results; event queries. |
| Resources | DXT1 to DXT5 and ATI1 compressed textures, the common integer and float formats, cube and volume textures, mipmap auto-generation, managed-pool dirty-region uploads, `StretchRect` with format conversion and YUY2/UYVY decoding, `GetDC`. The system-memory and scratch pools live CPU-side and allocate no Metal texture until bound for sampling. |
| Depth and stencil | Sampleable depth (INTZ, DF16, DF24) with hardware shadow compare, depth bias and slope-scale bias, depth clamp for pre-transformed geometry. The full stencil test, both faces, every operation. |
| Sampling and output | Anisotropic filtering, per-sampler LOD bias, sRGB read and write (the write through the target's sRGB view, so it happens after blending), alpha test, scissor, separate alpha blend, blend factor, colour write masks. |
| Multiple render targets | Four targets with independent formats and write masks, blending on each, `Clear` reaching all of them. |
| Multisampling | 2x and 4x everywhere, 8x where the device offers it, on swap chains, render targets and depth-stencil surfaces, resolved for every consumer. `D3DRS_MULTISAMPLEMASK`. |
| Presentation | Windowed and fullscreen swap chains, mode enumeration, hardware and software cursors, MetalFX upscaling with `render.scale`, HDR output. |

### Not implemented yet

Each of these fails cleanly, with an absent cap bit or a documented error
return, so applications take their own fallback paths.

| Feature | Behaviour |
| --- | --- |
| Non-solid fill modes | Metal has no wireframe. The state is accepted, warned once and drawn solid. |
| Timestamp and other niche query types | Creation reports `D3DERR_NOTAVAILABLE`, as the spec allows. |
| Scaled, sub-rect or converting depth-to-depth `StretchRect` | The whole-surface 1:1 copy between same-format DEFAULT-pool depth surfaces works, including a multisample resolve; anything else returns `D3DERR_INVALIDCALL`. |

### Deliberately not implemented

| Feature | Why |
| --- | --- |
| D3D9Ex | No `Direct3DCreate9Ex`, shared handles or D3D9On12. The extended interface is a different contract, built for the Vista compositor, and the games this project targets are plain D3D9. |
| Physical display-mode switching | The mode a fullscreen game picks is meant to stay virtual, see [Fullscreen and display modes](#fullscreen-and-display-modes). |
| Device loss | Presentation is a composited Metal layer and no exclusive mode is taken, so a lost display is never a lost device. `TestCooperativeLevel` reports `D3D_OK` across focus changes and `D3DERR_DEVICENOTRESET` only after a failed `Reset`. Reporting a loss that did not happen would make every fullscreen game rebuild its DEFAULT-pool working set on each activation change. No focus-window subclassing, no minimise. |
| Software paths | No reference or software rasterizer, no software vertex processing, no `RegisterSoftwareDevice`. HAL on the default Metal device is the only device; other adapters are not enumerated. `ProcessVertices` transforms through the current FVF and rejects an explicit declaration. |
| Legacy remnants | N-patch and RT-patch tessellation, vertex tweening, palettized textures, gamma ramp. Accepted or rejected per spec, non-functional. |

### Kept divergences

Divergences from D3D9 kept on purpose because closing them costs frame time,
memory, or the games that rely on the looser behaviour. The full rationale for
each sits with its conformance sites in
[`CONFORMANCE.md`](unix/conformance/CONFORMANCE.md).

| Divergence | Why | Knob |
| --- | --- | --- |
| `IDirect3DTexture9::LockRect` serves a level of a DEFAULT-pool 2D texture created without `D3DUSAGE_DYNAMIC`, which D3D9 rejects. The surface entry point, cube and volume locks still reject it. | A game that streams into such a texture would otherwise lose every upload. The cost is system memory: the level's staging, released once its upload retires, is re-created, and a partial lock after that release leaves the pixels outside its rect out of step with the GPU copy (warned once per texture). | none |
| `GetData(D3DGETDATA_FLUSH)` can answer a pending occlusion query at once instead of waiting for the GPU. | Off by default. A title that uses the poll loop only as a GPU fence and never reads the count pays API-thread time for nothing; the `wow` profile turns it on. | `query.flushImmediate` |
| Depth stores are elided where nothing reads the buffer back. | Content relying on depth surviving a pass it never cleared can read stale depth. Preserving it unconditionally costs the optimization on every frame of every game that does clear, which is all tested ones. | none |
| A partial `Lock` of a dynamic vertex or index buffer without `D3DLOCK_DISCARD` returns a pointer into memory a queued draw may still read. | D3D9 keeps the game's writes from landing under a draw the GPU has not reached; here `D3DLOCK_NOOVERWRITE` semantics apply by default. Matching D3D9 means stalling or renaming the backing on every such lock, and a dynamic buffer is what a UI or particle batcher locks dozens of times per frame. The rename path has been measured peaking near 1.4 GB of retained backings, which is why `memory.vbibRetentionCapMB` exists. | none |
| A partial `LockRect` of a texture level without `D3DLOCK_NOOVERWRITE` or `D3DLOCK_READONLY` returns a pointer into staging an upload may still read. | The same trade for a font atlas or lightmap page written a few rectangles at a time. A whole-level lock is renamed and its contents preserved, because Half-Life 2's lightmap pages rely on that. | none |
| A DEFAULT-pool `D3DUSAGE_WRITEONLY` static vertex or index buffer keeps no CPU copy once every byte has reached the GPU. | D3D9 preserves contents across a plain `Lock` whatever the usage says, so a title that reads back through the pointer sees zeros, and one that writes past its announced window loses those bytes (warned once). Inside a large-address-aware 32-bit title those copies measured near a gigabyte of the 4 GiB the title needs itself. An indexed triangle fan on a released index buffer copies it back off the GPU once, at one mid-frame GPU wait. | `buffer.ignoreLockBounds` keeps the copy |
| `D3DRS_MULTISAMPLEANTIALIAS = FALSE` is ignored. | Metal ties the sample count to the pass's attachments with no per-draw override. `D3DPRASTERCAPS_MULTISAMPLE_TOGGLE` is not advertised, which is how D3D9 says the toggle is unavailable, and the first write is logged. | none |

## Conformance and tests

Beyond the games, mtld3d is hardened against Wine's d3d9 test suite, the
de-facto D3D9 conformance suite. `make conformance` runs it against the
installed builtin, one runner process per PE architecture, and gates on a
per-site baseline; every remaining divergence is classified with a written
rationale in [`CONFORMANCE.md`](unix/conformance/CONFORMANCE.md), the ones
kept for speed included. `make test` runs the unit tests of the pure-Rust
core natively on the host and the end-to-end suite, listed in
[`COVERAGE.md`](windows/tests/COVERAGE.md), under Wine.

## Building from source

mtld3d builds on Apple Silicon macOS and targets macOS 15 or newer
(`unix/.cargo/config.toml` pins the deployment target). The PE side is built
for i386 and x86_64; `mtld3d.so` ships as both an x86_64 and an arm64 Mach-O,
because Wine loads it from the `lib/wine/<cpu>-unix` directory matching its
own build.

Two things have to exist before a build: a Wine build or install providing
`wine`, `winebuild` and `wineserver` plus its development tree
(`lib/wine/{i386,x86_64}-windows/` and `libwinecrt0.a`), and rustup.
`WINE_SDK` points at that Wine tree and must be exported before any target,
`make setup` included: the Makefile takes the Wine binaries from it, the shim's
build script links against `libwinecrt0.a` and `ntdll.a` there, and
`make conformance` finds Wine's d3d9 test binaries in it. `WINE_INSTALL_DIR`
names a second tree `make install` also copies into.

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

`make setup` installs both rustup toolchains (stable, 1.97 or newer, for
everything; nightly for `make fmt` only) with the four cross-compilation
targets, the cargo tools, the PE linker and archiver as symlinks onto the
toolchain's own `rust-lld` and `llvm-ar`, Rosetta 2 if it is missing, and the
pinned Windows SDK under `/opt/xwin` (about 3 GB, one `sudo` prompt for that
root-owned directory). It does not install Wine. `FP=1 make` turns frame
pointers on for the guest-pc sampling profiler.

`make install` copies the PE DLLs into `{i386,x86_64}-windows/` and the `.so`
into `{x86_64,aarch64}-unix/` under the tree the Wine build reads mtld3d
from: `lib/wine/d3d9/mtld3d/` in a build with a compat database, which keeps
every Direct3D implementation in its own subtree, or `lib/wine/` itself in an
older tree. The installed `d3d9.dll` copies get the Wine-builtin signature,
because the loader ignores unsigned PEs on the builtin search path; the build
outputs stay unsigned so they can also serve as a native override.

`make bundle` packs both flavours into `windows/target/mtld3d.tar.xz` and the
debug symbols (a `.pdb` beside each PE, a `.dSYM` beside the `.so`) into
`windows/target/mtld3d-debug.tar.xz`. Every DLL logs its release and image ID
on load, so a crash report names the archive that symbolicates it. Every make
leg is also a CI job, with the toolchains pinned in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Architecture

```
game.exe → d3d9.dll → mtld3d.dll → mtld3d.so
(PE, i386 or x64: one chain per arch)  (Mach-O, Wine's own arch)
```

`d3d9.dll` implements the API and holds every piece of D3D9 knowledge,
`mtld3d.dll` is the PE shim that owns Wine's unix-call globals, and
`mtld3d.so` is a pure Metal abstraction layer on the host. At runtime a frame
flows through three threads: the game's own API thread only snapshots state
into closures, an encoder thread translates them into Metal commands, and a
submit thread crosses the boundary to replay, present and commit. Each
hand-off has capacity one, so the pipeline never runs more than one frame
ahead per stage. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) has the
boundary contract, the threading model, the workspace layout, the logging
targets and the debugging toolkits.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) is the operating manual: the gates and
how to read their output, how conformance work is organised, and what a pull
request is expected to contain. The conventions live in
[`docs/CONVENTIONS.md`](docs/CONVENTIONS.md). The short version: `make check`
and `make test` are green before every commit, and one pull request is one
change.

## License

[zlib](LICENSE).
