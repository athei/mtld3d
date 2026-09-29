# mtld3d

Direct3D 9 for Wine on macOS, backed by Metal.

mtld3d replaces Wine's `d3d9.dll`. The PE side implements the D3D9 API and
translates it into Metal command buffers that a native library executes on the
host. The goal is the fastest Direct3D 9 implementation for Wine on macOS.
Direct3D 8 on the same core is planned
([#788](https://github.com/athei/mtld3d/issues/788)). Direct3D 10 and later
are a non-goal: Apple's D3DMetal and DXMT already serve them on macOS.

Conformance serves speed: where matching D3D9 exactly would cost frame time
and no game breaks, speed wins. Those trades are listed in
[`docs/STATUS.md`](docs/STATUS.md#kept-divergences).

## Features

- Shader Model 1 to 3 and the fixed-function pipeline are translated to Metal
  Shading Language by mtld3d's own translator.
- Asynchronous shader compilation, on by default. Shaders and pipelines a game
  uses for the first time build on worker threads. While a build runs on a
  cold cache, a draw into a target the game rebuilds every frame is skipped
  for a frame or two instead of stalling the frame, so the game does not
  stutter. Other draws wait, once per frame for all their builds together.
  Turned off, every frame waits for every build it needs.
- A shader cache, on by default. Translated shaders and pipelines are kept in
  a file next to the game and built again before the first frame of the next
  launch, so combinations seen before do not compile during play. Turned off,
  every launch compiles on first use.
- Frame pacing through Metal's minimum-duration present
  (`presentDrawable:afterMinimumDuration:`) for both vsync and the frame cap.
  Unlike pacing the game's thread or enabling display sync on the Metal
  layer, this works with ProMotion: the panel follows whatever rate the game
  sustains below its maximum. Vsync follows the game's present interval; the
  frame cap is off by default.
- Upscaling, off by default. The game can render below the presented size and
  have MetalFX's spatial scaler upscale the result. A mip LOD bias, on by
  default, keeps texture detail at the presented size; turned off, textures
  are sampled for the smaller render size.
- HDR output, on by default. On a display with EDR headroom the frame is
  expanded into that headroom by inverse tone mapping that follows the live
  headroom. Turned off, the frame is presented as SDR. SDR displays run the
  SDR path either way.
- A software cursor. It is drawn in its own overlay window independently of
  the game's frames, so it is not tied to the frame rate and has
  hardware-cursor latency. Under HDR it is tone-mapped like the frame, where
  the macOS cursor stays at SDR brightness, and showing or hiding it does not
  delay the next present by a refresh as the hardware cursor does. On by
  default whenever HDR output is active, and it can be forced on or off.
  Turned off, the game gets the hardware cursor.
- Rendering off the game's thread. The game's thread records state, and
  encoder, submit and presenter threads do the rest, at most one frame ahead.
- Built-in profiles for the few games that need settings of their own,
  matched on the executable and its version resource.

Every switch, with its default, is in [`mtld3d.conf`](mtld3d.conf).
[`docs/STATUS.md`](docs/STATUS.md) lists everything that is implemented.

## Requirements

- macOS 15 or 26, on Apple Silicon or Intel. CI covers macOS 15 and 26 on
  Apple Silicon and macOS 15 on Intel, because no macOS 26 Intel runners
  exist. Intel on macOS 26 is expected to work.
- A Wine from [wine-build](https://github.com/athei/wine-build), based on
  CrossOver 26; CI pins a release of it. Older Wine or CrossOver releases are
  not expected to work.
- CrossOver 27's arm64 Wine is wired into the Makefile and tested locally,
  but not run in CI.
- A 64-bit prefix or bottle. 32-bit games run in it through WoW64.
- Rosetta 2 for an x86_64 Wine. An arm64 Wine translates x86 itself.
- [x87sidecar](https://github.com/athei/x87sidecar) under an x86_64 Wine.
  D3D9-era games do their floating-point math in x87 instructions, which
  Rosetta 2 translates slowly. The wine-build releases carry the patch it
  needs.

## Installation

Download `mtld3d.tar.xz` from
[GitHub Releases](https://github.com/athei/mtld3d/releases). It installs
either as a Wine builtin, replacing the stock d3d9 of a Wine tree you own, or
as a native DLL override per prefix, which is the route for CrossOver.
[`INSTALL.md`](INSTALL.md), also inside the bundle, has the steps for both.
Its [Fullscreen](INSTALL.md#fullscreen) section shows how to keep a
fullscreen game's mode change virtual.

## Configuration

mtld3d reads `mtld3d.conf` from the directory of the game's `.exe` at every
`Direct3DCreate9`. The interface that call returns keeps what was read, for
itself and every device it creates. A missing file means defaults. The
`MTLD3D_CONFIG` environment variable takes the same entries, separated by
semicolons.

Each option is resolved from four layers. A later layer wins, key by key:

1. The built-in default.
2. The built-in profile for the running game, if there is one.
3. `mtld3d.conf`.
4. `MTLD3D_CONFIG`.

A profile matches on the executable name plus the version resource its
vendor linked in, and the log names the profile that matched. The
[sample `mtld3d.conf`](mtld3d.conf) documents every option, its default and
why it exists. [`app_profile.rs`](windows/core/src/app_profile.rs) lists the
profiles and the reason for each setting they make.

## Logs and bug reports

Every process writes `mtld3d-logs/<exe>-<pid>.log` next to the executable,
never to the standard streams. `RUST_LOG` filters it: unset, everything logs
at `info`, and `RUST_LOG=mtld3d=warn` quiets the whole project. The log
targets, the levels and the F12 frame capture are described in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#logging).

A game that fails is reported as a
[`game-compat`](https://github.com/athei/mtld3d/labels/game-compat) issue,
with its log attached. Each release also carries `mtld3d-debug.tar.xz`, the
symbols that make a crash in that log readable. If mtld3d does not seem to
load at all, start with the [Troubleshooting](INSTALL.md#troubleshooting)
section of `INSTALL.md`.

## Games

[`docs/GAMES.md`](docs/GAMES.md) lists the games tested so far.

## Status

[`docs/STATUS.md`](docs/STATUS.md) lists what is implemented, what is not
yet, what never will be, and the divergences from D3D9 kept on purpose.

## Building from source

```sh
make setup        # once: toolchains and SDKs, but not Wine
make              # build every shipped binary
make install      # install into the Wine tree WINE_SDK names
```

[`docs/BUILDING.md`](docs/BUILDING.md) covers the prerequisites, the tests,
the arm64 Wine and the release bundle.

## Architecture

![Component diagram: game.exe calls d3d9.dll through the D3D9 COM API;
d3d9.dll, which links mtld3d-core, calls the function mtld3d_unix_call
exported by mtld3d.dll; mtld3d.dll crosses the Wine PE/Unix boundary into mtld3d.so, which drives
Metal. The PE side is i386 or x86_64, one chain per architecture; the host
side is Mach-O in Wine's own architecture.](docs/architecture.svg)

`d3d9.dll` implements the API and holds every piece of D3D9 knowledge,
`mtld3d.dll` is the PE shim that owns Wine's unix-call globals, and
`mtld3d.so` is a Metal abstraction layer on the host.
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) has the boundary contract, the
threading model and the debugging toolkits.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) is the operating manual, and
[`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) holds the code rules.

## License

[zlib](LICENSE).
