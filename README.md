# mtld3d

Direct3D 9 for Wine on macOS, backed by Metal.

mtld3d replaces Wine's `d3d9.dll`. The PE side implements the
application-facing D3D9 API and records commands for the native Unix runtime,
which translates them into Metal command buffers and submits them. The goal is
the fastest Direct3D 9 implementation for Wine on macOS. Direct3D 8 on the
same core is planned ([#788](https://github.com/athei/mtld3d/issues/788)).
Every other Direct3D version is a non-goal: D3D10 and later are already served
on macOS by Apple's D3DMetal and by DXMT.

Conformance serves speed: where matching D3D9 exactly would cost frame time
and no game breaks, speed wins. Those trades are listed in
[`docs/STATUS.md`](docs/STATUS.md#kept-divergences).

## Features

- **Shader translation.** Shader Model 1 to 3 and the fixed-function
  pipeline are translated to Metal Shading Language by mtld3d's own
  translator.
- **Multithreading.** The game's thread records each draw and state change into
  fixed records. Translation into Metal, submission and presentation run in a
  three-thread pipeline in the native Unix library that stays a fixed one frame
  ahead of the game, so latency is bounded. Those threads and the shader workers
  take no address space from a 32-bit game, and under an arm64 Wine they run as
  native arm64 code.
- **Background shader builds.** Shaders and pipelines a game uses for the
  first time build on worker threads. A frame waits once for all the builds
  it needs, not once per draw, and a pipeline whose shaders are ready builds
  ahead of shaders still queued. Asynchronous compilation goes further on a
  cold cache: a draw into a target the game rebuilds every frame is skipped
  for a frame or two instead of waited for, so the game does not stutter.
  Asynchronous compilation is on by default. Turned off, those draws wait
  with the rest.
- **Shader cache.** On by default. Translated shaders and pipelines are kept
  in a file next to the game and built again before the first frame of the
  next launch, so combinations seen before do not compile during play.
  Turned off, every launch compiles on first use.
- **Frame pacing.** Vsync and the frame cap both use Metal's minimum-duration
  present. Unlike pacing the game's thread or enabling display sync on the
  Metal layer, this works with ProMotion: the panel follows whatever rate the
  game sustains below its maximum. Vsync follows the game's present interval;
  the frame cap is off by default.
- **Upscaling.** Off by default. The game can render below the presented size
  and have MetalFX's spatial scaler upscale the result. A mip LOD bias, on by
  default, keeps texture detail at the presented size; turned off, textures
  are sampled for the smaller render size.
- **HDR output.** On by default. On a display with EDR headroom the frame is
  expanded into that headroom by inverse tone mapping that follows the live
  headroom. Turned off, or on a display without EDR headroom, the frame is
  presented as SDR.
- **Software cursor.** The cursor is drawn in its own overlay window
  independently of the game's frames, so it is not tied to the frame rate
  and has hardware-cursor latency. Under HDR it is tone-mapped like the
  frame, where the macOS cursor stays at SDR brightness. Showing or hiding it
  does not delay the next present by a refresh as the hardware cursor does.
  It is on by default whenever HDR output is active, and can be forced on or
  off. Turned off, the game gets the hardware cursor.
- **Game profiles.** The few games that need settings of their own get a
  built-in profile.

Every switch, with its default, is in [`mtld3d.conf`](mtld3d.conf).
[`docs/STATUS.md`](docs/STATUS.md) lists what is implemented, what is not
yet, what never will be, and the divergences from D3D9 kept on purpose.

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
  Rosetta 2 translates slowly; the wine-build releases carry its patch.

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
`Direct3DCreate9` and keeps it for that interface and its devices. A missing
file means defaults. The `MTLD3D_CONFIG` environment variable takes the same
entries, separated by semicolons.

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

## Logs, games and bug reports

Every process writes `mtld3d-logs/<exe>-<pid>.log` next to the executable,
never to the standard streams. `RUST_LOG` filters it: unset, everything logs
at `info`, and `RUST_LOG=mtld3d=warn` quiets the whole project. The log
targets, the levels and the Ctrl+Shift+F7 frame capture are described in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#logging). The capture hotkey
avoids the keys other programs take before the game sees them: a bare F12 is
Steam's default screenshot key, and the Metal HUD's menu takes Shift+F8 to
Shift+F12 and Ctrl+Shift+F9 to Ctrl+Shift+F12. It takes the Control key, not
Command, and holding Command blocks it, since Wine's Mac driver reports
Command as Alt. On a MacBook keyboard F7 needs Fn unless "Use F1, F2, etc.
keys as standard function keys" is on.

[`docs/GAMES.md`](docs/GAMES.md) lists the games tested so far and
[how to report one](docs/GAMES.md#reporting-a-game) that fails. Each release
also carries `mtld3d-debug.tar.xz`, the symbols that make a crash in a log
readable. If mtld3d does not seem to load at all, start with the
[Troubleshooting](INSTALL.md#troubleshooting) section of `INSTALL.md`.

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
d3d9.dll, which links mtld3d-core, records each frame and calls the function
mtld3d_unix_call exported by mtld3d.dll; mtld3d.dll crosses the Wine PE/Unix
boundary into mtld3d.so, the native runtime, whose encoder, submit, presenter
and worker threads drive Metal. The PE side is i386 or x86_64, one chain per
architecture, and with EC=1 an x64 game can load ARM64X builds of d3d9.dll
and mtld3d.dll instead; the host side is Mach-O in Wine's own
architecture.](docs/architecture.svg)

`d3d9.dll` implements the COM API, keeps the state the game can observe, and
records each frame into fixed records. `mtld3d.dll` is the PE shim that owns
Wine's unix-call globals. `mtld3d.so` is the native runtime on the host: an
encoder thread translates the recorded frame into Metal commands, a submit
thread commits them, a presenter thread presents, and worker threads compile
shaders and pipelines and prewarm the shader cache.
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) has the boundary contract, the
threading model and the debugging toolkits.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) is the operating manual, and
[`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) holds the code rules.

## License

[zlib](LICENSE).
