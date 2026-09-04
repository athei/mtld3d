# mtld3d

Direct3D 9 for Wine on macOS, backed by Metal.

mtld3d replaces Wine's `d3d9.dll`. The PE side implements the D3D9 API and
translates it into Metal command buffers that a native library executes on the
host. The goal is the fastest Direct3D 9 implementation for Wine on macOS.
Direct3D 8 on the same core is planned. Every other Direct3D version is a
non-goal: D3D10 and later are already served on macOS by Apple's D3DMetal and
by DXMT.

Conformance serves speed rather than defining it: where matching D3D9 exactly
would cost frame time, speed wins as long as no game breaks. Those trades are
listed in [`docs/STATUS.md`](docs/STATUS.md#kept-divergences).

## Requirements

What the test suites run on in CI, see
[`ci.yml`](.github/workflows/ci.yml):

- macOS 15 or macOS 26, on Apple Silicon or Intel.
- A Wine from [wine-build](https://github.com/athei/wine-build), the release
  CI pins, which is based on CrossOver 26. CrossOver 27 with its arm64 Wine
  has been tested by hand. Older Wine or CrossOver releases are not expected
  to work.
- A 64-bit prefix or bottle; 32-bit games run in it through WoW64.
- Rosetta 2 for an x86_64 Wine. An arm64 Wine translates x86 itself and needs
  none.

D3D9-era games do their floating-point math in x87 instructions, which
Rosetta 2 translates slowly. Under an x86_64 Wine, run the game with
[x87sidecar](https://github.com/athei/x87sidecar); the wine-build releases
carry the patch it needs.

## Installation

Download `mtld3d.tar.xz` from
[GitHub Releases](https://github.com/athei/mtld3d/releases). It installs
either as a Wine builtin, replacing the stock d3d9 for a Wine tree you own,
or as a native override per prefix, which is the route for CrossOver.
[`INSTALL.md`](INSTALL.md), also inside the bundle, has the steps for both,
the prefix markers a hand-installed builtin needs, and a troubleshooting
section that starts with the two log lines confirming mtld3d loaded.

## Configuration

Runtime options live in `mtld3d.conf` next to the game's `.exe`, read once at
`Direct3DCreate9`. The [sample](mtld3d.conf) documents every key with its
default and the reason it exists; a missing file means defaults. Every key
can also be set at launch through the `MTLD3D_CONFIG` environment variable,
which wins over the file. Keys worth knowing by name: `render.scale` renders
at a fraction of the presented size and lets MetalFX upscale, `present.maxFps`
caps the frame rate, and `color.hdr.enable` and `cursor.software` govern the
HDR present path and the cursor overlay that comes with it, both on by default
on a display with EDR headroom.

Below the file and the environment sits a third layer. A few games need
options nobody should have to discover, so mtld3d ships profiles for them,
matched on the executable name plus the version resource its vendor linked
in. A profile only supplies starting values, which the file and the
environment override key by key; `RUST_LOG=mtld3d::d3d9=info` names the
profile that matched. The reason behind every key is the comment on the
profile's entry in [`app_profile.rs`](windows/core/src/app_profile.rs).

- `gta-iv`, Grand Theft Auto IV: `adapter.spoof=amd`, `caps.dfFormats=false`,
  `depth.aliasSameSize=true`.
- `wow`, World of Warcraft 1.12 and 3.3.5: `query.flushImmediate=true`.

## Fullscreen

A fullscreen device sets the display mode the game picked, as native D3D9
does, and covers the monitor with a borderless window. To keep that mode-set
virtual, set Wine's `EmulateModeset` key in the prefix before launching the
game:

```sh
wine reg add 'HKCU\Software\Wine\X11 Driver' /v EmulateModeset /d Y /f
```

With it the game window is scaled onto the physical display, letterboxed when
the aspect differs, and mouse input is mapped into the mode; without it the
whole desktop switches resolution. `render.scale` multiplies on top of the
mode. The [Fullscreen section of `INSTALL.md`](INSTALL.md#fullscreen) has the
rest.

## Logging and diagnostics

Every process writes `<exe>-<pid>.log` into `mtld3d-logs` next to the
executable (`log.dir` moves it), never to the standard streams. `RUST_LOG`
filters it: unset, everything logs at `info`, and `RUST_LOG=mtld3d=warn` is
the single switch for the whole project. Pressing F12 in a game records the
next three frames as `[dump]` log lines and, with `MTL_CAPTURE_ENABLED=1` in
the game's environment, as a GPU trace beside the log. The log targets, the
levels and the F12 mechanics are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#logging).

## Tested games

| Game | Status |
| --- | --- |
| World of Warcraft 1.12 | Plays, `wow` profile, the primary target |
| World of Warcraft 3.3.5a | Plays, `wow` profile |
| Half-Life 2 | Plays |
| Grand Theft Auto IV | Plays, `gta-iv` profile |
| 3DMark05 | Runs end to end |
| Unigine Tropics | Runs |
| Gunmetal | Starts and benchmarks |

Games that fail are tracked as `game-compat` issues in the
[tracker](https://github.com/athei/mtld3d/issues); reports are welcome.

## Status

[`docs/STATUS.md`](docs/STATUS.md) lists what is implemented, what is not
yet, what never will be, and the divergences from D3D9 kept on purpose.

## Conformance and tests

`make test` runs the unit tests of the pure-Rust core natively on the host and
the end-to-end suite under Wine, one leg per PE architecture. `make
conformance` runs Wine's d3d9 test suite against the installed builtin and
gates on a per-site baseline; every remaining divergence is classified with a
written rationale in [`CONFORMANCE.md`](unix/conformance/CONFORMANCE.md).

## Building from source

`WINE_SDK` must point at a Wine build or install providing `wine`,
`winebuild` and `wineserver` plus its development tree, exported before any
make target. The toolchain and SDK versions CI pins are in
[`ci.yml`](.github/workflows/ci.yml) and the Makefile; the deployment target
is in [`unix/.cargo/config.toml`](unix/.cargo/config.toml).

```sh
make setup        # one-time toolchain bootstrap; does not install Wine
make              # PE side for i386 and x86_64, .so for x86_64 and arm64
make install      # install into the Wine tree WINE_SDK names
make test         # unit tests on the host, end-to-end suite under Wine
make check        # the pre-commit gate: fmt, clippy, audit, doc
```

`make bundle` packs the release tarball and its debug symbols. Every other
target and variable is documented in the Makefile beside its definition.

## Architecture

```
game.exe → d3d9.dll → mtld3d.dll → mtld3d.so
(PE, i386 or x64: one chain per arch)  (Mach-O, Wine's own arch)
```

`d3d9.dll` implements the API and holds every piece of D3D9 knowledge,
`mtld3d.dll` is the PE shim that owns Wine's unix-call globals, and
`mtld3d.so` is a pure Metal abstraction layer on the host. A frame flows
through three threads: the game's API thread snapshots state, an encoder
thread translates it into Metal commands, and a submit thread replays,
presents and commits. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) has the
boundary contract, the threading model, the workspace layout and the
debugging toolkits.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) is the operating manual: the gates and
how to read their output, how conformance work is organised, and what a pull
request is expected to contain. The conventions live in
[`docs/CONVENTIONS.md`](docs/CONVENTIONS.md). The short version: `make check`
and `make test` are green before every commit, and one pull request is one
change.

## License

[zlib](LICENSE).
