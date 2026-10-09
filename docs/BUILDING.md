# Building mtld3d

How to build mtld3d from source, install it into a Wine tree, run the gates
and tests, and pack a release. Every target and variable is documented in the
[Makefile](../Makefile) beside its definition; this file describes the
workflow and points there for the details.

## Prerequisites

- The Xcode Command Line Tools (`xcode-select --install`), for `clang`,
  `dsymutil` and `git`.
- [rustup](https://rustup.rs). `make setup` installs the toolchains through
  it.
- `python3`, which the Makefile uses to clone directory trees and which
  `make check` uses for two of its checks. The Command Line Tools provide one.
- Rosetta 2, to run the x86_64 Wine. `make setup` installs it when it is
  missing.
- A Wine tree for `WINE_SDK`, described below.

`WINE_SDK` names a Wine build or install tree holding `bin/wine`,
`bin/winebuild`, `bin/wineserver` and a `lib/wine` that carries Wine's link
archives for each PE architecture, since the build links against those
archives. Every target needs it, and the Makefile stops at once when it is
unset. `make install` and the test legs install into the same tree. A
[wine-build](https://github.com/athei/wine-build) release has everything,
including the `d3d9_test.exe` binaries that `make conformance` runs, and is
what CI uses.

The versions CI pins (the Rust toolchains, the cargo tools, nextest and the
wine-build release) are in the `env` block of
[`ci.yml`](../.github/workflows/ci.yml). The MSVC CRT and Windows SDK
versions are pinned in the Makefile, and the macOS deployment target is in
[`unix/.cargo/config.toml`](../unix/.cargo/config.toml).

## Setup

`make setup` bootstraps a machine once: the pinned Rust toolchains and
targets, the PE linker symlinks, xwin and the MSVC SDK in `/opt/xwin` (about
3 GB, needs `sudo`), nextest, cargo-edit and Rosetta 2. It does not install
Wine. Each piece is also a `setup-*` target of its own.

## Build

`make` builds every binary that ships:

- `d3d9.dll` and `mtld3d.dll` for i686 and x86_64, under
  `windows/target/<target>/<profile>/`
- `mtld3d.so` for x86_64 and arm64 hosts, under
  `unix/target/<target>/<profile>/`, each with its `.dSYM`

The profile is `release` by default. `PROD=1` selects `production`, the
profile that ships, with fat LTO and no debug assertions. That covers the C
and C++ the build scripts compile too: under `PROD=1` the Makefile passes
`-DNDEBUG` to every C and C++ object of both workspaces, among them snmalloc,
which the PE DLLs and `mtld3d.so` both use as their allocator. Every
production install and `make bundle` then run `production-assert-gate` on the
binaries they ship, which fails the build if one still carries a C or C++
assertion path; `make PROD=1 production-assert-gate` runs it on its own.
[`CONVENTIONS.md`](CONVENTIONS.md#production-carries-no-debug-assertions)
has the rule. Build production through make: a plain
`cargo build --profile production` gets no `NDEBUG`. `PERF`, `FP` and
`CRUMB` add perf telemetry, frame pointers and a breadcrumb ring for
debugging heap corruption; the Makefile says what each one does, and
[`ARCHITECTURE.md`](ARCHITECTURE.md#debugging-heap-corruption--mtld3d_crumb1-mmap-breadcrumb)
describes the breadcrumb ring.

## Instruction-set baselines

The PE DLLs are guest code, which an x86 translator runs: Rosetta under an
x86_64 Wine, and the translator an arm64 Wine carries (FEX, in CrossOver
27). [`windows/.cargo/config.toml`](../windows/.cargo/config.toml) sets their
CPU baseline, for the Rust and for the C and C++ that build scripts compile.
Neither goes past SSE4.2 or uses BMI; the comments there say why.

The i686 build targets `pentium4`, plain SSE2. The 32-bit build that matters
today is the one World of Warcraft 1.12 and 3.3.5a load under Rosetta, and
there `pentium4` measured faster than `nehalem`: 9 to 15 % less time per call
for the state setters, the shader-constant uploads and a vertex-buffer lock,
and 2.4 % off the buffer benchmark's frame, while the World of Warcraft 1.12
frame moved by +0.8 %, inside that run's noise, and the fixed-function draws
by +0.5 %. Two per-draw gauges of the perf summary rose: the draw snapshot
(`draw_snapshot_ms`) by 9.5 % and its key building (`draw_snapshot_keys_ms`)
by 7.3 %. Under the arm64 Wine's translator the same change cost about 7 %
on a clean draw and 2 to 3 % on a frame, so the baseline is worth measuring
again once that translator is in production use.

The x86_64 build targets `x86-64-v2`. Lowering it to plain `x86-64` measured
neutral under both translators, so the SSE4.2 ceiling neither costs nor buys
anything measurable.

## Install

`make install` puts the build into the Wine tree `WINE_SDK` names, and also
into `WINE_INSTALL_DIR` when that is set. `d3d9.dll` gets Wine's builtin
signature there, so it replaces Wine's own d3d9 for that tree. A tree that
keeps Direct3D implementations in their own subtrees (`lib/wine/d3d9/mtld3d`)
gets the build there, with markers in the default directories.

Wine only loads a builtin whose name has a marker in the prefix, and
`wineboot` writes the markers when it creates a prefix. So a build installed
for the first time reaches prefixes created after the install, or an existing
prefix after `wineboot -u`. [`INSTALL.md`](../INSTALL.md) explains the
mechanism.

## Gates and tests

`make check` is the lint and documentation gate, and `make test` runs the
unit tests and the end-to-end suite; both are green before every commit.
[`CONTRIBUTING.md`](../CONTRIBUTING.md#the-gates) lists what each one runs.

`make check` also lints and unit-tests a `PERF=1` build, the only build of a
gate that compiles the perf telemetry and its tests. It goes into `target/perf`
of each workspace rather than into the default target directory, since every
crate whose build script reads the setting rebuilds whenever it changes. The
first `make check` in a checkout builds it from scratch, which takes a few
minutes and about 5.6 GB of disk (3.3 GB under `windows`, 2.3 GB under
`unix`). The Makefile says which legs it runs beside `clippy-perf`.

The test legs run with Apple's Metal validation layer on and the Metal HUD
off; the Makefile says why beside `MTL_HUD_ENABLED`. Setting
`MTL_HUD_ENABLED=1` on the command line or in the environment turns the HUD
on for the end-to-end suite, to watch a run. The conformance runner keeps it
off for its test processes either way.

Every test leg installs into the shared Wine tree first, so two checkouts
testing at once overwrite each other's builds. `ISOLATED=1` avoids that: it
clones the SDK and the prefix into `.wine-isolated/` inside the checkout and
runs everything against the clones.

`make conformance` runs Wine's own d3d9 test suite against the installed
build and compares the failures per site against a baseline.

[`CONTRIBUTING.md`](../CONTRIBUTING.md#reading-a-test-run) explains how to
read a test run, and
[`CONFORMANCE.md`](../unix/conformance/CONFORMANCE.md) explains the
conformance suite and its baseline.

## arm64 Wine

Two opt-in switches build, install, test and benchmark mtld3d under an arm64
Wine such as CrossOver 27's. Neither runs in CI or goes into release bundles.

Every arm64 test, conformance and benchmark leg runs in a private clone of
the Wine that `WINE_ARM64` names, with a prefix created for the run, so the
legs never see each other's DLLs and never write into `WINE_ARM64` itself.
Only `make install` writes there, and like any install it reaches prefixes
created afterwards.

### `ARM64=1`

The i686 and x86_64 builds that ship, under the arm64 Wine.
`ARM64=1 make install` puts them into its `i386-windows` and
`x86_64-windows`, with the arm64 `mtld3d.so`, and replaces the `d3d9.dll` and
`mtld3d.dll` in its `aarch64-windows` with x64 markers, so that x64 processes
load the x86_64 build. [`INSTALL.md`](../INSTALL.md) says why. The PE DLLs
run translated there, and everything in `mtld3d.so`, the encoder, submit,
presenter and compile threads included, runs as native arm64 code.
`ARM64=1 make test` and `ARM64=1 make conformance` add a leg per architecture
under that Wine.

`ARM64=1` needs only `WINE_ARM64`.

### `EC=1`

`d3d9.dll` and `mtld3d.dll` as ARM64X images, whose ARM64EC half an x64 game
runs as native code instead of translating it. `EC=1 make` builds them under
`windows/target/arm64x/<profile>/`. `EC=1 make install` puts them into the
`aarch64-windows` of `WINE_ARM64`, with the arm64 `mtld3d.so`; the x86
builds install as they do without `EC=1`, and 32-bit games keep loading the
i686 build. `EC=1 make test` adds a leg that runs the x64 end-to-end suite
against the pair, `EC=1 make conformance` one that runs the x86_64
conformance binary against it, `EC=1 make check` lints both halves, and
`EC=1 make bundle` adds the pair to both archives.

The build needs a toolchain the x86 builds do not:

- The Rust targets `aarch64-pc-windows-msvc` and `arm64ec-pc-windows-msvc`,
  which `EC=1 make setup-rust` adds.
- [llvm-mingw](https://github.com/mstorsjo/llvm-mingw), for its ARM64 and
  ARM64EC CRT. `LLVM_MINGW` names the install.
- LLD 23 or newer as `lld-link`, such as Homebrew's (`brew install lld`).
  `ARM64X_LLD` names it. An older LLD links an image whose x64 view runs the
  ARM64 view's TLS callbacks.
- Wine's ARM64X link archives, which wine-build's "ARM64X link libraries"
  step stages as `dist/wine-arm64x`. `WINE_SDK_ARM64X` names that tree.

### Which target reads which path

None of the four paths has a default. Each comes from the environment, and a
target that needs one fails naming it when it is unset or points at the wrong
thing.

- `LLVM_MINGW` and `ARM64X_LLD` are read by the `EC=1` build.
- `WINE_SDK_ARM64X` is read by the `EC=1` build and by `EC=1 make check`.
- `WINE_ARM64` is read by the install, test and conformance targets under
  either switch, by `make bench` and `make bench-ab` under either switch, and
  always by `make bench-variants`. So `EC=1 make` and `EC=1 make check` run
  without it.

The Makefile header describes each path.

### Both switches

With both switches, `make install` puts all three builds in and x64
processes get the ARM64X one, and `make test` and `make conformance` run all
three arm64 legs. A benchmark measures one layout, so `make bench` and
`make bench-ab` take one switch at a time.

## Benchmarks

`make bench` runs the synthetic benchmarks once against this checkout.
`make bench-ab BASE=<ref>` builds `BASE` and this checkout and runs the
benchmarks against both in alternating rounds, which is how a change is
checked for a slowdown. `make bench-variants` compares one build in several
layouts instead: on the SDK's Wine, on the arm64 Wine and, with `EC=1`, as
the ARM64X build. All three build the production profile, and `bench-ab` and
`bench-variants` add the perf telemetry.
[`CONTRIBUTING.md`](../CONTRIBUTING.md#benchmarks) explains how to run them
and how to read the results.

## Release bundle

`make bundle` builds the production profile and packs two archives into
`windows/target/`: `mtld3d.tar.xz`, the bundle users install, and
`mtld3d-debug.tar.xz`, the symbols for the same binaries.
`make version-check TAG=vX.Y.Z` checks that a tag agrees with the version
both workspaces carry. The release procedure is in
[`CONTRIBUTING.md`](../CONTRIBUTING.md#cutting-a-release).
