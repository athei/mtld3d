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
- A Wine tree, described below.
- For the arm64 Wine, the ARM64X build's toolchain:
  - [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) 20260826 or newer,
    whose ARM64X CRT the ARM64X build links: its `aarch64-w64-mingw32`
    sysroot carries the ARM64 and ARM64EC halves together. An older release
    keeps ARM64EC in a sysroot of its own, which the link does not read.
    `LLVM_MINGW` names the install.
  - LLD 23 or newer as `lld-link`, such as llvm-mingw's own `bin/lld-link`
    (release 20261006 carries LLD 23.1.3, and CI links with it) or
    Homebrew's (`brew install lld`). `ARM64X_LLD` names it. An older LLD
    links an ARM64X image whose x64 view runs the ARM64 view's TLS
    callbacks.
- For an x86_64 Wine on Apple Silicon, Rosetta 2. `make setup` installs it
  when it is missing. The arm64 Wine runs x86 code through its own
  translator and does not need it.

The Makefile knows one Wine, the one `WINE_SDK` names. It serves as both the
Wine every leg runs under and the SDK every build reads: a tree holding
`bin/wine`, `bin/winebuild`, `bin/wineserver` and a `lib/wine` that carries
Wine's link archives for each PE architecture the builds link against.
Which Wine it is decides the Makefile's mode, read once from the Wine's own
`lib/wine`:

- The arm64 mode, for wine-build's arm64 Wine
  ([wine-build](https://github.com/athei/wine-build)), the primary target,
  whose `lib/wine` holds `aarch64-unix`. It also carries the mtld3d tree
  `lib/wine/d3d9/mtld3d` with the compatdb.so that loads it, and the ARM64X
  link archives. The i686, x86_64 and ARM64X builds all take `winebuild` and
  their link archives from it, and an x64 game loads the ARM64X build.
- The x86_64 mode, for an x86_64 Wine such as a wine-build x86_64 release,
  whose `lib/wine` holds `x86_64-unix`: the Wine an Intel Mac runs natively,
  and that CI's macos-15 image runs under Rosetta and its Intel image
  natively. An x64 game loads the x86_64 build there, and the ARM64X build,
  which such a Wine cannot run, is not built, linted, installed, tested or
  bundled, so neither llvm-mingw nor LLD nor the two ARM64X Rust targets are
  needed. Every make run in this mode prints a line that says so.

A build, lint or documentation target first checks that the Wine holds what
its mode needs and stops naming what is missing; a runtime-only Wine is
refused.

`WINE_SDK` has no default, and the Makefile stops at once when it is unset.
The variables of the earlier two-Wine layout, `WINE_X86_64`, `WINE_ARM64` and
`WINE_SDK_ARM64X`, and its switches `EC` and `ARM64`, are no longer read: one
of them in the environment or on the command line stops the run with a
message naming `WINE_SDK`.

The versions CI pins (the Rust toolchains, the cargo tools, nextest, the two
wine-build releases, the arm64 Wine that macos-26 runs and the x86_64 one
that macos-15 and the Intel image run, and llvm-mingw) are in the `env`
block of
[`ci.yml`](../.github/workflows/ci.yml). The MSVC CRT and Windows SDK
versions are pinned in the Makefile, and the macOS deployment target is in
[`unix/.cargo/config.toml`](../unix/.cargo/config.toml).

## Setup

`make setup` bootstraps a machine once: the pinned Rust toolchains and
targets (among them `aarch64-pc-windows-msvc` and `arm64ec-pc-windows-msvc`,
the two halves of the ARM64X build, which only the arm64 mode builds), the PE
linker symlinks, xwin and the
MSVC SDK in `/opt/xwin` (about 3 GB, needs `sudo`), nextest, cargo-edit and
Rosetta 2. It installs neither Wine, llvm-mingw nor LLD. Each piece is also
a `setup-*` target of its own.

## Build

`make` builds every binary that ships:

- `d3d9.dll` and `mtld3d.dll` for i686 and x86_64, under
  `windows/target/<target>/<profile>/`
- in the arm64 mode, `d3d9.dll` and `mtld3d.dll` as ARM64X images, under
  `windows/target/arm64x/<profile>/`: one image with an ARM64EC half, which
  an x64 game on an arm64 Wine runs as native code instead of translating it,
  and an ARM64 half for arm64 processes. Rust has no ARM64X target, so each
  crate is built once per half as a static library and the Makefile links
  both into one image (`windows-arm64x`, which says what that takes).
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

The x86 PE DLLs are guest code, which an x86 translator runs: Rosetta under
an x86_64 Wine, and the translator an arm64 Wine carries (FEX). The ARM64X
build is native code and has no such baseline.
[`windows/.cargo/config.toml`](../windows/.cargo/config.toml) sets the x86
builds' CPU baseline, for the Rust and for the C and C++ that build scripts
compile. Neither goes past SSE4.2 or uses BMI; the comments there say why.

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

`make install` puts the build into the Wine `WINE_SDK` names and into
`WINE_INSTALL_DIR` when that is set, a game's Wine tree, which may be an
arm64 or an x86_64 Wine; a `WINE_INSTALL_DIR` naming `WINE_SDK` itself is
written once. `d3d9.dll` gets Wine's builtin signature there, so it replaces
Wine's own d3d9 for that tree. A tree that keeps Direct3D implementations in
their own subtrees (`lib/wine/d3d9/mtld3d`) gets the build there, with
markers in the default directories.

Each Wine gets only what it loads, judged by its own `lib/wine`. An arm64
Wine, which holds `aarch64-unix`, gets the i686 build in `i386-windows`, the
ARM64X build in `aarch64-windows`, from where an arm64 Wine loads the
builtins of an x64 process, and the arm64 `mtld3d.so`. An x86_64 Wine, which
holds `x86_64-unix`, gets the i686 build in `i386-windows`, the x86_64 build
in `x86_64-windows` and the x86_64 `mtld3d.so`. The two 64-bit builds never
meet in one tree: an arm64 Wine never reads its `x86_64-windows`, and an
x86_64 Wine cannot run ARM64X. The install writes only into directories the
Wine already has and fails on a missing one rather than creating it. An
arm64 Wine has to be a wine-build arm64 bundle, whose compatdb.so loads
mtld3d from its subtree.

Wine only loads a builtin whose name has a marker in the prefix, and
`wineboot` writes the markers when it creates a prefix. So a build installed
for the first time reaches prefixes created after the install, or an existing
prefix after `wineboot -u`. [`INSTALL.md`](../INSTALL.md) explains the
mechanism.

## Gates and tests

`make check` is the lint and documentation gate, and `make test` runs the
unit tests and the end-to-end suite; both are green before every commit.
[`CONTRIBUTING.md`](../CONTRIBUTING.md#the-gates) lists what each one runs.
`make check` lints every PE build of the mode, in the arm64 mode the two
halves of the ARM64X one included.

`make test` runs two end-to-end legs under `WINE_SDK` in either mode, the
builds a game loads there: `test-e2e-i686`, the 32-bit suite against the i686
build, and
the x64 suite against the 64-bit build, `test-e2e-arm64x` against the ARM64X
build in the arm64 mode, whose ARM64EC half its processes run, and
`test-e2e-x86_64` against the x86_64 build in the x86_64 mode. Both install
into the one tree and share one prefix, since the two builds go into
different directories. The arm64 mode's `make test` runs a third leg,
`test-e2e-arm64`: the suite built for `aarch64-pc-windows-msvc`, native ARM64
processes, against the ARM64X build's ARM64 half, which the arm64 Wine's
compatdb gives them from arm64-11.19-2 on. A native ARM64 process loads a
builtin only behind a hybrid placeholder in the prefix's `system32`, and a
prefix made before an install stamped hybrid ones keeps the AMD64 copies it
took then, so this leg installs into a private clone of `WINE_SDK` with a
prefix created after the install. Its test binaries link like the x86 legs',
through `lld-link` and the xwin splat, which holds aarch64 libraries for them
(`make setup-xwin` adds them to a splat that lacks them). In the arm64 mode
`make test-e2e-x86_64` runs the x64 suite against the x86_64 build, emulated
under the same Wine; `make test` leaves it out there. That build would take
the ARM64X build's place in `aarch64-windows`, so the leg installs it into a
private clone of `WINE_SDK` with a prefix created for the run, and neither
leg ever writes into `WINE_SDK` itself. A leg of the other mode, such as
`test-e2e-arm64x` or `test-e2e-arm64` under an x86_64 Wine, stops saying
why.

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
clones `WINE_SDK` and the prefix into `.wine-isolated/` inside the checkout
once and runs everything against the clones.

`make conformance` runs Wine's own d3d9 test suite against the installed
builds and compares the failures per site against a baseline: the i386
`d3d9_test.exe` against the i686 build (`conformance-i686`) and the x86_64
one against the 64-bit build (`conformance-arm64x` in the arm64 mode,
`conformance-x86_64` in the x86_64 mode). Every leg, under either Wine, runs
one build of those tests, the one wine-build's arm64 Wine release publishes
in its `lib/wine/tests`: the baseline names each failing site by its
`file:line`, which moves between Wine versions, and a PE test runs under any
Wine. `WINE_TESTS` names that directory. In the arm64 mode it defaults to
`$WINE_SDK/lib/wine/tests`; in the x86_64 mode, whose Wine ships another
build of the tests, it has no default and the conformance targets stop
without it. Its `{i386,x86_64}-windows/d3d9_test.exe` are the two binaries.

The baseline's header names the Wine version those tests were built from,
and the conformance targets tell the runner that version: the first line of
`$WINE_TESTS/VERSION`, which wine-build writes beside the tests from
arm64-11.19-2 on (the `wine --version` string of their build) and CI's
x86_64-Wine images extract with them, else the `wine --version` of the Wine
tree `WINE_TESTS` lies in when that tree has a runnable `bin/wine`, as
`WINE_SDK` does in the arm64 mode. Tests with neither have no known version:
the run skips the comparison with the header and a re-record keeps it.

[`CONTRIBUTING.md`](../CONTRIBUTING.md#reading-a-test-run) explains how to
read a test run, and
[`CONFORMANCE.md`](../unix/conformance/CONFORMANCE.md) explains the
conformance suite and its baseline.

## arm64 Wine

wine-build's arm64 Wine runs x86 programs through FEX. It keeps mtld3d in
`lib/wine/d3d9/mtld3d`, and its compatdb.so prepends that tree to the builtin
search of every x86 process and, from arm64-11.19-2 on, of every native ARM64
one, mtld3d being its d3d9 default for them. An x64
program runs there as an ARM64EC process, which loads its builtins from
`aarch64-windows`, so the 64-bit modules sit in the tree's `aarch64-windows`
whatever their machine. The placeholders `make install` stamps for them in
the default `lib/wine/aarch64-windows` follow the rule wine-build's bundle
stamps its own by: hybrid ones (`winebuild -marm64x`, an ARM64 placeholder
with ARM64EC metadata) when every d3d9 tree that has the DLL holds an ARM64X
build of it, as after an install of the ARM64X build, and AMD64 ones
otherwise, as when the x86_64 build is installed there. A hybrid one also
takes a winebuild that writes the ARM64EC metadata into it, which
wine-build's does from arm64-11.19-2 on; an older one writes a plain ARM64
placeholder an x64 process refuses, and the install fails naming it before it
changes anything in the Wine. An x64 process accepts both kinds; a native
ARM64 one loads a name only behind a hybrid placeholder. A prefix copies the
placeholders when it is created and takes a changed one on its next
`wineboot -u`. A runtime-only arm64 Wine without that tree, such as
CrossOver's, is refused.

Under that Wine the PE side of a 32-bit game, and of the x86_64 build, runs
translated, while the ARM64X build's ARM64EC half runs as native code, and
everything in `mtld3d.so`, the encoder, submit, presenter and compile threads
included, runs as native arm64 code.

An x86_64 Wine runs the whole PE side as x86 code, natively on an Intel Mac
and through Rosetta 2 on Apple Silicon, and loads the x86_64 `mtld3d.so`.

### Which target reads which path

None of these paths has a default, but for `WINE_TESTS` in the arm64 mode.
Each comes from the environment or the command line, and a target that needs
one fails naming it when it is unset or points at the wrong thing.

- `WINE_SDK` is read by every target: the builds, lints and docs take
  `winebuild` and the link archives from it, and `make install` and every
  test, conformance and benchmark leg install into it or into a clone of it
  and run under it.
- `WINE_TESTS` is read by the conformance targets: the `lib/wine/tests` of
  wine-build's arm64 Wine release, whose `d3d9_test.exe` binaries every leg
  runs. `$WINE_SDK/lib/wine/tests` unless it is given in the arm64 mode,
  required in the x86_64 mode.
- `LLVM_MINGW` and `ARM64X_LLD` are read by the ARM64X build, which `make`,
  `make bundle`, `make stage` and every target that installs the ARM64X build
  run in the arm64 mode. The x86_64 mode reads neither.
- `WINE_INSTALL_DIR`, optional, is read by `make install` and by every leg
  that installs into `WINE_SDK`.

The Makefile header describes each path.

## Benchmarks

`make bench` runs the synthetic benchmarks once against this checkout.
`make bench-ab BASE=<ref>` builds `BASE` and this checkout and runs the
benchmarks against both in alternating rounds, which is how a change is
checked for a slowdown. Both measure the i686 build by default and the
mode's 64-bit build with `ARCH=arm64x` or, in the x86_64 mode,
`ARCH=x86_64`. `make bench-variants` compares two builds of this checkout
instead, in the arm64 mode: the x86_64 build, emulated, against the ARM64X
build, both under `WINE_SDK`. All three build the production profile, and
`bench-ab` and `bench-variants` add the perf telemetry.
[`CONTRIBUTING.md`](../CONTRIBUTING.md#benchmarks) explains how to run them
and how to read the results.

## Release bundle

`make bundle` builds the production profile and packs two archives into
`windows/target/`: `mtld3d.tar.xz`, the bundle users install, and
`mtld3d-debug.tar.xz`, the symbols for the same binaries. Built in the arm64
mode, the bundle carries all three PE builds: the ARM64X pair in
`wine/aarch64-windows` and its unmarked `d3d9.dll` in
`native/aarch64-windows` for an arm64 Wine, and the x86_64 pair for an
x86_64 one; [`INSTALL.md`](../INSTALL.md) says which goes where. Built in the
x86_64 mode, it carries no `aarch64-windows` entries, and an arm64 Wine takes
its x86_64 pair into `aarch64-windows` instead. CI's release job builds the
bundle in the arm64 mode, on macos-26 with the arm64 Wine, from the release
after 0.13.0 on; 0.13.0 and older were built in the x86_64 mode.
`make version-check TAG=vX.Y.Z` checks that a tag agrees with the version
both workspaces carry. The release procedure is in
[`CONTRIBUTING.md`](../CONTRIBUTING.md#cutting-a-release).
