ifndef WINE_SDK
$(error WINE_SDK is not set)
endif

# Clone the directory tree $(1) to $(2), cheapest mechanism first. On one APFS
# volume clonefile(2) takes a directory and clones the whole hierarchy in a
# single call, so the cost is the call and not the file count, where `cp -c -R`
# asks for a clone per file and pays for the walk: 12 GB over 88000 files is
# under a second against eleven. No stock command line tool exposes the
# directory form, hence python3 and ctypes. A source on another volume fails
# with EXDEV and one on a volume that is not APFS cannot clone at all, which is
# what the two `cp` fallbacks are for, the second of them copying the bytes.
# Prints nothing on stdout, so `$(shell ...)` can call it.
#
# Copying a single file takes `cp -c` instead: one clonefile(2) for the one
# file, and cp itself falls back to a byte copy when the destination is on
# another volume or on a volume that cannot clone.
define clone_tree
{ mkdir -p $$(dirname $(2)) && { python3 -c 'import ctypes, sys; lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib"); lib.clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32); sys.exit(0 if lib.clonefile(sys.argv[1].encode(), sys.argv[2].encode(), 0) == 0 else 1)' $(1) $(2) 2>/dev/null || { rm -rf $(2); cp -c -R $(1) $(2) 2>/dev/null; } || { rm -rf $(2); cp -R $(1) $(2); }; }; }
endef

# ISOLATED=1 runs every install-bearing target against a private clone of the
# Wine SDK inside this checkout: the SDK the tools come from, the tree the
# builds install into and the prefix the tests boot all move under
# `.wine-isolated`, so parallel worktrees, and the game bundle a maintainer is
# playing from, never see each other's builds. `clone_tree` seeds both for free
# on the same volume: the SDK from `WINE_SDK`, the prefix from the ambient
# `WINEPREFIX` (or `~/.wine`) so no prefix boots from scratch; each clone is
# made once and reused, and `clean-isolated` removes them. Without the knob,
# `make install` and every test leg keep targeting the shared trees, which is
# how the game gets a build.
ISOLATED_ROOT := $(CURDIR)/.wine-isolated

# Where a checkout records that it has an isolated environment: one line per
# checkout root, in the one directory every worktree of this repository shares,
# so `clean-isolated-orphans` can still find an environment whose checkout is
# not beside the others and whose server is no longer running. Empty when git
# does not answer, and every use of it is quoted, so nothing is read or written
# then. Two checkouts registering at once can only ever duplicate a line, which
# the reader folds away.
ISOLATED_REGISTRY := $(patsubst %,%/mtld3d-isolated-roots,$(shell git rev-parse --path-format=absolute --git-common-dir 2>/dev/null))

# Cleaning is the one goal that must not make what it is about to delete: the
# clones below are seeded at parse time, before any recipe runs, so `make
# ISOLATED=1 clean-isolated` would re-clone the SDK and the prefix and then
# remove them. Skipped only when every goal named is one of the cleaners, so a
# mixed command line still gets its clones.
ISOLATED_CLEANING := $(if $(MAKECMDGOALS),$(if $(filter-out clean-isolated clean-isolated-orphans clean-bench-ab,$(MAKECMDGOALS)),,1))

ifeq ($(ISOLATED),1)
ISOLATED_SDK_SOURCE := $(WINE_SDK)
ISOLATED_PREFIX_SOURCE := $(or $(WINEPREFIX),$(HOME)/.wine)
ifneq ($(ISOLATED_CLEANING),1)
$(shell [ -d $(ISOLATED_ROOT)/sdk ] || $(call clone_tree,$(ISOLATED_SDK_SOURCE),$(ISOLATED_ROOT)/sdk))
$(shell [ -d $(ISOLATED_ROOT)/prefix ] || [ ! -d $(ISOLATED_PREFIX_SOURCE) ] || $(call clone_tree,$(ISOLATED_PREFIX_SOURCE),$(ISOLATED_ROOT)/prefix))
$(if $(ISOLATED_REGISTRY),$(shell grep -qxF '$(CURDIR)' '$(ISOLATED_REGISTRY)' 2>/dev/null || echo '$(CURDIR)' >> '$(ISOLATED_REGISTRY)'))
endif
override WINE_SDK := $(ISOLATED_ROOT)/sdk
override WINE_INSTALL_DIR := $(ISOLATED_ROOT)/sdk
override WINEPREFIX := $(ISOLATED_ROOT)/prefix
export WINEPREFIX
$(info ==> ISOLATED=1: WINE_SDK=$(WINE_SDK) WINE_INSTALL_DIR=$(WINE_INSTALL_DIR) WINEPREFIX=$(WINEPREFIX))
endif
export WINE_SDK

# The prefix the test legs boot, named unconditionally: ISOLATED=1 points it at
# the clone above, otherwise it is the ambient `WINEPREFIX` or Wine's default.
# `configure-test-prefix` serialises on a lock file inside it, so the legs that
# share one prefix are exactly the legs that contend for it, and two checkouts
# with their own clones never wait on each other.
TEST_PREFIX      := $(or $(WINEPREFIX),$(HOME)/.wine)
TEST_PREFIX_LOCK := $(TEST_PREFIX)/.mtld3d-configure.lock

# The Wine tools this Makefile runs, named by absolute path out of the same
# install we build against and install into. Not found on PATH, and NOT by
# exporting one either: make execs a simple recipe line itself rather than
# through a shell, and that lookup reads make's own environment, so a PATH
# exported here is never consulted, so a bare name only resolves on a machine
# whose shell already has the SDK on PATH. The loader needs no PATH of its own;
# it finds wineserver and its libraries relative to its own location.
WINE       := $(WINE_SDK)/bin/wine
WINEBUILD  := $(WINE_SDK)/bin/winebuild
WINESERVER := $(WINE_SDK)/bin/wineserver

# Distribution bundles default to the production profile; PROD=0 overrides
# for a quick release-profile bundle. So do `make bench` and `make bench-host`:
# `release` carries debug assertions, whose checks would be most of what they
# measure. The default holds for every goal of the invocation, so each runs
# alone: beside `test` it would build the suite's layer without debug
# assertions.
ifneq ($(filter bundle,$(MAKECMDGOALS)),)
PROD ?= 1
endif
ifneq ($(filter bench,$(MAKECMDGOALS)),)
ifneq ($(filter-out bench,$(MAKECMDGOALS)),)
$(error `make bench` runs alone: its PROD=1 default would also apply to $(filter-out bench,$(MAKECMDGOALS)))
endif
PROD ?= 1
endif
# `make bench-ab` compares two builds of the profile the numbers are measured
# in, production with the perf summary, and both legs build exactly that: a
# leg of another profile would put the difference between the profiles into
# every verdict. So the defaults are fixed rather than defaults.
ifneq ($(filter bench-ab,$(MAKECMDGOALS)),)
ifneq ($(filter-out bench-ab,$(MAKECMDGOALS)),)
$(error `make bench-ab` runs alone: its PROD=1 PERF=1 would also apply to $(filter-out bench-ab,$(MAKECMDGOALS)))
endif
PROD ?= 1
PERF ?= 1
ifneq ($(PROD) $(PERF),1 1)
$(error `make bench-ab` builds both legs with PROD=1 PERF=1; PROD=$(PROD) PERF=$(PERF) would compare another profile)
endif
endif
ifneq ($(filter bench-host,$(MAKECMDGOALS)),)
ifneq ($(filter-out bench-host,$(MAKECMDGOALS)),)
$(error `make bench-host` runs alone: its PROD=1 default would also apply to $(filter-out bench-host,$(MAKECMDGOALS)))
endif
PROD ?= 1
endif

ifeq ($(PROD),1)
PROFILE  := production
$(info ==> PROD=1: cargo profile `production` (fat LTO + codegen-units=1))
else
PROFILE  := release
endif

ifeq ($(CRUMB),1)
export MTLD3D_CRUMB := 1
$(info ==> CRUMB=1: cfg(mtld3d_crumb) breadcrumb ring buffer enabled)
endif

ifeq ($(PERF),1)
export MTLD3D_PERF := 1
$(info ==> PERF=1: cfg(perf_tracking) compile-time perf telemetry enabled)
else ifeq ($(PERF),0)
export MTLD3D_PERF := 0
endif

# Frame pointers are opt-in: the toolchain default decides for a normal build,
# and FP=1 forces them on for the guest-pc sampling profiler, whose stack walks
# follow the guest frame-pointer chain, and without them every walk stops at
# the leaf, so a profile captured on an end-user machine cannot attribute cost
# to callers. Applies to the PE and unix builds alike; on aarch64 it changes
# nothing, the platform ABI mandates a frame pointer there.
#
# Written as a `cfg(all())` entry (matches every target) rather than a
# `RUSTFLAGS` environment variable: cargo joins rustflags arrays across config
# sources, and joins the cfg table with the triple table, so this ADDS to the
# flags each `.cargo/config.toml` pins per target (target-cpu, target features,
# linker search paths) where `RUSTFLAGS` would replace them wholesale.
#
# Build-script C/C++ is not covered; see the note on `-fno-omit-frame-pointer`
# in `windows/.cargo/config.toml`.
ifeq ($(FP),1)
$(info ==> FP=1: frame pointers forced (guest stack walks for the sampling profiler))
FRAME_POINTERS := --config 'target."cfg(all())".rustflags=["-C","force-frame-pointers=yes"]'
endif

# Which toolchains to use. Both float for a developer, who tracks stable and
# nightly and whose `rustc -V` is allowed to differ from anyone else's, and CI
# pins both to exact versions (see .github/workflows/ci.yml), because clippy runs
# nursery + pedantic with warnings denied, so a new release landing on a runner
# image would otherwise redden a run nobody here touched.
#
# Every cargo and rustc line below names its toolchain with rustup's `+` syntax,
# so the choice is visible where it is used and lives in no environment variable.
# It is also the only form that outranks an ambient RUSTUP_TOOLCHAIN, so a shell
# that pins one cannot silently redirect a build here. The cost is that a cargo
# line added later has to carry the prefix too.
RUST_STABLE  ?= stable
RUST_NIGHTLY ?= nightly

# The cargo-installed tools a build needs: `xwin` splats the MSVC SDK. Floating
# here and pinned by ci.yml, same split as the toolchains, because until this
# was a variable they were the one input a run took from whatever the registry
# happened to hold that day. Developer-only tooling is deliberately not in here,
# see `setup-dev`.
CARGO_TOOLS ?= xwin
# nextest runs the host-native unit tests and comes as a prebuilt universal
# binary (`setup-nextest`) rather than through `cargo install`, which saves a
# build machine the compile. `latest` floats for a developer; ci.yml pins a
# version. The end-to-end suite does not use it: its runner is `unix/e2e`.
NEXTEST_VERSION ?= latest

PE_i386     := i686-pc-windows-msvc
PE_x64      := x86_64-pc-windows-msvc
# Release/Wine targets for the unix half. Wine picks the `.so` out of
# `lib/wine/<cpu>-unix` by the arch of the Wine build that loads it, so there is
# one artifact per Wine host ISA: x86_64 for today's Wine on macOS, aarch64 for
# the arm64 Wine being prepared. The PE side is unaffected (it stays x86 either
# way), so both artifacts serve the same i386/x86_64 DLLs.
UNIX_TARGET_x64    := x86_64-apple-darwin
UNIX_TARGET_arm64  := aarch64-apple-darwin
UNIX_WINEDIR_x64   := x86_64-unix
UNIX_WINEDIR_arm64 := aarch64-unix
# Native host target for unit tests + clippy — whatever this machine is
# (aarch64-apple-darwin on Apple Silicon). Builds/runs without Rosetta. Expanded
# where it is used rather than up front, so a test runner that replays a staged
# build (`STAGE=`, below) and has no rustc at all still parses this file.
UNIX_NATIVE_TARGET  = $(shell rustc +$(RUST_STABLE) -vV | sed -n 's/^host: //p')
# The machine's own arch by the kernel's name, `arm64` or `x86_64`: which staged
# conformance runner binary is the native one here.
HOST_ARCH := $(shell uname -m)


OUT_i386       := windows/target/$(PE_i386)/$(PROFILE)
OUT_x64        := windows/target/$(PE_x64)/$(PROFILE)
OUT_unix_x64   := unix/target/$(UNIX_TARGET_x64)/$(PROFILE)
OUT_unix_arm64 := unix/target/$(UNIX_TARGET_arm64)/$(PROFILE)

# `make stage` packs everything a test machine needs out of one build:
# both PE arches, both unix `.so` builds, the e2e test binaries per PE arch,
# and the e2e and conformance runners for both host arches. `STAGE=<dir>`
# then points the install, e2e and conformance targets at an unpacked stage
# instead of a build: the OUT_* dirs become the staged ones, the build
# prerequisites drop, and both runners are the staged binaries for this
# machine, run against the staged test binaries.
# That is how CI builds once on a fast machine and fans the suites out over
# runners that carry no toolchain, and how a slow or old machine can run the
# suites against a build made elsewhere.
STAGE_DIR := $(CURDIR)/windows/target/stage
STAGE_OUT := $(CURDIR)/windows/target/mtld3d-stage.tar
ifdef STAGE
OUT_i386       := $(STAGE)/i386-windows
OUT_x64        := $(STAGE)/x86_64-windows
OUT_unix_x64   := $(STAGE)/$(UNIX_WINEDIR_x64)
OUT_unix_arm64 := $(STAGE)/$(UNIX_WINEDIR_arm64)
endif

XWIN_CACHE := $(HOME)/Library/Caches/xwin
# What the splat in /opt/xwin actually holds, written there by `setup-xwin` and
# read back by it to decide whether upstream has moved on. Stamped inside the
# splat rather than inferred from the download cache so the decision survives a
# cleared cache, and so CI can cache the splat alone instead of a second copy of
# the downloads next to it.
XWIN_STAMP := /opt/xwin/.xwin-packages
# The MSVC CRT and the Windows SDK are PINNED. Without these two flags `xwin`
# takes whatever is newest in Microsoft's channel manifest at the moment it
# runs, so an upstream release nobody here asked for would change the headers
# and import libs under a build, on one machine before another. Bump them
# deliberately; `setup-xwin` notices on its own, because the package names it
# stamps carry the versions.
XWIN_CRT_VERSION := 14.44.17.14
XWIN_SDK_VERSION := 10.0.26100

XWIN := xwin --accept-license --arch x86,x86_64 \
	--crt-version $(XWIN_CRT_VERSION) --sdk-version $(XWIN_SDK_VERSION) \
	--cache-dir $(XWIN_CACHE)

# Wine's own d3d9 test binaries, as published inside the SDK bundle by
# wine-build's bundle step (`make install` puts our builtin `d3d9.dll` into the
# same tree, which is what the tests then exercise). The conformance runner
# takes explicit paths and knows no Wine layout, so this is the only place the
# layout is written down. One binary per arch, one runner process per binary.
D3D9_TEST_i686   := $(WINE_SDK)/lib/wine/tests/i386-windows/d3d9_test.exe
D3D9_TEST_x86_64 := $(WINE_SDK)/lib/wine/tests/x86_64-windows/d3d9_test.exe

# Which unix `.so` arch the ambient WINE_SDK actually loads: Wine resolves unix
# libs by the arch of the Wine build itself, so a test or conformance leg needs
# that one installed and the other is inert. Probed from the loader (lazily, so
# only legs that use it pay for it) and overridable.
SDK_UNIX_ARCH ?= $(if $(findstring arm64,$(shell file -b $(WINE_SDK)/bin/wine)),arm64,x64)

# Hard-fail on any warning (cargo counts emitted warnings, including ones
# replayed from cache, and errors at the end of the run) — applied only to the
# `check` legs so normal builds and a plain `cargo +$(RUST_STABLE) clippy` stay
# warning-tolerant. Unlike `-D warnings` (via clippy args or RUSTDOCFLAGS)
# this changes no compiler flags, so check runs share the build cache with
# plain invocations.
DENY_WARNINGS := --config 'build.warnings="deny"'

# Sorted for its side effect of dropping a duplicate: under ISOLATED=1 both
# name the same clone, and the install loops must write it once.
INSTALL_DIRS := $(sort $(WINE_SDK) $(WINE_INSTALL_DIR))

# Both overridable, unlike the rest of these: the HUD and the validation layer
# are here to catch Metal misuse on a real GPU, and a caller running against a
# paravirtual one (a CI runner) has reason to turn them off, since neither has
# anything useful to say about a device that does not implement the counters they
# read.
export MTL_HUD_ENABLED ?= 1
export MTL_DEBUG_LAYER ?= 1
# Apple's variable, read by the Main Thread Checker that the test config below
# loads into every test process (`debug.mainThreadChecker=true`): with it set,
# a report of an AppKit call off the main thread ends the process at the call,
# on the offending thread, so the runner charges the death to the test that
# made it instead of the report scrolling past in stderr. Inert for a process
# that does not load the checker, which is every game.
export MTC_CRASH_ON_REPORT ?= 1
export WINEDLLOVERRIDES = mscoree,mshtml=
export WINEDEBUG=+msync
export WINEMSYNC=1

# Quiet locally, echoing under CI: a CI log is read after the fact by someone
# who cannot re-run the command, so the command line is the most useful thing
# in it. GitHub Actions (and every other runner) sets CI.
ifndef CI
MAKEFLAGS += --silent
endif

BUNDLE_NAME  := mtld3d.tar.xz
BUNDLE_OUT   := $(CURDIR)/windows/target/$(BUNDLE_NAME)
BUNDLE_STAGE := $(CURDIR)/windows/target/bundle

# Symbols for the same build, packed separately: nobody installing the layer
# needs them, but a crash report from a tester is unreadable without them. The
# `BUILD` file inside names the build, matching the identity every DLL logs on
# load, so an archive can be paired with a log without guessing.
DEBUG_NAME   := mtld3d-debug.tar.xz
DEBUG_OUT    := $(CURDIR)/windows/target/$(DEBUG_NAME)
DEBUG_STAGE  := $(CURDIR)/windows/target/bundle-debug
# Same expression `unix/shared/build.rs` stamps into the binaries, including the
# fall back to the manifest version outside a checkout, so the two cannot drift.
BUILD_ID     := $(shell git describe --tags --always 2>/dev/null || \
                        sed -n 's/^version = "\(.*\)"/v\1/p' windows/Cargo.toml)

# The tag a release is cut at. Defaults to the one on HEAD, so a checkout of a
# tag needs no argument; the release job passes the ref it was triggered by, and
# a maintainer about to cut a release passes the tag they are about to create
# (`make version-check TAG=v0.9.0`), which is the only way to check a bump
# before the tag exists.
TAG          ?= $(shell git describe --tags --exact-match 2>/dev/null)

# Target naming, two vocabularies with one rule each:
#
#   windows / unix   the two cargo workspaces, which are also the two
#                    directories. A leg split by workspace says which one it
#                    covers: `doc-unix`, `install-windows-i686`.
#   pe / native      the target family: a PE cross-compile versus this machine's
#                    own arch. A leg split by target says which one it takes:
#                    `clippy-pe-i686`, `clippy-native`. The clippy legs are named
#                    this way because the target is what actually splits them
#                    (see the comment there), and the native leg spans a crate
#                    from each workspace, so no workspace name would fit it.
#
# The arch suffix is the target's own spelling, i686 / x86_64 for PE and
# x64 / arm64 for the unix `.so`, matching the OUT_* variables above.
#
# Every target here is phony: the recipes write into cargo's target dirs and the
# Wine install, never into a file named after the target.
.PHONY: all windows windows-i686 windows-x86_64 unix unix-x64 unix-arm64 \
	install install-windows-i686 install-windows-x86_64 install-unix-x64 install-unix-arm64 \
	bundle version-check stage clean-isolated clean-isolated-orphans \
	configure-test-prefix configure-test-prefix-locked configure-test-prefix-session \
	configure-test-prefix-boot \
	test test-unit test-e2e-i686 test-e2e-x86_64 bench bench-ab bench-compare bench-shape clean-bench-ab bench-host bench-host-build \
	conformance conformance-i686 conformance-x86_64 \
	conformance-baseline conformance-baseline-i686 conformance-baseline-x86_64 \
	conformance-intel conformance-intel-i686 conformance-intel-x86_64 \
	conformance-scale conformance-scale-i686 conformance-scale-x86_64 \
	conformance-baseline-scale-i686 conformance-baseline-scale-x86_64 \
	conformance-baseline-intel-i686 conformance-baseline-intel-x86_64 \
	conformance-isolate fmt fmt-check clippy clippy-pe-i686 clippy-pe-x86_64 \
	clippy-native audit test-isolation test-e2e-discovery doc doc-windows doc-unix check clean upgrade \
	upgrade-incompat setup setup-rust setup-nextest setup-dev setup-xwin \
	setup-rosetta \
	xwin-dir fetch

all: windows unix

windows: windows-i686 windows-x86_64
unix: unix-x64 unix-arm64

# Per-arch build leaves. Each PE arch and each unix arch is independent, so a
# job (or a developer) that only needs one does not pay for the others; the
# aggregates above keep the everything-at-once habit.
#
# mtld3d.dll is only ever a Wine builtin (it owns the unix-call globals), so it
# gets the builtin signature at build time. d3d9.dll stays an ordinary native PE
# here: `install` and `bundle` mark their staged copies instead, so the build
# output can also be loaded as a native override in Wine distributions we don't
# control (CrossOver).
#
# The "fake DLL" placeholder is a prefix marker for the mtld3d builtin name,
# since Wine resolves a builtin by finding a marker for that NAME in the
# prefix's system directories before it loads the real module out of lib/wine.
# `install` does not need one and does not ship one: wineboot stamps a marker
# for every builtin it finds in lib/wine when it creates a prefix, and the
# install targets run first. Only `bundle` carries it, under prefix-markers/
# rather than wine/, for the case where that ordering does not hold: an
# existing prefix, or a Wine installation we do not control.
windows-i686:
	cd windows && cargo +$(RUST_STABLE) build --profile $(PROFILE) --target $(PE_i386) $(FRAME_POINTERS)
	$(WINEBUILD) --builtin $(OUT_i386)/mtld3d.dll
	$(WINEBUILD) --fake-module -o $(OUT_i386)/mtld3d.fake.dll -m32 --dll $(OUT_i386)/mtld3d.dll

windows-x86_64:
	cd windows && cargo +$(RUST_STABLE) build --profile $(PROFILE) --target $(PE_x64) $(FRAME_POINTERS)
	$(WINEBUILD) --builtin $(OUT_x64)/mtld3d.dll
	$(WINEBUILD) --fake-module -o $(OUT_x64)/mtld3d.fake.dll -m64 --dll $(OUT_x64)/mtld3d.dll

# On Mach-O the DWARF stays behind in the compiler's `.o` files, with only a
# debug map in the dylib pointing at them by absolute path; `dsymutil` walks
# that map and gathers the DWARF into a `.dSYM`, the shippable equivalent of
# an MSVC `.pdb`. Run it on a copy already named `mtld3d.so`, because it
# stamps the inner DWARF file after the input's basename and lldb looks it up
# by that name — renaming the bundle afterwards produces one lldb won't find.
unix-x64:
	cd unix && cargo +$(RUST_STABLE) build --profile $(PROFILE) --target $(UNIX_TARGET_x64) $(FRAME_POINTERS)
	cp -c $(OUT_unix_x64)/libmtld3d_unix.dylib $(OUT_unix_x64)/mtld3d.so
	rm -rf $(OUT_unix_x64)/mtld3d.so.dSYM
	dsymutil $(OUT_unix_x64)/mtld3d.so

unix-arm64:
	cd unix && cargo +$(RUST_STABLE) build --profile $(PROFILE) --target $(UNIX_TARGET_arm64) $(FRAME_POINTERS)
	cp -c $(OUT_unix_arm64)/libmtld3d_unix.dylib $(OUT_unix_arm64)/mtld3d.so
	rm -rf $(OUT_unix_arm64)/mtld3d.so.dSYM
	dsymutil $(OUT_unix_arm64)/mtld3d.so

install: install-windows-i686 install-windows-x86_64 install-unix-x64 install-unix-arm64

# Per-arch install leaves, named after the build leaf each one installs: a test
# leg installs the one PE arch it exercises plus the one unix `.so` its Wine
# loads, and only `install` (and `bundle`) covers everything.
#
# A Wine tree holds mtld3d in one of two layouts. A tree bundled with a compat
# database keeps every Direct3D implementation in its own subtree and picks one
# per process: `lib/wine/d3d9/mtld3d/<arch>` is prepended to the builtin search
# and the default dirs carry only fake-module markers (winebuild
# `--fake-module`, the placeholder wineboot stamps into the prefix so the
# prepended tree's DLL loads). An older tree loads straight from the default
# dirs. `MTLD3D_TREE` names the subtree a given root uses, and each leaf writes
# into it, re-stamping the markers in the new layout so a root whose markers
# an earlier install overwrote is whole again.
#
# The d3d9.dll copies under lib/wine get the builtin signature in place: the
# loader ignores unsigned PEs on the builtin search path. Symbols travel with
# each binary, the `.pdb` beside every PE and the `.dSYM` beside the `.so`, so a
# local crash symbolicates against the installed files with no extra flags.
define MTLD3D_TREE
if [ -d $(1)/lib/wine/d3d9/mtld3d ]; then echo $(1)/lib/wine/d3d9/mtld3d; else echo $(1)/lib/wine; fi
endef

install-windows-i686: $(if $(STAGE),,windows-i686)
	for dir in $(INSTALL_DIRS); do \
		tree=$$($(call MTLD3D_TREE,$$dir)) ; \
		mkdir -p $$tree/i386-windows ; \
		cp -c $(OUT_i386)/mtld3d.dll  $(OUT_i386)/mtld3d.pdb  $$tree/i386-windows/ ; \
		cp -c $(OUT_i386)/d3d9.dll    $(OUT_i386)/d3d9.pdb    $$tree/i386-windows/ ; \
		$(WINEBUILD) --builtin $$tree/i386-windows/d3d9.dll ; \
		if [ $$tree != $$dir/lib/wine ]; then \
			rm -f $$dir/lib/wine/i386-windows/d3d9.pdb $$dir/lib/wine/i386-windows/mtld3d.pdb ; \
			$(WINEBUILD) --fake-module -o $$dir/lib/wine/i386-windows/d3d9.dll   -m32 --dll $$tree/i386-windows/d3d9.dll ; \
			$(WINEBUILD) --fake-module -o $$dir/lib/wine/i386-windows/mtld3d.dll -m32 --dll $$tree/i386-windows/mtld3d.dll ; \
		fi ; \
	done

install-windows-x86_64: $(if $(STAGE),,windows-x86_64)
	for dir in $(INSTALL_DIRS); do \
		tree=$$($(call MTLD3D_TREE,$$dir)) ; \
		mkdir -p $$tree/x86_64-windows ; \
		cp -c $(OUT_x64)/mtld3d.dll   $(OUT_x64)/mtld3d.pdb   $$tree/x86_64-windows/ ; \
		cp -c $(OUT_x64)/d3d9.dll     $(OUT_x64)/d3d9.pdb     $$tree/x86_64-windows/ ; \
		$(WINEBUILD) --builtin $$tree/x86_64-windows/d3d9.dll ; \
		if [ $$tree != $$dir/lib/wine ]; then \
			rm -f $$dir/lib/wine/x86_64-windows/d3d9.pdb $$dir/lib/wine/x86_64-windows/mtld3d.pdb ; \
			$(WINEBUILD) --fake-module -o $$dir/lib/wine/x86_64-windows/d3d9.dll   -m64 --dll $$tree/x86_64-windows/d3d9.dll ; \
			$(WINEBUILD) --fake-module -o $$dir/lib/wine/x86_64-windows/mtld3d.dll -m64 --dll $$tree/x86_64-windows/mtld3d.dll ; \
		fi ; \
	done

# Both unix arches create the directory the Wine tree lacks: a Wine only ever
# loads the one matching its own build, so the other copy is inert, and a tree
# that later gains an arm64 loader is already served. In the subtree layout the
# default unix dir carries no mtld3d.so at all, so one an earlier install left
# there goes.
install-unix-x64: $(if $(STAGE),,unix-x64)
	for dir in $(INSTALL_DIRS); do \
		tree=$$($(call MTLD3D_TREE,$$dir)) ; \
		mkdir -p $$tree/$(UNIX_WINEDIR_x64) ; \
		cp -c $(OUT_unix_x64)/mtld3d.so        $$tree/$(UNIX_WINEDIR_x64)/ ; \
		rm -rf $$tree/$(UNIX_WINEDIR_x64)/mtld3d.so.dSYM ; \
		$(call clone_tree,$(OUT_unix_x64)/mtld3d.so.dSYM,$$tree/$(UNIX_WINEDIR_x64)/mtld3d.so.dSYM) ; \
		if [ $$tree != $$dir/lib/wine ]; then \
			rm -rf $$dir/lib/wine/$(UNIX_WINEDIR_x64)/mtld3d.so $$dir/lib/wine/$(UNIX_WINEDIR_x64)/mtld3d.so.dSYM ; \
		fi ; \
	done

install-unix-arm64: $(if $(STAGE),,unix-arm64)
	for dir in $(INSTALL_DIRS); do \
		tree=$$($(call MTLD3D_TREE,$$dir)) ; \
		mkdir -p $$tree/$(UNIX_WINEDIR_arm64) ; \
		cp -c $(OUT_unix_arm64)/mtld3d.so      $$tree/$(UNIX_WINEDIR_arm64)/ ; \
		rm -rf $$tree/$(UNIX_WINEDIR_arm64)/mtld3d.so.dSYM ; \
		$(call clone_tree,$(OUT_unix_arm64)/mtld3d.so.dSYM,$$tree/$(UNIX_WINEDIR_arm64)/mtld3d.so.dSYM) ; \
		if [ $$tree != $$dir/lib/wine ]; then \
			rm -rf $$dir/lib/wine/$(UNIX_WINEDIR_arm64)/mtld3d.so $$dir/lib/wine/$(UNIX_WINEDIR_arm64)/mtld3d.so.dSYM ; \
		fi ; \
	done

# The gate a release tag has to pass. Every shipped binary stamps `git describe`
# (BUILD_ID above), so a tag that disagrees with the version the workspaces
# carry ships DLLs naming a release nobody published, and nothing says so until
# a user reports a version that does not exist. The locks are read alongside the
# manifests because they record the version of every local crate: a bump that
# refreshed one workspace's lock and not the other leaves the tree disagreeing
# with itself.
#
# A local crate is one the lock gives no `source`. Clearing the name on that
# line is what keeps a registry crate's own version out of the comparison, since
# every package but ours has one.
define LOCK_VERSIONS
function chk() {
    if (n != "" && v != "\"" w "\"") {
        printf "version-check: %s records %s for %s, the manifest says \"%s\"\n", \
            FILENAME, v, n, w
        e = 1
    }
    n = ""; v = ""
}
/^name = /    { n = $$3 }
/^version = / { v = $$3 }
/^source = /  { n = "" }
/^$$/         { chk() }
END           { chk(); exit e }
endef
export LOCK_VERSIONS

version-check:
	test -n "$(TAG)" || { echo "version-check: HEAD carries no tag and none was given; pass the tag you are about to create, TAG=vX.Y.Z" >&2; exit 2; }
	for ws in windows unix; do \
		manifest=$$(sed -n 's/^version = "\(.*\)"/v\1/p' $$ws/Cargo.toml | head -1) ; \
		test "$$manifest" = "$(TAG)" || { echo "version-check: $$ws/Cargo.toml says $$manifest, the tag says $(TAG)" >&2; exit 1; } ; \
		awk -v w="$${manifest#v}" "$$LOCK_VERSIONS" $$ws/Cargo.lock >&2 || exit 1 ; \
	done
	echo "version-check: $(TAG) matches both workspaces"

# Distribution bundle, serving both install routes (see INSTALL.md, which is
# shipped inside): wine/ mirrors a Wine installation's lib/wine/ with every
# PE builtin-marked (drop-in for a Wine tree the user owns), while native/
# holds the unmarked d3d9.dll for the DLL-override route (required on
# CrossOver). The fake placeholders are the prefix markers for the custom
# mtld3d builtin name. wine/ carries both unix arches, so the same tree drops
# into an x86_64 or an arm64 Wine; each loads only the `.so` matching its own
# build.
#
# Two archives come out of one run: the bundle users install, and the symbols
# that make a crash report from one of them readable.
bundle: all
	rm -rf $(BUNDLE_STAGE) $(BUNDLE_OUT) $(DEBUG_STAGE) $(DEBUG_OUT)
	mkdir -p $(BUNDLE_STAGE)/wine/i386-windows
	mkdir -p $(BUNDLE_STAGE)/wine/x86_64-windows
	mkdir -p $(BUNDLE_STAGE)/wine/$(UNIX_WINEDIR_x64)
	mkdir -p $(BUNDLE_STAGE)/wine/$(UNIX_WINEDIR_arm64)
	mkdir -p $(BUNDLE_STAGE)/native/i386-windows
	mkdir -p $(BUNDLE_STAGE)/native/x86_64-windows
	mkdir -p $(BUNDLE_STAGE)/prefix-markers/syswow64
	mkdir -p $(BUNDLE_STAGE)/prefix-markers/system32
	cp -c $(OUT_i386)/mtld3d.dll           $(BUNDLE_STAGE)/wine/i386-windows/
	cp -c $(OUT_i386)/d3d9.dll             $(BUNDLE_STAGE)/wine/i386-windows/
	cp -c $(OUT_x64)/mtld3d.dll            $(BUNDLE_STAGE)/wine/x86_64-windows/
	cp -c $(OUT_x64)/d3d9.dll              $(BUNDLE_STAGE)/wine/x86_64-windows/
	# Markers live outside wine/, and already carry the name they need in the
	# prefix, so both routes are a plain copy into the matching system dir with
	# no rename. Keeping them out of wine/ is what stops `cp -R wine/*` from
	# dragging them onto the builtin search path, where wineboot would stamp a
	# second, useless marker under the name "mtld3d.fake.dll".
	cp -c $(OUT_i386)/mtld3d.fake.dll      $(BUNDLE_STAGE)/prefix-markers/syswow64/mtld3d.dll
	cp -c $(OUT_x64)/mtld3d.fake.dll       $(BUNDLE_STAGE)/prefix-markers/system32/mtld3d.dll
	$(WINEBUILD) --builtin $(BUNDLE_STAGE)/wine/i386-windows/d3d9.dll
	$(WINEBUILD) --builtin $(BUNDLE_STAGE)/wine/x86_64-windows/d3d9.dll
	cp -c $(OUT_unix_x64)/mtld3d.so        $(BUNDLE_STAGE)/wine/$(UNIX_WINEDIR_x64)/
	cp -c $(OUT_unix_arm64)/mtld3d.so      $(BUNDLE_STAGE)/wine/$(UNIX_WINEDIR_arm64)/
	cp -c $(OUT_i386)/d3d9.dll             $(BUNDLE_STAGE)/native/i386-windows/
	cp -c $(OUT_x64)/d3d9.dll              $(BUNDLE_STAGE)/native/x86_64-windows/
	cp -c $(CURDIR)/mtld3d.conf            $(BUNDLE_STAGE)/
	cp -c $(CURDIR)/INSTALL.md             $(BUNDLE_STAGE)/
	cp -c $(CURDIR)/LICENSE                $(BUNDLE_STAGE)/
	# The identity every binary stamps has to be this build's: a bundle
	# assembled from a stale target dir is how a release ships DLLs naming the
	# previous tag, and neither the version gate nor the archive itself can see
	# that. A binary carries exactly one of these, so finding this one is
	# enough. The two prefix markers are winebuild placeholders holding no code,
	# so they carry no stamp and are not swept.
	test -n "$(BUILD_ID)" || { echo "bundle: this build has no identity to check the binaries against" >&2; exit 1; }
	for f in $(BUNDLE_STAGE)/wine/i386-windows/*.dll \
	         $(BUNDLE_STAGE)/wine/x86_64-windows/*.dll \
	         $(BUNDLE_STAGE)/wine/$(UNIX_WINEDIR_x64)/mtld3d.so \
	         $(BUNDLE_STAGE)/wine/$(UNIX_WINEDIR_arm64)/mtld3d.so ; do \
		LC_ALL=C grep -a -q -F "$(BUILD_ID)" $$f || { echo "bundle: $$f is not stamped $(BUILD_ID); it was built at another commit, rebuild it" >&2; exit 1; } ; \
	done
	tar -cJf $(BUNDLE_OUT) -C $(BUNDLE_STAGE) wine native prefix-markers mtld3d.conf INSTALL.md LICENSE
	# The symbols for exactly these binaries, as a second archive. Laid out by
	# arch alone, with no wine/native split: debug info has no install route, and
	# the two d3d9.dll flavors are one binary with one `.pdb`.
	mkdir -p $(DEBUG_STAGE)/i386-windows
	mkdir -p $(DEBUG_STAGE)/x86_64-windows
	mkdir -p $(DEBUG_STAGE)/$(UNIX_WINEDIR_x64)
	mkdir -p $(DEBUG_STAGE)/$(UNIX_WINEDIR_arm64)
	echo $(BUILD_ID)                    > $(DEBUG_STAGE)/BUILD
	cp -c $(OUT_i386)/d3d9.pdb             $(DEBUG_STAGE)/i386-windows/
	cp -c $(OUT_i386)/mtld3d.pdb           $(DEBUG_STAGE)/i386-windows/
	cp -c $(OUT_x64)/d3d9.pdb              $(DEBUG_STAGE)/x86_64-windows/
	cp -c $(OUT_x64)/mtld3d.pdb            $(DEBUG_STAGE)/x86_64-windows/
	$(call clone_tree,$(OUT_unix_x64)/mtld3d.so.dSYM,$(DEBUG_STAGE)/$(UNIX_WINEDIR_x64)/mtld3d.so.dSYM)
	$(call clone_tree,$(OUT_unix_arm64)/mtld3d.so.dSYM,$(DEBUG_STAGE)/$(UNIX_WINEDIR_arm64)/mtld3d.so.dSYM)
	tar -cJf $(DEBUG_OUT) -C $(DEBUG_STAGE) BUILD i386-windows x86_64-windows \
		$(UNIX_WINEDIR_x64) $(UNIX_WINEDIR_arm64)

# The test hand-off (see STAGE above): the install inputs laid out exactly as
# the OUT_* dirs hold them, the e2e test binaries per PE arch, and the e2e and
# conformance runners for either host arch. A plain tar, since the artifact
# store drops execute bits and the `.dSYM` directories otherwise.
stage: all
	rm -rf $(STAGE_DIR) $(STAGE_OUT)
	mkdir -p $(STAGE_DIR)/i386-windows $(STAGE_DIR)/x86_64-windows
	mkdir -p $(STAGE_DIR)/$(UNIX_WINEDIR_x64) $(STAGE_DIR)/$(UNIX_WINEDIR_arm64)
	mkdir -p $(STAGE_DIR)/tests/i686 $(STAGE_DIR)/tests/x86_64
	mkdir -p $(STAGE_DIR)/e2e/x86_64 $(STAGE_DIR)/e2e/arm64
	mkdir -p $(STAGE_DIR)/conformance/x86_64 $(STAGE_DIR)/conformance/arm64
	cp -c $(OUT_i386)/mtld3d.dll $(OUT_i386)/mtld3d.pdb $(OUT_i386)/mtld3d.fake.dll \
		$(OUT_i386)/d3d9.dll $(OUT_i386)/d3d9.pdb $(STAGE_DIR)/i386-windows/
	cp -c $(OUT_x64)/mtld3d.dll $(OUT_x64)/mtld3d.pdb $(OUT_x64)/mtld3d.fake.dll \
		$(OUT_x64)/d3d9.dll $(OUT_x64)/d3d9.pdb $(STAGE_DIR)/x86_64-windows/
	cp -c $(OUT_unix_x64)/mtld3d.so $(STAGE_DIR)/$(UNIX_WINEDIR_x64)/
	$(call clone_tree,$(OUT_unix_x64)/mtld3d.so.dSYM,$(STAGE_DIR)/$(UNIX_WINEDIR_x64)/mtld3d.so.dSYM)
	cp -c $(OUT_unix_arm64)/mtld3d.so $(STAGE_DIR)/$(UNIX_WINEDIR_arm64)/
	$(call clone_tree,$(OUT_unix_arm64)/mtld3d.so.dSYM,$(STAGE_DIR)/$(UNIX_WINEDIR_arm64)/mtld3d.so.dSYM)
	$(call E2E_EXES_ASSIGN,$(call E2E_EXES,$(PE_i386))); cp -c $$exes $(STAGE_DIR)/tests/i686/
	$(call E2E_EXES_ASSIGN,$(call E2E_EXES,$(PE_x64))); cp -c $$exes $(STAGE_DIR)/tests/x86_64/
	cd unix && cargo +$(RUST_STABLE) build --profile $(PROFILE) -p mtld3d-e2e --target $(UNIX_TARGET_x64)
	cd unix && cargo +$(RUST_STABLE) build --profile $(PROFILE) -p mtld3d-e2e --target $(UNIX_TARGET_arm64)
	cp -c unix/target/$(UNIX_TARGET_x64)/$(PROFILE)/mtld3d-e2e $(STAGE_DIR)/e2e/x86_64/
	cp -c unix/target/$(UNIX_TARGET_arm64)/$(PROFILE)/mtld3d-e2e $(STAGE_DIR)/e2e/arm64/
	cd unix && cargo +$(RUST_STABLE) build --profile $(PROFILE) -p mtld3d-conformance --target $(UNIX_TARGET_x64)
	cd unix && cargo +$(RUST_STABLE) build --profile $(PROFILE) -p mtld3d-conformance --target $(UNIX_TARGET_arm64)
	cp -c unix/target/$(UNIX_TARGET_x64)/$(PROFILE)/mtld3d-conformance $(STAGE_DIR)/conformance/x86_64/
	cp -c unix/target/$(UNIX_TARGET_arm64)/$(PROFILE)/mtld3d-conformance $(STAGE_DIR)/conformance/arm64/
	tar -cf $(STAGE_OUT) -C $(STAGE_DIR) .

# E2E test environment overrides (the global exports above target the game):
#   - shaderCache.enable=false  — the on-disk cache would serve stale MSL across
#     runs, and the suite's processes must not race it.
#   - shader.asyncCompile=false: a draw whose build is in flight is kept in
#     its frame, whose submission waits for the build, instead of being left
#     out, so a test's first frame shows every draw it made. The builds still
#     run on the worker threads, which the waiting encoder steals from, so
#     the suite exercises them either way;
#     `async_compile.rs` turns the option on for the frames it leaves out.
#   - color.hdr.enable=false    the shipped default is on, and it resolves off
#     the running machine's panel, so leaving it would make the suite take the
#     HDR present route on an EDR Mac and the SDR one elsewhere. Pin it so the
#     results mean the same thing everywhere; the HDR route is exercised by real
#     runs and by the present-pipeline tests, not by the e2e assertions.
#   - debug.mainThreadChecker=true loads Apple's Main Thread Checker into each
#     test process, so an AppKit call off the main thread fails the run where
#     it is made (MTC_CRASH_ON_REPORT, exported above) instead of surfacing as
#     a rare death later in Wine's own code.
#   - WINEDEBUG= (empty)        — silence the +msync debug channel's per-call spam.
# MTL_DEBUG_LAYER stays on (inherited) so Metal API misuse fails the tests.
#
# SCALE=<n> additionally reruns the whole e2e suite at `render.scale = <n>`,
# i.e. rasterizing the back buffer smaller than the resolution D3D9 reports and
# letting MetalFX resolve it. Every coordinate the suite asserts on is in the
# reported space, so a passing scaled run is the evidence that the logical and
# render spaces stayed separate. `make test SCALE=0.75` is what one CI leg
# runs; try 0.5 and a non-dividing 0.67 too, since those catch rounding that a
# clean fraction hides.
#
# INTEL=1 reruns the whole e2e suite under every `intel.*` key, i.e. with the
# device answers an Intel/AMD Mac gives: packed 16-bit formats expanded, 32-bit
# float filtering denied, Managed buffers with didModifyRange after each write,
# and the 256-byte linear texture alignment. Every assertion has to hold there
# too; a test that probes a capability asks the device and asserts the answer
# it gets, which is also what lets the suite run on real Intel hardware.
#
# LOG_DIR=<absolute unix path> puts every test process's log file (and its GPU
# traces) in one directory instead of beside each test binary, so a machine
# that is only reachable through its artifacts (a CI runner) can hand the logs
# back, which every CI end-to-end leg does. The layer reads the directory on
# the PE side, where the unix root is drive Z, and the runner writes the whole
# stderr of every process that died there beside those logs and moves the
# layer's log of that process next to it, out of the layer's own retention.
# Abnormal exits after complete test reports also keep captured stdout, stderr
# and exit status together in a .process-log file, without changing verdicts.
# Ten files of each kind are kept per directory.
INTEL_CONF := intel.expandPacked16=true;intel.denyFloat32Filtering=true;intel.managedMemory=true;intel.linearAlign256=true
MTLD3D_CONF_TEST := shaderCache.enable=false;shader.asyncCompile=false;color.hdr.enable=false;debug.mainThreadChecker=true$(if $(SCALE),;render.scale=$(SCALE))$(if $(INTEL),;$(INTEL_CONF))$(if $(LOG_DIR),;log.dir=Z:$(LOG_DIR))
# Quoted: the config separator is `;`, which the shell would otherwise read as
# a command separator and run the rest of the line as its own command.
MTLD3D_TEST_ENV := MTLD3D_CONFIG='$(MTLD3D_CONF_TEST)' WINEDEBUG=

# Every leg that needs a prefix depends on `install-windows-*` first, so
# `mtld3d.dll` is already in lib/wine when wineboot creates the prefix and
# wine.inf's `11,,*` wildcard stamps its marker along with every other builtin.
# A prefix that predates the install does not get one and cannot load mtld3d;
# `wineboot -u`, or deleting it and re-running, fixes that.
# $(1) = the `reg add` arguments. Captured and shown only on failure.
define WINE_REG_ADD
out=$$($(WINE) reg add $(1) 2>&1) || { echo "wine reg add $(1) failed:" >&2; echo "$$out" >&2; exit 1; }
endef

# Reads one value back through the running wineserver, so the answer is the
# session's own rather than what the prefix last flushed to disk. False for a
# value that is absent, and for one that holds anything else. Wine's own
# chatter goes to stderr and is dropped; `reg query` prints the value name, its
# type and its data on one line, and ends that line the Windows way, so the
# pattern has to allow the carriage return the data is followed by.
# $(1) = the key, $(2) = the value name, $(3) = the data it must hold.
define WINE_REG_IS
$(WINE) reg query $(1) /v $(2) 2>/dev/null | grep -qE '^[[:space:]]*$(2)[[:space:]]+REG_[A-Z]+[[:space:]]+$(3)[[:space:]]*$$'
endef

# Reads one value out of the prefix's registry file instead, with no wineserver
# at all, so it is the prefix's answer only while no server holds the prefix:
# a running server writes the file when it likes. Wine keeps HKCU in `user.reg`,
# one section per key opening with the key's name in brackets, its backslashes
# doubled, and the time it was written, then one `"<name>"=<data>` line per
# value. The key and the line reach awk through the environment, which leaves
# backslashes as they are. False when the file or the value is absent.
# $(1) = the key under HKCU as the file spells it, $(2) = the whole value line.
define WINE_REG_FILE_IS
KEY='[$(1)]' LINE='$(2)' awk 'substr($$0, 1, 1) == "[" { here = ($$0 == ENVIRON["KEY"] || index($$0, ENVIRON["KEY"] " ") == 1) } here && $$0 == ENVIRON["LINE"] { found = 1 } END { exit !found }' '$(TEST_PREFIX)/user.reg' 2>/dev/null
endef


# Pin the prefix's display state for the tests, once per prefix rather than
# once per leg.
#
# Configuring ends the prefix's wineserver, so a leg that does it beside a
# running leg takes down the server that leg's test processes are attached to,
# and they end with their tests unaccounted for. Two guards make that
# impossible. `lockf` holds an exclusive flock(2) on a file in the prefix for
# the sub-make, waits for whoever holds it, and drops it when that sub-make
# ends however it ends, so a leg killed mid-configure leaves no stale lock. And
# the sub-make configures nothing when the prefix is already configured, which
# it is from the first leg of a checkout on.
configure-test-prefix:
	mkdir -p $(TEST_PREFIX)
	lockf -k $(TEST_PREFIX_LOCK) $(MAKE) configure-test-prefix-locked

# Runs only while that lock is held. A prefix whose persistent wineserver is up
# and whose three keys read back through it is configured, and there is nothing
# to restart. `wineserver -k0` is the server probe: it sends signal 0 to
# whatever holds the server's lock file, so it reports whether a server runs
# without touching one and without blocking, which `-w` cannot do here since
# the persistent server below never terminates on its own. The answer cannot go
# stale between the probe and the decision: the lock is what a leg holds while
# it ends and reboots a server, so nothing can be mid-boot here, and a probe
# that finds no server also proves this prefix has no test process attached to
# one.
#
# A prefix with no server whose registry file already holds the three values
# (a clone of a configured prefix, or one whose server a finished run stopped)
# needs no session to write them and so no restart: the server it boots reads
# them before it enumerates the display. It only needs its persistent server
# started, which saves the reg-add session and the restart after it, the
# larger part of configuring on a fresh clone.
configure-test-prefix-locked:
	if $(WINESERVER) -k0 >/dev/null 2>&1 \
		&& $(call WINE_REG_IS,'HKCU\Software\Wine\WineDbg',ShowCrashDialog,0x0) \
		&& $(call WINE_REG_IS,'HKCU\Software\Wine\X11 Driver',EmulateModeset,Y) \
		&& $(call WINE_REG_IS,'HKCU\Software\Wine\Mac Driver',RetinaMode,Y); \
	then exit 0; fi; \
	if ! $(WINESERVER) -k0 >/dev/null 2>&1 \
		&& $(call WINE_REG_FILE_IS,Software\\Wine\\WineDbg,"ShowCrashDialog"=dword:00000000) \
		&& $(call WINE_REG_FILE_IS,Software\\Wine\\X11 Driver,"EmulateModeset"="Y") \
		&& $(call WINE_REG_FILE_IS,Software\\Wine\\Mac Driver,"RetinaMode"="Y"); \
	then $(MAKE) configure-test-prefix-boot; exit; fi; \
	$(MAKE) configure-test-prefix-session

# The persistent server and the boot the session ends with, alone and loud:
# on this path no `reg add` has shown that Wine runs in the prefix, so a boot
# that fails, or leaves no server behind, fails the target with what wineboot
# said. Its output goes to a file rather than a pipe, since the residents
# wineboot leaves behind inherit it and would hold a pipe open forever.
configure-test-prefix-boot:
	$(WINESERVER) -p >/dev/null 2>&1 || { echo "wineserver -p for $(TEST_PREFIX) failed" >&2; exit 1; }
	log=$$(mktemp "$${TMPDIR:-/tmp}/mtld3d-wineboot.XXXXXX") || exit 1; \
	$(WINE) wineboot </dev/null >"$$log" 2>&1 && $(WINESERVER) -k0 >/dev/null 2>&1; status=$$?; \
	[ $$status -eq 0 ] || { echo "wine wineboot in $(TEST_PREFIX) failed or left no server:" >&2; cat "$$log" >&2; }; \
	rm -f "$$log"; exit $$status

configure-test-prefix-session:
	# Keep automated tests non-interactive and independent of mutable prefix
	# display settings. EmulateModeset prevents physical host mode changes;
	# RetinaMode keeps Win32 monitor geometry in the same physical-pixel space
	# as mtld3d's adapter modes. The first `reg add` also creates the prefix if
	# it does not exist yet, which is the case on a fresh machine. Quiet when
	# it works, since the prefix boot chatters; everything Wine said when it
	# does not, since that is the only account of why.
	$(call WINE_REG_ADD,'HKCU\Software\Wine\WineDbg' /v ShowCrashDialog /t REG_DWORD /d 0 /f)
	$(call WINE_REG_ADD,'HKCU\Software\Wine\X11 Driver' /v EmulateModeset /t REG_SZ /d Y /f)
	$(call WINE_REG_ADD,'HKCU\Software\Wine\Mac Driver' /v RetinaMode /t REG_SZ /d Y /f)
	# A wineserver session enumerates the display once, when its desktop
	# starts, and serves that geometry to every process in it afterwards. The
	# session the first `reg add` boots to create the prefix enumerates it
	# before RetinaMode is written, so a test process attaching to that session
	# reads monitor geometry in the point space rather than the physical-pixel
	# one the keys above pin. End it, so the next session enumerates with the
	# keys in place: `-k` shuts the server down cleanly, flushing the registry
	# on the way out, and `-w` covers the case where it had to be killed
	# outright. Both precede the persistent server below, which never
	# terminates on its own, so a `-w` after it would never return.
	-$(WINESERVER) -k >/dev/null 2>&1
	-$(WINESERVER) -w >/dev/null 2>&1
	# Pre-boot a persistent wineserver so individual test processes attach to it
	# instead of each paying boot cost (and briefly holding its stdio). Both
	# lines detach stdio: the persistent server (and the winedevice.exe residents
	# wineboot leaves behind) would otherwise inherit make's stdout/stderr and
	# hold a consumer pipe open forever, so `make test | ...` never sees EOF even
	# though make itself exited.
	-$(WINESERVER) -p >/dev/null 2>&1
	-$(WINE) wineboot >/dev/null 2>&1

test: test-unit test-e2e-i686 test-e2e-x86_64

# Host-native unit tests, built for this machine's native arch (no Rosetta).
# Needs no install and no wine at all, which is why it is its own leg: the
# windows workspace singles out mtld3d-core (its other members are PE-only and
# can't build for the host target) and must override its i686 default; the unix
# workspace already defaults to the host, so just run all of it.
test-unit:
	cd windows && cargo +$(RUST_STABLE) nextest run -p mtld3d-core -p mtld3d-types --target $(UNIX_NATIVE_TARGET)
	cd unix && cargo +$(RUST_STABLE) nextest run

# The e2e suite, one leg per PE arch: each installs the arch it exercises plus
# the unix `.so` this SDK's Wine loads, so the two legs are independent jobs.
#
# The suite is four test binaries per arch (`windows/tests/tests`: the
# one-process suite `e2e`, and `exit_code`, `unload` and `snmalloc_drift`,
# which need a process of their own), and the runner in `unix/e2e` runs each once under
# Wine, every test of a binary on `JOBS` threads of that one process, each
# with its own device. Only a failure, a crash or a hang costs another
# process: the runner marks the test it attributes the end to and runs the
# rest again. So a run is eight Wine launches, and its report counts every
# test rather than stopping at a summary.
#
# JOBS=<n> is how many tests run at once, each on its own thread with its
# own device. A parallel run is the only thing here that keeps several devices
# alive at once, which is what a game does with a launcher or an overlay
# beside it, so it is what holds the per-device rule: what one device owns
# is keyed by that device, and a process-wide registry keyed without one
# hands two devices each other's work.
# Rendering tests use borderless windows: Wine builds a framed window's
# title bar and controls on the AppKit main thread, serializing the tests
# even when their windows stay hidden. Window-management tests explicitly
# keep framed windows so that their style and teardown paths stay covered.
#
# The default of 4 assumes a Wine built from `cx-26-patched` at or after
# the winemac change that releases the D3DMetal client surfaces outside the
# window data lock; no wine-build release up to `cx-26.3.0-3` carries it.
# Without it the two locks are taken in both orders (`macdrv_DestroyWindow`
# holds the window data and takes win32u's surface lock,
# `update_client_surfaces` and `detach_client_surfaces` the other way
# round), so a window torn down on one thread while another thread creates,
# moves or destroys its own deadlocks the process. The report is still
# right there, because the watchdog charges the hang and runs the rest one
# at a time, but the suite takes ~130 s instead of ~10 s and JOBS=1 avoids
# it outright. CI passes JOBS=1 for a reason of its own, that device
# creation cannot overlap on a paravirtual GPU, so its value says nothing
# about this default.
#
# TIMEOUT=<secs> is how long a process may go without reporting a result
# before the runner kills it and charges the hang to the test that was
# running (default 60); the same bound covers a process that has closed
# stdout but will not exit. A process tree that keeps stderr open after the
# process is gone gets a one-second grace, not the bound: nothing it says
# after that is the test's.
#
# A scaled or Intel-variant run reports the whole suite instead of stopping at
# the first failure, and FAIL_FAST=0 asks for that on any run. The point of
# `SCALE` and `INTEL` is to survey which assertions still hold in the reported
# space or under the Intel answers, and one dependent test would otherwise
# hide every later test's behaviour there. The default run keeps fail-fast:
# there the first failure is a regression to fix, not a survey to read.
#
# FILTER='<patterns>' narrows the run to the tests whose id (`<binary>::<test
# path>`, e.g. `e2e::msaa::resolve_counts_edge_pixels`) contains any of the
# whitespace-separated patterns: `msaa::` is one file, `stencil` every test
# with the word. A filter that selects nothing passes.
JOBS ?= 4
TIMEOUT ?= 60
E2E_FLAGS := --jobs $(JOBS) --timeout $(TIMEOUT) $(if $(filter 0,$(FAIL_FAST))$(SCALE)$(INTEL),--no-fail-fast) $(if $(FILTER),--filter '$(FILTER)') $(if $(LOG_DIR),--log-dir '$(LOG_DIR)')

# The test binaries of one PE arch, from cargo's own account of what it built:
# `cargo test --no-run` prints one JSON message per artifact, and the test
# targets are the only ones with an executable. The JSON is held until Cargo
# succeeds, so a partial artifact list from a failed build never reaches a
# consumer. A glob over `deps/` would also pick up the stale hashes of earlier
# builds. --tests excludes examples, including the visible cursor probe, which
# are not libtest executables. Expanded inside a recipe, where the `$$(...)` is
# the shell's. From a stage the binaries are the staged ones. $(2) is extra
# cargo arguments (`make bench` names its profile).
define E2E_EXES_BUILD
cd windows && cargo +$(RUST_STABLE) test --tests --no-run -p mtld3d-tests --target $(1) --message-format=json-render-diagnostics
endef
define E2E_EXES
$$(output=$$(mktemp "$${TMPDIR:-/tmp}/mtld3d-e2e-exes.XXXXXX") || exit; \
	$(call E2E_EXES_BUILD,$(1)) $(2) > "$$output"; result_code=$$?; \
	if [ "$$result_code" -eq 0 ]; then \
		sed -n 's/^.*"executable":"\([^"]*\.exe\)".*/\1/p' "$$output"; result_code=$$?; \
	fi; \
	rm -f "$$output"; exit "$$result_code")
endef
define E2E_EXES_ASSIGN
exes="$(1)" || exit $$?
endef
E2E_EXES_i686   = $(if $(STAGE),$(STAGE)/tests/i686/*.exe,$(call E2E_EXES,$(PE_i386)))
E2E_EXES_x86_64 = $(if $(STAGE),$(STAGE)/tests/x86_64/*.exe,$(call E2E_EXES,$(PE_x64)))

# From a build here the runner is built and run through cargo, from its own
# workspace so its `.cargo/config.toml` applies; from a stage it is the staged
# binary for this machine's arch (compare `CONFORMANCE_BIN`).
E2E_RUNNER_DIR := $(if $(STAGE),.,unix)
E2E_RUNNER     := $(if $(STAGE),$(STAGE)/e2e/$(HOST_ARCH)/mtld3d-e2e,cargo +$(RUST_STABLE) run --profile $(PROFILE) -p mtld3d-e2e --)
# Builds that runner without running it, in its directory; nothing to build from a stage.
E2E_RUNNER_BUILD := $(if $(STAGE),true,cargo +$(RUST_STABLE) build --profile $(PROFILE) -p mtld3d-e2e)

test-e2e-i686: install-windows-i686 install-unix-$(SDK_UNIX_ARCH)
	$(MAKE) configure-test-prefix
	$(call E2E_EXES_ASSIGN,$(E2E_EXES_i686)); cd $(E2E_RUNNER_DIR) && $(MTLD3D_TEST_ENV) \
		$(E2E_RUNNER) --wine $(WINE) $(E2E_FLAGS) -- $$exes

test-e2e-x86_64: install-windows-x86_64 install-unix-$(SDK_UNIX_ARCH)
	$(MAKE) configure-test-prefix
	$(call E2E_EXES_ASSIGN,$(E2E_EXES_x86_64)); cd $(E2E_RUNNER_DIR) && $(MTLD3D_TEST_ENV) \
		$(E2E_RUNNER) --wine $(WINE) $(E2E_FLAGS) -- $$exes

# d3d9 conformance (NOT part of `make test`): run Wine's upstream d3d9 test exe
# against our installed builtin d3d9.dll, then diff per-site failure counts
# against the checked-in baseline. Many subtests fail by design, see
# unix/conformance/CONFORMANCE.md. The test exes ship inside the Wine SDK
# ($(D3D9_TEST_*) above); the runner takes them as paths and finds its
# baseline.txt in the crate dir.
#
# One arch per runner process, so the two gates are independent jobs. Every leg
# runs the same four subtests for its arch. The `-intel` legs run the same
# binary with `--variant intel`, which turns every `intel.*` config key on, and
# record under their own `<arch>+intel` baseline entries.
# From a build here the runner is built and run through cargo; from a stage it
# is the staged binary for this machine's arch. The assets directory is named
# explicitly either way: the runner's compiled-in default is the crate path on
# the machine that built it. The wineserver of the same SDK is named too: a
# subtest killed at its budget is sampled, and a subtest parked waiting for a
# server reply shows nothing of the server on its own stacks, so the server
# that serves this prefix is sampled beside it.
CONFORMANCE_BIN = $(if $(STAGE),$(STAGE)/conformance/$(HOST_ARCH)/mtld3d-conformance,cd unix && cargo +$(RUST_STABLE) run --profile $(PROFILE) -p mtld3d-conformance --)
CONFORMANCE_RUN = $(CONFORMANCE_BIN) --wine $(WINE_SDK)/bin/wine --wineserver $(WINESERVER) --assets $(CURDIR)/unix/conformance

# $(1) = arch (i686|x86_64), $(2) = extra runner args. Checks the exe up front
# so a bundle that predates the published test binaries says so, rather than
# failing four times inside the runner.
# LOG=<filter> is the RUST_LOG the test processes run under (the runner's
# default is `off`: the counts are the measurement). With
# MTLD3D_CONFORMANCE_RAW_DIR set, each process's log file lands in a directory
# beside its raw output, so LOG=debug there keeps what the layer did before a
# process ended without its summary, and the samples the runner takes of a
# process it kills at its budget, and of the wineserver of its prefix, land
# beside them.
define conformance_leg
	$(MAKE) configure-test-prefix
	test -f $(D3D9_TEST_$(1)) || { echo "$(D3D9_TEST_$(1)) is missing: re-bundle the Wine SDK, this one predates the published d3d9 test binaries" >&2; exit 2; }
	$(CONFORMANCE_RUN) --arch $(1) --exe $(D3D9_TEST_$(1)) $(2) $(if $(LOG),--log $(LOG))
endef

conformance: conformance-i686 conformance-x86_64

conformance-i686: install-windows-i686 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,i686)

conformance-x86_64: install-windows-x86_64 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,x86_64)

conformance-intel: conformance-intel-i686 conformance-intel-x86_64

conformance-intel-i686: install-windows-i686 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,i686,--variant intel)

conformance-intel-x86_64: install-windows-x86_64 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,x86_64,--variant intel)

# The same binary at `render.scale = 0.75`, the scaled leg CI runs for one
# arch; both arches exist as local tools and both record in the baseline.
conformance-scale: conformance-scale-i686 conformance-scale-x86_64

conformance-scale-i686: install-windows-i686 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,i686,--variant scale)

conformance-scale-x86_64: install-windows-x86_64 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,x86_64,--variant scale)

# Re-record the baseline. All six legs write the same baseline.txt (each
# replacing only its own entries), so they must run in sequence: hence
# recursive make in the recipe rather than prerequisites, which `-j` could
# interleave.
conformance-baseline:
	$(MAKE) conformance-baseline-i686
	$(MAKE) conformance-baseline-x86_64
	$(MAKE) conformance-baseline-intel-i686
	$(MAKE) conformance-baseline-intel-x86_64
	$(MAKE) conformance-baseline-scale-i686
	$(MAKE) conformance-baseline-scale-x86_64

conformance-baseline-i686: install-windows-i686 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,i686,--update-baseline)

conformance-baseline-x86_64: install-windows-x86_64 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,x86_64,--update-baseline)

conformance-baseline-intel-i686: install-windows-i686 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,i686,--variant intel --update-baseline)

conformance-baseline-intel-x86_64: install-windows-x86_64 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,x86_64,--variant intel --update-baseline)

conformance-baseline-scale-i686: install-windows-i686 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,i686,--variant scale --update-baseline)

conformance-baseline-scale-x86_64: install-windows-x86_64 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,x86_64,--variant scale --update-baseline)

# Flap characterization: run ONE subtest REPEAT times and print a per-site flap
# report (which sites fire deterministically vs flutter run-to-run), the
# evidence for tagging a site `flaky` in CONFORMANCE.md, and the way to make a
# subtest that dies one run in a few die in one sitting. Tune with ONLY (device|
# visual|stateblock|d3d9ex), ARCH (i686|x86_64), REPEAT (default 20), VARIANT
# (native|intel, the `intel` legs' forced answers) and LOG (see above). With
# MTLD3D_CONFORMANCE_RAW_DIR set, every run keeps its own raw output,
# `<leg>-<subtest>-<n>.log`, and its process's log file beside it.
ONLY ?= device
ARCH ?= i686
REPEAT ?= 20
VARIANT ?= native
conformance-isolate: install-windows-$(ARCH) install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,$(ARCH),--only $(ONLY) --repeat $(REPEAT) --variant $(VARIANT))

# The synthetic benchmarks (NOT part of `make test`), the `#[ignore]`d tests of
# `windows/tests/tests/e2e/bench_*.rs` (`windows/tests/COVERAGE.md` has a row
# for each file): frames shaped like World of Warcraft 1.12's and 3.3.5a's busy
# frames (`wow_112_busy_frame`, `wow_335a_busy_frame`), frames that each meet
# pixel shaders never seen before with the shader cache off, an EVENT-query
# throttle under the `wow` profile's query keys and under the D3D9 defaults
# (`query_poll_wow`, `query_poll_spec`), the API thread's cost of one call
# of each kind (`api_call_cost`), buffer locks and texture streaming at a
# game's rates (`dynamic_buffer_churn`, `texture_streaming`), and fresh
# processes from launch to their first frame on a populated shader cache
# (`cold_start`). They are `#[ignore]`d tests of the e2e
# binary, so the suite reports them ignored; this runs them alone, one at a
# time in one process, through the runner's `--ignored`, for one PE arch
# (ARCH, default i686, the arch the game ships). They measure and never assert
# on a time, so a run is red only when a device call fails.
#
# The layer and the benchmark binary are built with the production profile,
# the one that ships, since `release` compiles in debug assertions whose
# checks cost the encoder more than the frame does; PROD=0 measures `release`
# instead, and each report states the profile and whether debug assertions
# were on. The Metal validation layer and HUD are off: both cost frame time
# and neither is under test. The configuration is the suite's without the
# Main Thread Checker, then BENCH_CONFIG='key=value;key=value', appended last
# the way SCALE is, so its entries win over the ones before it (the stutter
# benchmark's own `shaderCache.enable=false` still wins over them). PERF=1
# builds the layer with its perf summary. Each benchmark writes
# `bench-<name>.txt` into LOG_DIR (default `.codex/evidence/bench`), beside
# the layer's log that a PERF=1 build's summary rows are copied from, and
# `bench-<name>.metrics` next to it, the same numbers plus the build's
# identity, its address-space samples, a scene's per-pass shape and, from a
# PERF=1 build, the counters of the layer's `perf-kv` lines as `perf.*`, one
# record per line for a program comparing two builds (`bench.rs` documents
# the format). A run first deletes both kinds of file left by the one
# before, prints the reports at the end and says where the metrics files
# are. FILTER='<patterns>' narrows the run as it does for `make test`, e.g.
# `FILTER=stutter`.
#
# BENCH_CORPUS='<path> <path>' names real shader caches (`mtld3d_shaders.bin`)
# for the benchmarks that read them: `cold_start` under `make bench` and the
# host emitter under `make bench-host` and `make bench-ab`, which name a corpus
# the same way. Its name is the name of the directory its file sits in, the
# path resolved first (a relative one against this checkout, so a bare file
# name is named after the checkout's directory, and `~` is not expanded), with
# every character other than an ASCII letter or digit turned into `_`. Each
# target copies the caches into a `corpus` directory of its own output, one
# `<name>/mtld3d_shaders.bin` apiece, and the benchmark takes the name from
# that directory; the directory is emptied first, so a run without BENCH_CORPUS
# measures none. Paths may not contain spaces. Two paths of one name, or a
# file at the filesystem root, stop the run before it builds.
BENCH_DIR := $(or $(LOG_DIR),$(CURDIR)/.codex/evidence/bench)
bench_corpus_name = $(shell printf '%s' '$(notdir $(patsubst %/,%,$(dir $(abspath $(1)))))' | tr -c 'A-Za-z0-9' '_')
BENCH_CORPUS_NAMES := $(foreach f,$(BENCH_CORPUS),$(call bench_corpus_name,$(f)))
BENCH_CORPUS_DUPLICATES := $(strip $(foreach n,$(sort $(BENCH_CORPUS_NAMES)),$(if $(filter-out 1,$(words $(filter $(n),$(BENCH_CORPUS_NAMES)))),$(n))))
ifneq ($(filter bench bench-host bench-ab,$(MAKECMDGOALS)),)
ifneq ($(words $(BENCH_CORPUS)),$(words $(BENCH_CORPUS_NAMES)))
$(error BENCH_CORPUS: a file at the filesystem root has no directory to name its corpus after)
endif
ifneq ($(BENCH_CORPUS_DUPLICATES),)
$(error BENCH_CORPUS: each corpus is named after its directory, and more than one path is named: $(BENCH_CORPUS_DUPLICATES))
endif
endif
# The staged copy of the corpus at path $(2) under the output directory $(1).
bench_corpus_copy = $(1)/corpus/$(call bench_corpus_name,$(2))/mtld3d_shaders.bin
# Empty $(1)/corpus and copy every BENCH_CORPUS entry into it.
bench_stage_corpus = rm -rf '$(1)/corpus'$(foreach f,$(BENCH_CORPUS), && mkdir -p '$(dir $(call bench_corpus_copy,$(1),$(f)))' && cp '$(abspath $(f))' '$(call bench_corpus_copy,$(1),$(f))')
# How long a benchmark process may print nothing before the runner kills it
# as hung. It bounds silence, not a process: `make bench` and a `make bench-ab`
# round run every benchmark in one process, and each benchmark prints a line
# where its measured frames start and another when it reports, so a round of
# many benchmarks is bounded per benchmark and never by their sum.
BENCH_TIMEOUT ?= 300
BENCH_TARGET := $(if $(filter x86_64,$(ARCH)),$(PE_x64),$(PE_i386))
BENCH_EXES = $(if $(STAGE),$(STAGE)/tests/$(ARCH)/*.exe,$(call E2E_EXES,$(BENCH_TARGET),--profile $(PROFILE)))
BENCH_CONF := shaderCache.enable=false;color.hdr.enable=false
MTLD3D_CONF_BENCH := $(BENCH_CONF);log.dir=Z:$(BENCH_DIR)$(if $(BENCH_CONFIG),;$(BENCH_CONFIG))
# Builds the benchmark binaries and names the one that carries the benchmarks,
# the one-process suite, in the shell variable `suite`.
define BENCH_SUITE_ASSIGN
$(call E2E_EXES_ASSIGN,$(BENCH_EXES)); suite=; \
	for exe in $$exes; do case $$exe in */e2e-*.exe|*/e2e.exe) suite=$$exe;; esac; done; \
	[ -n "$$suite" ] || { echo "no e2e test binary among: $$exes" >&2; exit 2; }
endef
bench: install-windows-$(ARCH) install-unix-$(SDK_UNIX_ARCH)
	$(MAKE) configure-test-prefix
	mkdir -p '$(BENCH_DIR)' && rm -f '$(BENCH_DIR)'/bench-*.txt '$(BENCH_DIR)'/bench-*.metrics
	$(call bench_stage_corpus,$(BENCH_DIR))
	$(BENCH_SUITE_ASSIGN); \
	cd $(E2E_RUNNER_DIR) && MTLD3D_CONFIG='$(MTLD3D_CONF_BENCH)' WINEDEBUG= MTL_DEBUG_LAYER=0 MTL_HUD_ENABLED=0 \
		$(E2E_RUNNER) --wine $(WINE) --jobs 1 --timeout $(BENCH_TIMEOUT) --ignored \
		$(if $(FILTER),--filter '$(FILTER)') --log-dir '$(BENCH_DIR)' -- $$suite
	if ls '$(BENCH_DIR)'/bench-*.txt >/dev/null 2>&1; then cat '$(BENCH_DIR)'/bench-*.txt; \
		echo "make bench: metrics in:"; ls -1 '$(BENCH_DIR)'/bench-*.metrics 2>/dev/null || echo "  none"; \
	else echo "make bench: no benchmark ran; FILTER='$(FILTER)' matches none of them"; fi

# `make bench-ab BASE=<ref>` measures a change: the same benchmarks against
# two builds of the layer, BASE and this checkout, interleaved, and then a
# verdict per metric. RUNS (default 5) is how many rounds each benchmark gets,
# a round being one run of either build back to back, the one that goes first
# alternating, and each build's run one process running every benchmark in
# libtest's order, the same in both.
# BENCH_SET picks the benchmarks: `wow` (the default) the ones that stand for
# the game this layer serves first, `full` every benchmark, and anything else
# a space-separated list of test-name filters, each selecting the benchmarks
# whose test path contains it (BENCH_SET=dynamic_buffer_churn rechecks one);
# a name the checkout does not carry yet is skipped with a note. ACCEPT=a,b names the
# exact metrics (draw counts and the like, which the workload fixes) whose
# change is expected, and BENCH_CONFIG is appended to both legs' configuration
# as it is for `make bench`. BASE=HEAD is an A/A run, the way to see how much
# the machine moves the numbers by itself, and with uncommitted changes it
# compares them against the commit they sit on.
#
# Both legs build PROD=1 PERF=1 into ISOLATED=1 trees of their own: BASE in a
# detached worktree under the main checkout's `.codex/worktrees`, kept for
# the next run and removed by `make clean-bench-ab`, and this checkout in its
# own `.wine-isolated`. Both trees are taken down and cloned again from the
# SDK and the prefix this invocation would otherwise use on every run, and
# both prefixes are configured by this checkout's `configure-test-prefix`, so
# a Wine rebuilt since the last run, an older BASE's prefix settings or a test
# run in this checkout's prefix cannot make the legs differ in more than the
# layer; the runner checks that both run one Wine before it starts, and the
# report names it. The clones cost about a second a tree; what costs is a
# prefix's first boot, and a clone of a configured prefix only boots once
# (see `configure-test-prefix-locked`). The two legs build at the same time,
# the base's output going to `build-base.log` in the run's directory, and the
# two prefixes are configured at the same time while the benchmark binary and
# the runner build, each prefix's output going to `configure-<leg>.log` there
# and shown when it fails. The persistent
# wineservers of both prefixes are stopped when the run ends, however it
# ends. The candidate's
# benchmark binary drives both legs: it links `d3d9` by name and nothing of
# the layer's, so the workload is the same on either side. Every run checks
# that the layer it loaded carries its leg's `git describe` stamp, computed
# here the way `unix/shared/build.rs` stamps it. The two legs must also load
# two different `d3d9.dll` images, except in a true A/A run (BASE is HEAD and
# the tree is clean), where a deterministic build may give both the same one.
#
# The runs, their layer logs and the report go to a directory of their own,
# `<base>-vs-<candidate>-<time>` under LOG_DIR (default
# `.codex/evidence/bench-ab` in the main checkout, so the results outlive the
# worktree that made them). After its timed rounds every benchmark whose
# metrics declare `shape` lines runs once more per leg with the pass trace on
# (the rest of the layer at warn but for the lines that name its build and
# the perf windows, `shape.rs` has the filter), untimed and stopped once its
# log holds the steady submissions, into
# `<leg>/shape/`, and the report compares the passes and load/store decisions
# of its steady submission between the legs; a difference fails the run
# unless ACCEPT names `shape` or `shape:<bench>`. Exit 1 is a regression
# or a shape change, 2 a run or a directory that cannot be trusted. `make
# bench-compare AB_DIR=<that directory>` judges it again, shapes included,
# with another ACCEPT for instance, into a report of its own
# (`report-compare-<time>.txt`) beside the one the run wrote.
#
# The host emitter benchmark (`make bench-host`) runs in rounds of its own,
# the first benchmark processes of the run, before the end-to-end benchmarks
# and whatever BENCH_SET names (only their short `--list` under Wine comes
# before it), so that no benchmark's Wine process is still exiting while it
# times host code. It is host
# code, so each leg builds and runs its own tree's `emit_corpus` with the
# leg's profile, and BENCH_CORPUS names the shader caches both legs read
# (none: the synthetic corpora alone); the same staged copies are linked into
# every end-to-end run's directory, so `cold_start` measures them as it does
# under `make bench`. A BASE whose Makefile has no
# `bench-host-build` predates the benchmark, and then neither leg runs it.
#
# Nothing else may run on the machine meanwhile, tests, builds and games
# included: the verdicts are only as good as the quiet of the machine.
RUNS ?= 5
BENCH_SET ?= wow
BENCH_SET_wow := wow_112_busy_frame wow_335a_busy_frame query_poll_wow query_poll_spec api_call_cost \
	dynamic_buffer_churn texture_streaming
BENCH_SET_full :=
# The runner's --bench filters for BENCH_SET: a named set's list (none for
# `full`), or BENCH_SET itself when it names no set.
BENCH_FILTERS = $(if $(filter wow full,$(BENCH_SET)),$(if $(BENCH_SET_$(BENCH_SET)),--bench '$(BENCH_SET_$(BENCH_SET))'),--bench '$(BENCH_SET)')
BENCH_CHECKOUT = $(patsubst %/,%,$(dir $(shell git rev-parse --path-format=absolute --git-common-dir)))
BENCH_AB_ROOT = $(abspath $(or $(LOG_DIR),$(BENCH_CHECKOUT)/.codex/evidence/bench-ab))
BENCH_CONF_AB := $(BENCH_CONF)$(if $(BENCH_CONFIG),;$(BENCH_CONFIG))
# What each leg's isolated clones are taken from: the SDK and prefix this
# invocation names, before an ISOLATED=1 of its own pointed them at clones.
BENCH_SDK_SOURCE := $(if $(filter 1,$(ISOLATED)),$(ISOLATED_SDK_SOURCE),$(WINE_SDK))
BENCH_PREFIX_SOURCE := $(if $(filter 1,$(ISOLATED)),$(ISOLATED_PREFIX_SOURCE),$(or $(WINEPREFIX),$(HOME)/.wine))
BENCH_LEG_MAKE = ISOLATED=1 PROD=1 PERF=1 WINE_SDK='$(BENCH_SDK_SOURCE)' WINEPREFIX='$(BENCH_PREFIX_SOURCE)'
BENCH_LEG_INSTALL = install-windows-$(ARCH) install-unix-$(SDK_UNIX_ARCH)
# `make` for the sub-makes that run beside each other in one bench-ab recipe
# line. A line that names `$(MAKE)` itself runs even under `make -n`, and these
# lines also build the benchmark binary and boot the prefixes, which a dry run
# must only print.
BENCH_SUBMAKE = $(MAKE)
# What each leg's sub-make builds: its install, and its host emitter benchmark
# when BASE has one.
BENCH_LEG_BUILD = $(BENCH_LEG_INSTALL) $(if $(BENCH_HOST_AB),bench-host-build)
# Says, in a recipe's shell, that the step $(2) failed when the status $(1) is
# not zero, and shows the end of its log $(3) in the run's directory.
bench_leg_failed = [ $(1) -eq 0 ] || { echo "make bench-ab: the $(2) failed; the end of $(BENCH_AB_OUT)/$(3):" >&2; tail -n 40 '$(BENCH_AB_OUT)/$(3)' >&2; }
# This checkout's `configure-test-prefix` on the isolated tree $(1), not
# isolated again: the tree is the leg's clone.
BENCH_LEG_CONFIGURE = ISOLATED= WINE_SDK='$(1)/sdk' WINE_INSTALL_DIR= WINEPREFIX='$(1)/prefix' configure-test-prefix
# Stops the persistent wineservers of both legs' prefixes, whatever state
# the run left them in; a leg that has no server is left as it is.
define BENCH_STOP_SERVERS
stop_servers() { for leg in '$(BENCH_BASE_ISO)' '$(ISOLATED_ROOT)'; do \
	[ -x "$$leg/sdk/bin/wineserver" ] && WINEPREFIX="$$leg/prefix" "$$leg/sdk/bin/wineserver" -k >/dev/null 2>&1 ; \
	done ; true ; }
endef
ifneq ($(filter bench-ab,$(MAKECMDGOALS)),)
ifeq ($(strip $(BENCH_SET)),)
$(error BENCH_SET is wow, full or a list of test-name filters, not empty)
endif
BENCH_BASE_SHA := $(shell git rev-parse --verify --quiet '$(BASE)^{commit}')
ifeq ($(BENCH_BASE_SHA),)
$(error `make bench-ab` needs BASE=<ref>, the commit to compare this checkout against$(if $(BASE),; $(BASE) names none))
endif
BENCH_BASE_SHORT := $(shell git rev-parse --short=12 $(BENCH_BASE_SHA))
# Whether the checkout differs from its commit. Untracked files do not count,
# the same as for the layer stamp: the build compiles what the tree tracks,
# and a file nothing tracks is not part of it.
BENCH_DIRTY := $(shell git status --porcelain --untracked-files=no)
BENCH_CAND_SHORT := $(shell git rev-parse --short=12 HEAD)$(if $(BENCH_DIRTY),-dirty)
# A true A/A run: BASE is this checkout's commit and nothing in the tree differs.
BENCH_SAME_IMAGE := $(if $(filter $(BENCH_BASE_SHA),$(shell git rev-parse HEAD)),$(if $(BENCH_DIRTY),,--allow-same-image))
BENCH_BASE_DIR := $(BENCH_CHECKOUT)/.codex/worktrees/bench-base-$(BENCH_BASE_SHORT)
BENCH_BASE_ISO := $(BENCH_BASE_DIR)/.wine-isolated
# The stamp `unix/shared/build.rs` compiles in: `git describe --tags --always`
# of the commit, never `--dirty`, so a candidate with uncommitted changes
# carries its commit's stamp.
BENCH_BASE_STAMP := $(shell git describe --tags --always $(BENCH_BASE_SHA))
BENCH_CAND_STAMP := $(shell git describe --tags --always)
BENCH_AB_OUT := $(BENCH_AB_ROOT)/$(BENCH_BASE_SHORT)-vs-$(BENCH_CAND_SHORT)-$(shell date +%Y%m%d-%H%M%S)
# Whether BASE carries the host emitter benchmark, read from its Makefile in
# git, since its worktree may not exist yet.
BENCH_HOST_AB := $(shell git show $(BENCH_BASE_SHA):Makefile 2>/dev/null | grep -q '^bench-host-build:' && echo 1)
BENCH_HOST_FLAGS = $(if $(BENCH_HOST_AB),--base-host '$(call BENCH_HOST_EXE,$(BENCH_BASE_DIR))' \
	--cand-host '$(call BENCH_HOST_EXE,$(CURDIR))' $(foreach f,$(BENCH_CORPUS),--host-corpus '$(call bench_corpus_copy,$(BENCH_AB_OUT),$(f))'))
endif
bench-ab:
	git -C '$(BENCH_CHECKOUT)' check-ignore -q '$(BENCH_BASE_DIR)' || \
		{ echo "make bench-ab: $(BENCH_CHECKOUT)/.codex is not ignored; add .codex/ to .git/info/exclude" >&2; exit 2; }
	[ -d '$(BENCH_BASE_DIR)' ] || git worktree add --detach '$(BENCH_BASE_DIR)' $(BENCH_BASE_SHA)
	test "$$(git -C '$(BENCH_BASE_DIR)' rev-parse HEAD)" = $(BENCH_BASE_SHA) || \
		{ echo "make bench-ab: $(BENCH_BASE_DIR) is not at $(BENCH_BASE_SHA); make clean-bench-ab removes it" >&2; exit 2; }
	$(call clean_isolated_at,$(BENCH_BASE_ISO))
	$(call clean_isolated_at,$(ISOLATED_ROOT))
	mkdir -p '$(BENCH_AB_OUT)'
	$(if $(BENCH_HOST_AB),,@echo "make bench-ab: BASE $(BENCH_BASE_SHORT) has no host emitter benchmark; neither leg runs it")
	$(BENCH_SUBMAKE) -C '$(BENCH_BASE_DIR)' $(BENCH_LEG_MAKE) $(BENCH_LEG_BUILD) > '$(BENCH_AB_OUT)/build-base.log' 2>&1 & base=$$!; \
		$(BENCH_SUBMAKE) $(BENCH_LEG_MAKE) $(BENCH_LEG_BUILD); cand=$$?; \
		wait $$base; base=$$?; \
		$(call bench_leg_failed,$$base,base leg's build,build-base.log); \
		[ $$base -eq 0 ] && [ $$cand -eq 0 ] || exit 2; \
		echo "make bench-ab: base leg built; its output is in $(BENCH_AB_OUT)/build-base.log"
	$(BENCH_SUBMAKE) $(call BENCH_LEG_CONFIGURE,$(BENCH_BASE_ISO)) > '$(BENCH_AB_OUT)/configure-base.log' 2>&1 & base=$$!; \
		$(BENCH_SUBMAKE) $(call BENCH_LEG_CONFIGURE,$(ISOLATED_ROOT)) > '$(BENCH_AB_OUT)/configure-cand.log' 2>&1 & cand=$$!; \
		( $(BENCH_SUITE_ASSIGN) && cd $(E2E_RUNNER_DIR) && $(E2E_RUNNER_BUILD) ); built=$$?; \
		wait $$base; base=$$?; wait $$cand; cand=$$?; \
		$(call bench_leg_failed,$$base,base prefix's configure-test-prefix,configure-base.log); \
		$(call bench_leg_failed,$$cand,candidate prefix's configure-test-prefix,configure-cand.log); \
		[ $$base -eq 0 ] && [ $$cand -eq 0 ] && [ $$built -eq 0 ] || { $(BENCH_STOP_SERVERS); stop_servers; exit 2; }
	$(if $(BENCH_CORPUS),$(call bench_stage_corpus,$(BENCH_AB_OUT)))
	$(BENCH_STOP_SERVERS); trap stop_servers EXIT; \
	$(BENCH_SUITE_ASSIGN); \
	cd $(E2E_RUNNER_DIR) && WINEDEBUG= MTL_DEBUG_LAYER=0 MTL_HUD_ENABLED=0 \
		$(E2E_RUNNER) bench-ab --out '$(BENCH_AB_OUT)' --runs $(RUNS) --timeout $(BENCH_TIMEOUT) \
		--base-wine '$(BENCH_BASE_ISO)/sdk/bin/wine' \
		--base-prefix '$(BENCH_BASE_ISO)/prefix' --base-stamp '$(BENCH_BASE_STAMP)' \
		--cand-wine '$(ISOLATED_ROOT)/sdk/bin/wine' \
		--cand-prefix '$(ISOLATED_ROOT)/prefix' --cand-stamp '$(BENCH_CAND_STAMP)' \
		--config '$(BENCH_CONF_AB)' $(BENCH_FILTERS) \
		$(BENCH_HOST_FLAGS) $(if $(BENCH_CORPUS),--corpus-dir '$(BENCH_AB_OUT)/corpus') \
		$(if $(ACCEPT),--accept '$(ACCEPT)') $(BENCH_SAME_IMAGE) --report '$(BENCH_AB_OUT)/report.txt' -- $$suite

bench-compare:
	test -n '$(AB_DIR)' || { echo "make bench-compare needs AB_DIR=<a directory make bench-ab wrote>" >&2; exit 2; }
	cd $(E2E_RUNNER_DIR) && $(E2E_RUNNER) bench-compare '$(abspath $(AB_DIR))' \
		$(if $(ACCEPT),--accept '$(ACCEPT)') --report '$(abspath $(AB_DIR))/report-compare-$(shell date +%Y%m%d-%H%M%S).txt'

# `make bench-shape GAME_LOG=<layer log> BENCH_METRICS=<bench-<name>.metrics>`
# calibrates a benchmark's scene against a game: it reads the last complete
# frame the game dumped with F12 into passes and prints them beside the
# benchmark's `shape` lines, flagging draw counts off by more than 10 %,
# fixed-function shares off by more than 10 points, textures per draw off by
# more than 1.0, and a different pass count. Exit 1 when anything is flagged.
# It runs nothing under Wine and judges no build; it is run by hand.
bench-shape:
	test -n '$(GAME_LOG)' -a -n '$(BENCH_METRICS)' || \
		{ echo "make bench-shape needs GAME_LOG=<a layer log with an F12 dump> and BENCH_METRICS=<a bench-<name>.metrics>" >&2; exit 2; }
	cd $(E2E_RUNNER_DIR) && $(E2E_RUNNER) bench-shape --game-log '$(abspath $(GAME_LOG))' \
		--metrics '$(abspath $(BENCH_METRICS))'

# The base worktrees `make bench-ab` keeps, each with its isolated Wine session
# and clones, taken down the way `clean-isolated` takes down a checkout's own.
clean-bench-ab:
	for wt in '$(BENCH_CHECKOUT)'/.codex/worktrees/bench-base-*; do \
		[ -d "$$wt" ] || continue ; \
		echo "==> $$wt" ; \
		$(call clean_isolated_at,$$wt/.wine-isolated) ; \
		git worktree remove --force "$$wt" ; \
	done

# The host emitter benchmark (NOT part of `make test` either):
# `windows/core/examples/emit_corpus.rs` times DXSO parsing and MSL emission
# and totals the size of the MSL, which is what Metal's compile time and so a
# first-use stutter grows with. It always runs a synthetic fixed-function
# corpus and a synthetic SM1-SM3 one; BENCH_CORPUS (above) adds each named
# `mtld3d_shaders.bin`, whose programmable records keep their DXSO, as a
# corpus of its own, staged under the `host` directory's `corpus`. Game
# caches stay out of the tree, so this is how their shaders get measured. Like `test-unit` it builds
# for this machine's own arch and needs no install and no Wine; like `bench` it
# builds with the production profile unless PROD=0 asks for `release`. The
# table goes to stdout, and each corpus writes `bench-host_emit_<corpus>.metrics`
# into the `host` directory under LOG_DIR (default `.codex/evidence/bench`),
# apart from the files `make bench` writes and deletes. `bench-host-build`
# builds the benchmark without running it, which is how `make bench-ab` gets
# each leg's own.
BENCH_HOST_DIR := $(BENCH_DIR)/host
# The benchmark `bench-host-build` builds in the checkout $(1).
BENCH_HOST_EXE = $(1)/windows/target/$(UNIX_NATIVE_TARGET)/$(PROFILE)/examples/emit_corpus
bench-host-build:
	cd windows && cargo +$(RUST_STABLE) build --profile $(PROFILE) -p mtld3d-core \
		--target $(UNIX_NATIVE_TARGET) --example emit_corpus

bench-host: bench-host-build
	mkdir -p '$(BENCH_HOST_DIR)' && rm -f '$(BENCH_HOST_DIR)'/bench-host_emit_*.metrics
	$(call bench_stage_corpus,$(BENCH_HOST_DIR))
	'$(call BENCH_HOST_EXE,$(CURDIR))' --metrics '$(abspath $(BENCH_HOST_DIR))' \
		$(foreach f,$(BENCH_CORPUS),'$(call bench_corpus_copy,$(BENCH_HOST_DIR),$(f))')

fmt:
	cd windows && cargo +$(RUST_NIGHTLY) fmt
	cd unix && cargo +$(RUST_NIGHTLY) fmt

fmt-check:
	cd windows && cargo +$(RUST_NIGHTLY) fmt --check
	cd unix && cargo +$(RUST_NIGHTLY) fmt --check

clippy: clippy-pe-i686 clippy-pe-x86_64 clippy-native

# Three independent clippy legs, split by the target they lint for so each is one
# job. No --all-targets on the whole-workspace PE runs: that would build every
# member's test targets for PE, including mtld3d-core's apple-only objc2 dev-deps
# (the SM3 corpus test), which hard `compile_error!` off Apple. Lib/bin only
# there; mtld3d-tests' integration tests aren't covered by those runs, so its own
# per-crate pass lints all its targets (it has no apple dev-deps).
clippy-pe-i686:
	cd windows && cargo +$(RUST_STABLE) clippy --target $(PE_i386) $(DENY_WARNINGS)
	cd windows && cargo +$(RUST_STABLE) clippy -p mtld3d-tests --target $(PE_i386) --all-targets $(DENY_WARNINGS)
	# `unix/shared` ships into both worlds but is a member of only the unix
	# workspace, so the windows legs build it as a plain path dependency with no
	# lint table: its `cfg(target_family = "windows")` arms (the PE image-ID
	# reader) would otherwise be linted by nothing. Lint it on the PE target that
	# reaches them.
	cd unix && cargo +$(RUST_STABLE) clippy -p mtld3d-shared --target $(PE_i386) $(DENY_WARNINGS)

clippy-pe-x86_64:
	cd windows && cargo +$(RUST_STABLE) clippy --target $(PE_x64) $(DENY_WARNINGS)
	cd windows && cargo +$(RUST_STABLE) clippy -p mtld3d-tests --target $(PE_x64) --all-targets $(DENY_WARNINGS)

# Everything that lints for this machine's own arch: mtld3d-core's test targets
# (the only place `#[cfg(test)]` blocks in the windows workspace are linted) and
# the whole unix workspace.
clippy-native:
	cd windows && cargo +$(RUST_STABLE) clippy -p mtld3d-core --target $(UNIX_NATIVE_TARGET) --all-targets $(DENY_WARNINGS)
	cd unix && cargo +$(RUST_STABLE) clippy --all-targets $(DENY_WARNINGS)

# The conventions clippy can't express: doc-comment shape, the Clone/Copy derive
# inventory, and the handful of patterns that are banned or confined to a known
# set of files. See docs/CONVENTIONS.md § Mechanical audit.
audit:
	./scripts/audit.sh

test-isolation:
	python3 scripts/test-isolation.py

test-e2e-discovery:
	python3 scripts/test-e2e-discovery.py

# rustdoc's own lints, which no other target sees: broken and private intra-doc
# links, malformed HTML in doc comments. `audit` gates the *shape* of a doc block
# and clippy gates its prose; only rustdoc knows whether its links resolve.
# `build.warnings` covers rustdoc warnings too, so no RUSTDOCFLAGS needed.
doc: doc-windows doc-unix

# The windows workspace is documented for a PE target, not the host: d3d9 and the
# shim are `cdylib`s with raw-dylib imports and only build for *-pc-windows-msvc,
# so a host run would silently skip them. i686 covers every member.
doc-windows:
	cd windows && cargo +$(RUST_STABLE) doc --no-deps --target $(PE_i386) $(DENY_WARNINGS)

doc-unix:
	cd unix && cargo +$(RUST_STABLE) doc --no-deps $(DENY_WARNINGS)

# One command to run before every commit: formatting, the full clippy sweep, the
# conventions audit, the Makefile regressions, and the doc build. fmt-check first
# (fast, fails early on drift); clippy reuses the target above; audit and the
# Makefile regressions are fast; doc stays last. Each leg is also its own target,
# so CI runs them as parallel jobs instead of this sequence.
check:
	$(MAKE) fmt-check
	$(MAKE) clippy
	$(MAKE) audit
	$(MAKE) test-isolation
	$(MAKE) test-e2e-discovery
	$(MAKE) doc

clean:
	cd windows && cargo +$(RUST_STABLE) clean
	cd unix && cargo +$(RUST_STABLE) clean

# Take down what ISOLATED=1 left under the isolated root $(1): the Wine session
# of its private prefix first, then the clones. Three steps, because a session
# is more than its server. Signalling the server alone leaves the service
# processes the prefix keeps (services.exe, the two winedevice.exe, plugplay.exe,
# svchost.exe, rpcss.exe) running, reparented to launchd, with their cwd in a
# prefix that is about to be deleted and their images mapped out of an SDK clone
# that is about to go with it.
#
# So the session is ended through the root's own `wineserver -k`, which takes
# the clients down with the server. The prefix is named on the command line
# rather than inherited, since the caller's WINEPREFIX is its own checkout's and
# every root this is called for may be another's. `-k` waits for the server to
# exit without a bound, and it execs a binary out of the tree that is about to
# go, so it runs in the background and gets five seconds before the helper
# itself is killed: a wedged prefix holds this root and no other. With no server
# up it returns at once and boots none.
#
# The server is then signalled by pid, which is how a root whose SDK clone is
# already gone is still taken down, and how one that did not answer `-k` ends:
# SIGTERM, five seconds, SIGKILL. A server is recognised by its executable path
# (`ps -o comm=`, one field however many spaces it holds), never by splitting a
# command line into words.
#
# Whatever is left is matched by what it holds open. A service process carries
# the Windows path `C:\windows\system32\winedevice.exe` as its `comm`, so its
# name says nothing about which checkout it belongs to, but its cwd is in the
# prefix and its images are mapped out of the SDK clone, and lsof reports both
# by path even once the tree is removed. Only a cwd and a mapped image count, so
# something that merely has a file under the root open is not taken for a member
# of the session. Each match is SIGTERMed, given five seconds, then SIGKILLed,
# and it is asked again for the same paths immediately before either signal, so
# a pid reused in between is left alone.
#
# The clones therefore only go once nothing is running out of them, and the one
# thing executed out of a directory that is about to be deleted is the bounded
# `-k` above. A delete can still meet files written into the tree meanwhile
# (a helper process on its way out, or macOS writing a `.DS_Store`), so it is
# tried again every half second for five seconds; a root still there after
# that fails the call, naming the processes that hold files under it (an
# `lsof +D` walk, affordable only on that path) or saying that none does. One logical shell line, so a caller that found a root of its own
# can run it inside a loop; $(1) arrives unquoted and is quoted here.
define clean_isolated_at
iso_root="$(1)" ; \
holds_isolated() { \
	lsof -n -P -w -a -p "$$1" -d cwd,txt -F n 2>/dev/null \
	| awk -v root="$$iso_root/" 'substr($$0, 1, 1) == "n" && substr($$0, 2, length(root)) == root { held = 1 } END { exit !held }' ; \
} ; \
if [ -x "$$iso_root/sdk/bin/wineserver" ]; then \
	WINEPREFIX="$$iso_root/prefix" "$$iso_root/sdk/bin/wineserver" -k >/dev/null 2>&1 & \
	ender=$$! ; \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
		kill -0 $$ender 2>/dev/null || break ; \
		sleep 0.5 ; \
	done ; \
	kill -9 $$ender 2>/dev/null || true ; \
	wait $$ender 2>/dev/null || true ; \
fi ; \
for pid in $$(pgrep -f '/\.wine-isolated/sdk/bin/wineserver'); do \
	[ "$$(ps -o comm= -p $$pid 2>/dev/null)" = "$$iso_root/sdk/bin/wineserver" ] || continue ; \
	kill $$pid 2>/dev/null || true ; \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
		kill -0 $$pid 2>/dev/null || break ; \
		sleep 0.5 ; \
	done ; \
	kill -9 $$pid 2>/dev/null || true ; \
done ; \
left=$$(lsof -n -P -w -d cwd,txt -F pn 2>/dev/null \
	| awk -v root="$$iso_root/" '/^p/ { pid = substr($$0, 2); held = 0; next } held { next } /^n/ { if (substr($$0, 2, length(root)) == root) { print pid; held = 1 } }') ; \
for pid in $$left; do \
	holds_isolated $$pid && kill $$pid 2>/dev/null || true ; \
done ; \
for i in 1 2 3 4 5 6 7 8 9 10; do \
	alive="" ; \
	for pid in $$left; do kill -0 $$pid 2>/dev/null && alive="$$alive $$pid" || true ; done ; \
	[ -n "$$alive" ] || break ; \
	sleep 0.5 ; \
done ; \
for pid in $$alive; do \
	holds_isolated $$pid && kill -9 $$pid 2>/dev/null || true ; \
done ; \
removed= ; \
for i in 1 2 3 4 5 6 7 8 9 10; do \
	rm -rf "$$iso_root" 2>/dev/null ; \
	[ -e "$$iso_root" ] || { removed=1 ; break ; } ; \
	sleep 0.5 ; \
done ; \
[ -n "$$removed" ] || { \
	holders=$$(lsof -n -P -w +D "$$iso_root" 2>/dev/null) ; \
	echo "cannot remove $$iso_root: files kept appearing under it for 5 s" >&2 ; \
	if [ -n "$$holders" ]; then echo "processes holding files under it:" >&2 ; echo "$$holders" >&2 ; \
	else echo "no process holds a file open under it; something writes into it between deletes (Finder's .DS_Store, Spotlight)" >&2 ; fi ; \
	rm -rf "$$iso_root" ; \
	false ; \
}
endef

# The clones and the server of this checkout. Named after the knob rather than
# folded into `clean`, which is about cargo output and must stay usable while an
# isolated test is running.
clean-isolated:
	$(call clean_isolated_at,$(ISOLATED_ROOT))

# The same for what a removed checkout left behind. Two things can survive it:
# the clones, when the checkout went but its directory did not (a `git worktree
# remove` that only got as far as the entry, a prune, a hand-deleted `.git`),
# and the server, when the directory went but the process did not (`git worktree
# remove` takes the directory, not the process). So the environments are
# enumerated three ways, none of which needs a server to be running and none of
# which needs the checkout to still be there:
#
#   - the clones by their place on disk, one and two levels under every
#     directory that holds a checkout of this repository (the main one's parent,
#     and the parent of each worktree git lists), which is where a worktree
#     lives (`../mtld3d-foo`, `../mtld3d-issues/issue-N`) and where its
#     neighbours are left behind;
#   - the clones a checkout recorded in `ISOLATED_REGISTRY` when it made them,
#     which is the only thing that reaches a checkout kept somewhere else, and
#     is read filtered by what is still on disk;
#   - the environments by what a process of one holds open, which is all that is
#     left to name a checkout whose directory went with the worktree. A server
#     maps its own executable out of the SDK clone and the service processes it
#     leaves behind have their cwd in the prefix, so this reaches a session the
#     server no longer holds together, which asking `ps` for a name cannot: a
#     service process answers with a Windows path that names no checkout.
#
# What is still a checkout is then dropped from all three: a directory that
# carries its own `.git` that git can still resolve is in use, and its
# environment is that checkout's own `clean-isolated` to take down, never this
# sweep's. What is left is shut down through its own SDK clone while the clones
# are still there, its clones removed, and its processes signalled by pid, never
# by a pattern that could reach a checkout in use. A process is recognised by a
# path it holds, never by its command line, so an editor whose command line
# merely carries one (or this recipe under a shell) is never taken for a member
# of a session, and a path with a space in it is still one field.
#
# The list of roots is fed to the loop on a descriptor of its own with the
# loop's own input closed, so that nothing a takedown runs can read the roots
# still to come. The record is rewritten at the end to what is still on disk,
# which is how a line for a checkout that is gone leaves it.
clean-isolated-orphans:
	{ { dirname "$$(git rev-parse --path-format=absolute --git-common-dir)" ; \
	    git worktree list --porcelain | sed -n 's|^worktree ||p' ; \
	  } | while IFS= read -r checkout; do dirname "$$checkout" ; done | sort -u \
	  | while IFS= read -r near; do \
		for tree in "$$near"/*/.wine-isolated "$$near"/*/*/.wine-isolated; do \
			[ -d "$$tree" ] && dirname "$$tree" ; \
		done ; \
	  done ; \
	  [ -f '$(ISOLATED_REGISTRY)' ] && while IFS= read -r recorded; do \
		[ -d "$$recorded/.wine-isolated" ] && printf '%s\n' "$$recorded" ; \
	  done < '$(ISOLATED_REGISTRY)' ; \
	  lsof -n -P -w -d cwd,txt -F n 2>/dev/null \
	  | sed -n 's|^n\(/.*\)/\.wine-isolated/.*|\1|p' ; \
	} | sort -u | while IFS= read -r root <&3; do \
		{ [ -e "$$root/.git" ] && git -C "$$root" rev-parse --git-dir >/dev/null 2>&1 ; } && continue ; \
		echo "==> orphan of $$root" ; \
		$(call clean_isolated_at,$$root/.wine-isolated) ; \
	done 3<&0 0</dev/null
	if [ -f '$(ISOLATED_REGISTRY)' ]; then \
		while IFS= read -r recorded; do \
			[ -d "$$recorded/.wine-isolated" ] && printf '%s\n' "$$recorded" ; \
		done < '$(ISOLATED_REGISTRY)' > '$(ISOLATED_REGISTRY).new' ; \
		mv '$(ISOLATED_REGISTRY).new' '$(ISOLATED_REGISTRY)' ; \
	fi

upgrade:
	cd windows && cargo +$(RUST_STABLE) update
	cd unix && cargo +$(RUST_STABLE) update

upgrade-incompat:
	cd windows && cargo +$(RUST_STABLE) upgrade --incompatible && cargo +$(RUST_STABLE) update
	cd unix && cargo +$(RUST_STABLE) upgrade --incompatible && cargo +$(RUST_STABLE) update

# One-time bootstrap for a development machine or a CI runner. Split into leaves
# for the same reason the test and lint targets are: a host-only leg needs
# neither the MSVC SDK nor Rosetta, and a lint leg needs no Wine, so each piece
# stands alone and this is the everything-at-once aggregate.
setup: setup-rust setup-nextest setup-dev setup-xwin setup-rosetta

setup-rust:
	@echo "==> rustup: install $(RUST_STABLE) and $(RUST_NIGHTLY) with the cross-compile targets"
	# `--profile minimal` plus the one component each toolchain is here for:
	# clippy for the lint legs (rustdoc travels with rustc, so `doc` is covered),
	# llvm-tools for the PE linker and archiver below, rustfmt for the fmt legs.
	# Nightly is only ever used for rustfmt; every build, lint and test leg runs
	# on stable.
	rustup toolchain install $(RUST_STABLE) --profile minimal --component clippy --component llvm-tools
	rustup target add --toolchain $(RUST_STABLE) \
		$(PE_i386) $(PE_x64) $(UNIX_TARGET_x64) $(UNIX_TARGET_arm64)
	rustup toolchain install $(RUST_NIGHTLY) --profile minimal --component rustfmt
	# `--locked`: taking every tool's own lockfile is what makes a CI runner and
	# a laptop install the same thing.
	# The toolchain is named rather than left to the exported RUSTUP_TOOLCHAIN
	# because cargo warns about the implicit override here, once per package it
	# builds: the toolchain comes from this environment and not from anything the
	# installed package asks for, which is what we want and worth saying out loud.
	@echo "==> cargo: install/upgrade $(CARGO_TOOLS)"
	cargo +$(RUST_STABLE) install --locked $(CARGO_TOOLS)
	# The PE linker and archiver, out of the toolchain's own llvm-tools rather
	# than a Homebrew LLVM: both LLD and llvm-ar choose their behaviour from the
	# name they are invoked under, so `lld-link` gets LLD's COFF driver and
	# `llvm-lib` gets llvm-ar's lib.exe-compatible mode, which is the syntax
	# cc-rs uses for an MSVC target. windows/.cargo/config.toml names both
	# without a path, so they go in the cargo bin directory, which is already on
	# PATH anywhere cargo works and is cached as one unit with the toolchain.
	@bin=$$(rustc +$(RUST_STABLE) --print sysroot)/lib/rustlib/$(UNIX_NATIVE_TARGET)/bin ; \
	dest=$${CARGO_HOME:-$$HOME/.cargo}/bin ; \
	echo "==> tools: $$dest/{lld-link,llvm-lib} -> $$bin/{rust-lld,llvm-ar}" ; \
	ln -sf $$bin/rust-lld $$dest/lld-link ; \
	ln -sf $$bin/llvm-ar $$dest/llvm-lib

# Tooling only a person uses: cargo-edit backs `upgrade` and `upgrade-incompat`,
# which no CI leg runs, so it stays out of `setup-rust` rather than being rebuilt
# from source on every cold cache for nothing.
setup-dev:
	@echo "==> cargo: install/upgrade cargo-edit"
	cargo +$(RUST_STABLE) install --locked cargo-edit

# Populate the cargo registry for both workspaces without building anything. A
# CI setup job runs this once so the legs that fan out afterwards start from a
# warm cargo home instead of each re-downloading the same crates.
fetch:
	@echo "==> cargo: fetch dependencies for both workspaces"
	cd windows && cargo +$(RUST_STABLE) fetch
	cd unix && cargo +$(RUST_STABLE) fetch

# The MSVC SDK splat lives at /opt/xwin and cannot move: that path is compiled
# into windows/.cargo/config.toml, as `-Lnative` for rustc and `-idirafter` for
# the build-script C/C++. /opt is root-owned on macOS, so creating the directory
# needs sudo. It is its own target because restoring a cached splat into that
# path needs the directory to exist and be writable first.
xwin-dir:
	@if mkdir -p /opt/xwin 2>/dev/null && [ -w /opt/xwin ]; then \
		echo "==> /opt/xwin: already user-writable"; \
	else \
		echo ""; \
		echo "    /opt/xwin will hold the splatted Windows SDK (~3 GB)."; \
		echo "    /opt is root-owned on macOS, so sudo is required to create the directory"; \
		echo "    and chown it to $$USER so 'xwin splat' (and future re-splats) can write."; \
		echo ""; \
		sudo mkdir -p /opt/xwin && sudo chown $$USER /opt/xwin; \
	fi

# Splat the Windows SDK, skipping the work when what is already installed matches
# upstream. A splat made before $(XWIN_STAMP) existed is adopted through the old
# download-cache listing, so an install that was already correct is never
# re-downloaded just to learn its own contents.
setup-xwin: setup-rust xwin-dir
	@echo "==> xwin: compare the pinned manifest to the splat in /opt/xwin"
	@set -e; \
	pkgs='Microsoft\.VC\.[0-9.]+\.CRT|Win11SDK_[0-9.]+'; \
	upstream=$$($(XWIN) list 2>/dev/null | grep -oE "$$pkgs" | sort -u); \
	installed=$$(cat $(XWIN_STAMP) 2>/dev/null || true); \
	if [ -z "$$installed" ]; then \
		installed=$$(ls $(XWIN_CACHE)/dl/ 2>/dev/null | grep -oE "$$pkgs" | sort -u); \
		if [ -n "$$installed" ]; then \
			echo "    no stamp yet, adopting the existing splat from the download cache"; \
		fi; \
	fi; \
	if [ -n "$$installed" ] && [ "$$upstream" = "$$installed" ] && [ -d /opt/xwin/crt ] && [ -d /opt/xwin/sdk ]; then \
		echo "    up to date, skipping splat"; \
		echo "$$installed" | sed 's/^/      /'; \
		echo "$$installed" > $(XWIN_STAMP); \
		exit 0; \
	fi; \
	if [ -z "$$installed" ]; then \
		echo "    nothing installed, first-time download"; \
	elif [ "$$upstream" != "$$installed" ]; then \
		echo "    upgrade available, wiping the download cache and the splat"; \
		echo "      installed: $$(echo $$installed | tr '\n' ' ')"; \
		echo "      upstream:  $$(echo $$upstream | tr '\n' ' ')"; \
		rm -rf $(XWIN_CACHE) /opt/xwin/crt /opt/xwin/sdk $(XWIN_STAMP); \
	else \
		echo "    splat incomplete, re-splatting from the download cache"; \
	fi; \
	$(XWIN) splat --output /opt/xwin; \
	echo "$$upstream" > $(XWIN_STAMP)

# The prebuilt nextest, a universal binary, into the cargo bin directory, which
# is on PATH wherever cargo is. Only the host-native unit tests use it.
setup-nextest:
	@echo "==> nextest: $(NEXTEST_VERSION) prebuilt into $${CARGO_HOME:-$$HOME/.cargo}/bin"
	mkdir -p $${CARGO_HOME:-$$HOME/.cargo}/bin
	curl -LsSf https://get.nexte.st/$(NEXTEST_VERSION)/mac | tar zxf - -C $${CARGO_HOME:-$$HOME/.cargo}/bin
	$${CARGO_HOME:-$$HOME/.cargo}/bin/cargo-nextest nextest --version

# The Wine we run is an x86_64 build and every PE it loads is x86 code, so the
# whole test path goes through Rosetta. A no-op where it is already installed,
# and on an Intel Mac, where the probe just runs natively.
setup-rosetta:
	@if arch -x86_64 /usr/bin/true 2>/dev/null; then \
		echo "==> rosetta: already present"; \
	else \
		echo "==> rosetta: installing (needed to run the x86_64 Wine)"; \
		sudo softwareupdate --install-rosetta --agree-to-license; \
	fi
