ifndef WINE_SDK
$(error WINE_SDK is not set)
endif
export WINE_SDK

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
# for a quick release-profile bundle.
ifneq ($(filter bundle,$(MAKECMDGOALS)),)
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

# The cargo-installed tools a CI leg actually needs: `xwin` splats the MSVC SDK,
# `cargo-nextest` runs every test leg. Floating here and pinned by ci.yml, same
# split as the toolchains, because until this was a variable they were the one
# input a run took from whatever the registry happened to hold that day.
# Developer-only tooling is deliberately not in here, see `setup-dev`.
CARGO_TOOLS ?= xwin cargo-nextest

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
# (aarch64-apple-darwin on Apple Silicon). Builds/runs without Rosetta.
UNIX_NATIVE_TARGET  := $(shell rustc +$(RUST_STABLE) -vV | sed -n 's/^host: //p')


OUT_i386       := windows/target/$(PE_i386)/$(PROFILE)
OUT_x64        := windows/target/$(PE_x64)/$(PROFILE)
OUT_unix_x64   := unix/target/$(UNIX_TARGET_x64)/$(PROFILE)
OUT_unix_arm64 := unix/target/$(UNIX_TARGET_arm64)/$(PROFILE)

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

INSTALL_DIRS := $(WINE_SDK) $(WINE_INSTALL_DIR)

# Both overridable, unlike the rest of these: the HUD and the validation layer
# are here to catch Metal misuse on a real GPU, and a caller running against a
# paravirtual one (a CI runner) has reason to turn them off, since neither has
# anything useful to say about a device that does not implement the counters they
# read.
export MTL_HUD_ENABLED ?= 1
export MTL_DEBUG_LAYER ?= 1
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
	bundle configure-test-prefix \
	test test-unit test-e2e-i686 test-e2e-x86_64 \
	conformance conformance-i686 conformance-x86_64 \
	conformance-baseline conformance-baseline-i686 conformance-baseline-x86_64 \
	conformance-isolate fmt fmt-check clippy clippy-pe-i686 clippy-pe-x86_64 \
	clippy-native audit doc doc-windows doc-unix check clean upgrade \
	upgrade-incompat setup setup-rust setup-dev setup-xwin \
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
	cp $(OUT_unix_x64)/libmtld3d_unix.dylib $(OUT_unix_x64)/mtld3d.so
	rm -rf $(OUT_unix_x64)/mtld3d.so.dSYM
	dsymutil $(OUT_unix_x64)/mtld3d.so

unix-arm64:
	cd unix && cargo +$(RUST_STABLE) build --profile $(PROFILE) --target $(UNIX_TARGET_arm64) $(FRAME_POINTERS)
	cp $(OUT_unix_arm64)/libmtld3d_unix.dylib $(OUT_unix_arm64)/mtld3d.so
	rm -rf $(OUT_unix_arm64)/mtld3d.so.dSYM
	dsymutil $(OUT_unix_arm64)/mtld3d.so

install: install-windows-i686 install-windows-x86_64 install-unix-x64 install-unix-arm64

# Per-arch install leaves, named after the build leaf each one installs: a test
# leg installs the one PE arch it exercises plus the one unix `.so` its Wine
# loads, and only `install` (and `bundle`) covers everything.
#
# The d3d9.dll copies under lib/wine get the builtin signature in place: the
# loader ignores unsigned PEs on the builtin search path. Symbols travel with
# each binary, the `.pdb` beside every PE and the `.dSYM` beside the `.so`, so a
# local crash symbolicates against the installed files with no extra flags.
install-windows-i686: windows-i686
	for dir in $(INSTALL_DIRS); do \
		cp $(OUT_i386)/mtld3d.dll  $(OUT_i386)/mtld3d.pdb  $$dir/lib/wine/i386-windows/ ; \
		cp $(OUT_i386)/d3d9.dll    $(OUT_i386)/d3d9.pdb    $$dir/lib/wine/i386-windows/ ; \
		$(WINEBUILD) --builtin $$dir/lib/wine/i386-windows/d3d9.dll ; \
	done

install-windows-x86_64: windows-x86_64
	for dir in $(INSTALL_DIRS); do \
		cp $(OUT_x64)/mtld3d.dll   $(OUT_x64)/mtld3d.pdb   $$dir/lib/wine/x86_64-windows/ ; \
		cp $(OUT_x64)/d3d9.dll     $(OUT_x64)/d3d9.pdb     $$dir/lib/wine/x86_64-windows/ ; \
		$(WINEBUILD) --builtin $$dir/lib/wine/x86_64-windows/d3d9.dll ; \
	done

# Both unix arches create the directory the Wine tree lacks: a Wine only ever
# loads the one matching its own build, so the other copy is inert, and a tree
# that later gains an arm64 loader is already served.
install-unix-x64: unix-x64
	for dir in $(INSTALL_DIRS); do \
		mkdir -p $$dir/lib/wine/$(UNIX_WINEDIR_x64) ; \
		cp $(OUT_unix_x64)/mtld3d.so        $$dir/lib/wine/$(UNIX_WINEDIR_x64)/ ; \
		rm -rf $$dir/lib/wine/$(UNIX_WINEDIR_x64)/mtld3d.so.dSYM ; \
		cp -R $(OUT_unix_x64)/mtld3d.so.dSYM   $$dir/lib/wine/$(UNIX_WINEDIR_x64)/ ; \
	done

install-unix-arm64: unix-arm64
	for dir in $(INSTALL_DIRS); do \
		mkdir -p $$dir/lib/wine/$(UNIX_WINEDIR_arm64) ; \
		cp $(OUT_unix_arm64)/mtld3d.so      $$dir/lib/wine/$(UNIX_WINEDIR_arm64)/ ; \
		rm -rf $$dir/lib/wine/$(UNIX_WINEDIR_arm64)/mtld3d.so.dSYM ; \
		cp -R $(OUT_unix_arm64)/mtld3d.so.dSYM $$dir/lib/wine/$(UNIX_WINEDIR_arm64)/ ; \
	done

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
	cp $(OUT_i386)/mtld3d.dll           $(BUNDLE_STAGE)/wine/i386-windows/
	cp $(OUT_i386)/d3d9.dll             $(BUNDLE_STAGE)/wine/i386-windows/
	cp $(OUT_x64)/mtld3d.dll            $(BUNDLE_STAGE)/wine/x86_64-windows/
	cp $(OUT_x64)/d3d9.dll              $(BUNDLE_STAGE)/wine/x86_64-windows/
	# Markers live outside wine/, and already carry the name they need in the
	# prefix, so both routes are a plain copy into the matching system dir with
	# no rename. Keeping them out of wine/ is what stops `cp -R wine/*` from
	# dragging them onto the builtin search path, where wineboot would stamp a
	# second, useless marker under the name "mtld3d.fake.dll".
	cp $(OUT_i386)/mtld3d.fake.dll      $(BUNDLE_STAGE)/prefix-markers/syswow64/mtld3d.dll
	cp $(OUT_x64)/mtld3d.fake.dll       $(BUNDLE_STAGE)/prefix-markers/system32/mtld3d.dll
	$(WINEBUILD) --builtin $(BUNDLE_STAGE)/wine/i386-windows/d3d9.dll
	$(WINEBUILD) --builtin $(BUNDLE_STAGE)/wine/x86_64-windows/d3d9.dll
	cp $(OUT_unix_x64)/mtld3d.so        $(BUNDLE_STAGE)/wine/$(UNIX_WINEDIR_x64)/
	cp $(OUT_unix_arm64)/mtld3d.so      $(BUNDLE_STAGE)/wine/$(UNIX_WINEDIR_arm64)/
	cp $(OUT_i386)/d3d9.dll             $(BUNDLE_STAGE)/native/i386-windows/
	cp $(OUT_x64)/d3d9.dll              $(BUNDLE_STAGE)/native/x86_64-windows/
	cp $(CURDIR)/mtld3d.conf            $(BUNDLE_STAGE)/
	cp $(CURDIR)/INSTALL.md             $(BUNDLE_STAGE)/
	cp $(CURDIR)/LICENSE                $(BUNDLE_STAGE)/
	tar -cJf $(BUNDLE_OUT) -C $(BUNDLE_STAGE) wine native prefix-markers mtld3d.conf INSTALL.md LICENSE
	# The symbols for exactly these binaries, as a second archive. Laid out by
	# arch alone, with no wine/native split: debug info has no install route, and
	# the two d3d9.dll flavors are one binary with one `.pdb`.
	mkdir -p $(DEBUG_STAGE)/i386-windows
	mkdir -p $(DEBUG_STAGE)/x86_64-windows
	mkdir -p $(DEBUG_STAGE)/$(UNIX_WINEDIR_x64)
	mkdir -p $(DEBUG_STAGE)/$(UNIX_WINEDIR_arm64)
	echo $(BUILD_ID)                    > $(DEBUG_STAGE)/BUILD
	cp $(OUT_i386)/d3d9.pdb             $(DEBUG_STAGE)/i386-windows/
	cp $(OUT_i386)/mtld3d.pdb           $(DEBUG_STAGE)/i386-windows/
	cp $(OUT_x64)/d3d9.pdb              $(DEBUG_STAGE)/x86_64-windows/
	cp $(OUT_x64)/mtld3d.pdb            $(DEBUG_STAGE)/x86_64-windows/
	cp -R $(OUT_unix_x64)/mtld3d.so.dSYM   $(DEBUG_STAGE)/$(UNIX_WINEDIR_x64)/
	cp -R $(OUT_unix_arm64)/mtld3d.so.dSYM $(DEBUG_STAGE)/$(UNIX_WINEDIR_arm64)/
	tar -cJf $(DEBUG_OUT) -C $(DEBUG_STAGE) BUILD i386-windows x86_64-windows \
		$(UNIX_WINEDIR_x64) $(UNIX_WINEDIR_arm64)

# E2E test environment overrides (the global exports above target the game):
#   - shaderCache.enable=false  — parallel test processes mustn't race the cache.
#   - color.hdr.enable=false    the shipped default is on, and it resolves off
#     the running machine's panel, so leaving it would make the suite take the
#     HDR present route on an EDR Mac and the SDR one elsewhere. Pin it so the
#     results mean the same thing everywhere; the HDR route is exercised by real
#     runs and by the present-pipeline tests, not by the e2e assertions.
#   - WINEDEBUG= (empty)        — silence the +msync debug channel's per-call spam.
# MTL_DEBUG_LAYER stays on (inherited) so Metal API misuse fails the tests.
#
# SCALE=<n> additionally reruns the whole e2e suite at `render.scale = <n>`,
# i.e. rasterizing the back buffer smaller than the resolution D3D9 reports and
# letting MetalFX resolve it. Every coordinate the suite asserts on is in the
# reported space, so a passing scaled run is the evidence that the logical and
# render spaces stayed separate. `make test SCALE=0.75` — try 0.5 and a
# non-dividing 0.67 too, since those catch rounding that a clean fraction hides.
MTLD3D_CONF_TEST := shaderCache.enable=false;color.hdr.enable=false$(if $(SCALE),;render.scale=$(SCALE))
# Quoted: the config separator is `;`, which the shell would otherwise read as
# a command separator and run the rest of the line as its own command.
MTLD3D_TEST_ENV := MTLD3D_CONFIG='$(MTLD3D_CONF_TEST)' WINEDEBUG=

# Every leg that needs a prefix depends on `install-windows-*` first, so
# `mtld3d.dll` is already in lib/wine when wineboot creates the prefix and
# wine.inf's `11,,*` wildcard stamps its marker along with every other builtin.
# A prefix that predates the install does not get one and cannot load mtld3d;
# `wineboot -u`, or deleting it and re-running, fixes that.
configure-test-prefix:
	# Keep automated tests non-interactive and independent of mutable prefix
	# display settings. EmulateModeset prevents physical host mode changes;
	# RetinaMode keeps Win32 monitor geometry in the same physical-pixel space
	# as mtld3d's adapter modes. The first `reg add` also creates the prefix if
	# it does not exist yet, which is the case on a fresh machine.
	$(WINE) reg add 'HKCU\Software\Wine\WineDbg' /v ShowCrashDialog /t REG_DWORD /d 0 /f >/dev/null 2>&1
	$(WINE) reg add 'HKCU\Software\Wine\X11 Driver' /v EmulateModeset /t REG_SZ /d Y /f >/dev/null 2>&1
	$(WINE) reg add 'HKCU\Software\Wine\Mac Driver' /v RetinaMode /t REG_SZ /d Y /f >/dev/null 2>&1
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
# A scaled run reports the whole suite instead of stopping at the first
# failure. The point of `SCALE` is to survey which assertions still hold in
# the reported space, and one scale-dependent test would otherwise hide every
# later test's scaled behaviour. The default run keeps fail-fast: there the
# first failure is a regression to fix, not a survey to read.
E2E_NEXTEST_FLAGS := $(if $(SCALE),--no-fail-fast)

test-e2e-i686: install-windows-i686 install-unix-$(SDK_UNIX_ARCH)
	$(MAKE) configure-test-prefix
	cd windows && $(MTLD3D_TEST_ENV) CARGO_TARGET_I686_PC_WINDOWS_MSVC_RUNNER=$(WINE) \
		cargo +$(RUST_STABLE) nextest run -p mtld3d-tests --target $(PE_i386) $(E2E_NEXTEST_FLAGS)

test-e2e-x86_64: install-windows-x86_64 install-unix-$(SDK_UNIX_ARCH)
	$(MAKE) configure-test-prefix
	cd windows && $(MTLD3D_TEST_ENV) CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER=$(WINE) \
		cargo +$(RUST_STABLE) nextest run -p mtld3d-tests --target $(PE_x64) $(E2E_NEXTEST_FLAGS)

# d3d9 conformance (NOT part of `make test`): run Wine's upstream d3d9 test exe
# against our installed builtin d3d9.dll, then diff per-site failure counts
# against the checked-in baseline. Many subtests fail by design, see
# unix/conformance/CONFORMANCE.md. The test exes ship inside the Wine SDK
# ($(D3D9_TEST_*) above); the runner takes them as paths and finds its
# baseline.txt in the crate dir.
#
# One arch per runner process, so the two gates are independent jobs. Every leg
# runs the same four subtests for its arch.
CONFORMANCE_RUN = cd unix && cargo +$(RUST_STABLE) run --profile $(PROFILE) -p mtld3d-conformance -- \
	--wine $(WINE_SDK)/bin/wine

# $(1) = arch (i686|x86_64), $(2) = extra runner args. Checks the exe up front
# so a bundle that predates the published test binaries says so, rather than
# failing four times inside the runner.
define conformance_leg
	$(MAKE) configure-test-prefix
	test -f $(D3D9_TEST_$(1)) || { echo "$(D3D9_TEST_$(1)) is missing: re-bundle the Wine SDK, this one predates the published d3d9 test binaries" >&2; exit 2; }
	$(CONFORMANCE_RUN) --arch $(1) --exe $(D3D9_TEST_$(1)) $(2)
endef

conformance: conformance-i686 conformance-x86_64

conformance-i686: install-windows-i686 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,i686)

conformance-x86_64: install-windows-x86_64 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,x86_64)

# Re-record the baseline. Both legs write the same baseline.txt (each replacing
# only its own arch's entries), so they must run in sequence: hence recursive
# make in the recipe rather than prerequisites, which `-j` could interleave.
conformance-baseline:
	$(MAKE) conformance-baseline-i686
	$(MAKE) conformance-baseline-x86_64

conformance-baseline-i686: install-windows-i686 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,i686,--update-baseline)

conformance-baseline-x86_64: install-windows-x86_64 install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,x86_64,--update-baseline)

# Flap characterization: run ONE subtest REPEAT times and print a per-site flap
# report (which sites fire deterministically vs flutter run-to-run), the
# evidence for tagging a site `flaky` in CONFORMANCE.md. Tune with ONLY (device|
# visual|stateblock|d3d9ex), ARCH (i686|x86_64), REPEAT (default 20).
ONLY ?= device
ARCH ?= i686
REPEAT ?= 20
conformance-isolate: install-windows-$(ARCH) install-unix-$(SDK_UNIX_ARCH)
	$(call conformance_leg,$(ARCH),--only $(ONLY) --repeat $(REPEAT))

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
# conventions audit, and the doc build. fmt-check first (fast, fails early on
# drift); clippy reuses the target above; audit is pure grep; doc last. Each leg
# is also its own target, so CI runs them as parallel jobs instead of this
# sequence.
check:
	$(MAKE) fmt-check
	$(MAKE) clippy
	$(MAKE) audit
	$(MAKE) doc

clean:
	cd windows && cargo +$(RUST_STABLE) clean
	cd unix && cargo +$(RUST_STABLE) clean

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
setup: setup-rust setup-dev setup-xwin setup-rosetta

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
	# `--locked`: cargo-nextest refuses to build any other way (it ships a
	# tripwire crate that fails the compile), and taking every tool's own
	# lockfile is what makes a CI runner and a laptop install the same thing.
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
