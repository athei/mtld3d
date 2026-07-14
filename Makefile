ifndef WINE_SDK
$(error WINE_SDK is not set)
endif
export WINE_SDK

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

PE_i386     := i686-pc-windows-msvc
PE_x64      := x86_64-pc-windows-msvc
# Release/Wine target: the unix `.so` must be x86_64 Mach-O (Wine's unix-call
# boundary), so shipped artifacts are always built for x86_64.
UNIX_RELEASE_TARGET := x86_64-apple-darwin
# Native host target for unit tests + clippy — whatever this machine is
# (aarch64-apple-darwin on Apple Silicon). Builds/runs without Rosetta.
UNIX_NATIVE_TARGET  := $(shell rustc -vV | sed -n 's/^host: //p')

OUT_i386 := windows/target/$(PE_i386)/$(PROFILE)
OUT_x64  := windows/target/$(PE_x64)/$(PROFILE)
OUT_unix := unix/target/$(UNIX_RELEASE_TARGET)/$(PROFILE)

XWIN_CACHE := $(HOME)/Library/Caches/xwin

INSTALL_DIRS := $(WINE_SDK) $(WINE_INSTALL_DIR)

export MTL_HUD_ENABLED = 1
export MTL_DEBUG_LAYER = 1
export WINEDBG_DISABLE_CRASH_DIALOG = 1
export WINEDEBUG=+msync
export WINEMSYNC=1

MAKEFLAGS += --silent

BUNDLE_NAME  := mtld3d.tar.xz
BUNDLE_OUT   := $(CURDIR)/windows/target/$(BUNDLE_NAME)
BUNDLE_STAGE := $(CURDIR)/windows/target/bundle

.PHONY: all windows unix install bundle test conformance conformance-baseline conformance-isolate fmt clippy audit doc check clean setup upgrade upgrade-incompat

all: windows unix

windows:
	cd windows && cargo build --profile $(PROFILE) --target $(PE_i386)
	cd windows && cargo build --profile $(PROFILE) --target $(PE_x64)
	# mtld3d.dll is only ever a Wine builtin (it owns the unix-call globals), so
	# it gets the builtin signature at build time. d3d9.dll stays an ordinary
	# native PE here — `install` and `bundle` mark their staged copies instead,
	# so the build output can also be loaded as a native override in Wine
	# distributions we don't control (CrossOver).
	winebuild --builtin $(OUT_i386)/mtld3d.dll
	winebuild --builtin $(OUT_x64)/mtld3d.dll
	# Tiny "fake DLL" placeholders for the mtld3d builtin name, one per arch.
	# Shipped into lib/wine; a prefix-setup step must copy them into the target
	# prefix's syswow64/system32, since Wine finds builtins by name in the
	# prefix, not lib/wine (`d3d9` gets its placeholder from wineboot, but
	# custom names don't).
	winebuild --fake-module -o $(OUT_i386)/mtld3d.fake.dll -m32 --dll $(OUT_i386)/mtld3d.dll
	winebuild --fake-module -o $(OUT_x64)/mtld3d.fake.dll -m64 --dll $(OUT_x64)/mtld3d.dll

unix:
	cd unix && cargo build --profile $(PROFILE) --target $(UNIX_RELEASE_TARGET)

install: all
	# The d3d9.dll copies under lib/wine get the builtin signature in place —
	# the loader ignores unsigned PEs on the builtin search path.
	for dir in $(INSTALL_DIRS); do \
		cp $(OUT_i386)/mtld3d.dll  $(OUT_i386)/mtld3d.pdb  $$dir/lib/wine/i386-windows/ ; \
		cp $(OUT_i386)/d3d9.dll    $(OUT_i386)/d3d9.pdb    $$dir/lib/wine/i386-windows/ ; \
		winebuild --builtin $$dir/lib/wine/i386-windows/d3d9.dll ; \
		cp $(OUT_x64)/mtld3d.dll   $(OUT_x64)/mtld3d.pdb   $$dir/lib/wine/x86_64-windows/ ; \
		cp $(OUT_x64)/d3d9.dll     $(OUT_x64)/d3d9.pdb     $$dir/lib/wine/x86_64-windows/ ; \
		winebuild --builtin $$dir/lib/wine/x86_64-windows/d3d9.dll ; \
		cp $(OUT_unix)/libmtld3d_unix.dylib $$dir/lib/wine/x86_64-unix/mtld3d.so ; \
		cp $(OUT_i386)/mtld3d.fake.dll      $$dir/lib/wine/i386-windows/ ; \
		cp $(OUT_x64)/mtld3d.fake.dll       $$dir/lib/wine/x86_64-windows/ ; \
	done

# Distribution bundle, serving both install routes (see INSTALL.md, which is
# shipped inside): wine/ mirrors a Wine installation's lib/wine/ with every
# PE builtin-marked (drop-in for a Wine tree the user owns), while native/
# holds the unmarked d3d9.dll for the DLL-override route (required on
# CrossOver). The fake placeholders are the prefix markers for the custom
# mtld3d builtin name.
bundle: all
	rm -rf $(BUNDLE_STAGE) $(BUNDLE_OUT)
	mkdir -p $(BUNDLE_STAGE)/wine/i386-windows
	mkdir -p $(BUNDLE_STAGE)/wine/x86_64-windows
	mkdir -p $(BUNDLE_STAGE)/wine/x86_64-unix
	mkdir -p $(BUNDLE_STAGE)/native/i386-windows
	mkdir -p $(BUNDLE_STAGE)/native/x86_64-windows
	cp $(OUT_i386)/mtld3d.dll           $(BUNDLE_STAGE)/wine/i386-windows/
	cp $(OUT_i386)/mtld3d.fake.dll      $(BUNDLE_STAGE)/wine/i386-windows/
	cp $(OUT_i386)/d3d9.dll             $(BUNDLE_STAGE)/wine/i386-windows/
	cp $(OUT_x64)/mtld3d.dll            $(BUNDLE_STAGE)/wine/x86_64-windows/
	cp $(OUT_x64)/mtld3d.fake.dll       $(BUNDLE_STAGE)/wine/x86_64-windows/
	cp $(OUT_x64)/d3d9.dll              $(BUNDLE_STAGE)/wine/x86_64-windows/
	winebuild --builtin $(BUNDLE_STAGE)/wine/i386-windows/d3d9.dll
	winebuild --builtin $(BUNDLE_STAGE)/wine/x86_64-windows/d3d9.dll
	cp $(OUT_unix)/libmtld3d_unix.dylib $(BUNDLE_STAGE)/wine/x86_64-unix/mtld3d.so
	cp $(OUT_i386)/d3d9.dll             $(BUNDLE_STAGE)/native/i386-windows/
	cp $(OUT_x64)/d3d9.dll              $(BUNDLE_STAGE)/native/x86_64-windows/
	cp $(CURDIR)/mtld3d.conf            $(BUNDLE_STAGE)/
	cp $(CURDIR)/INSTALL.md             $(BUNDLE_STAGE)/
	cp $(CURDIR)/LICENSE                $(BUNDLE_STAGE)/
	tar -cJf $(BUNDLE_OUT) -C $(BUNDLE_STAGE) wine native mtld3d.conf INSTALL.md LICENSE

# E2E test environment overrides (the global exports above target the game):
#   - shaderCache.enable=false  — parallel test processes mustn't race the cache.
#   - WINEDEBUG= (empty)        — silence the +msync debug channel's per-call spam.
# MTL_DEBUG_LAYER stays on (inherited) so Metal API misuse fails the tests.
MTLD3D_TEST_ENV := MTLD3D_CONFIG=shaderCache.enable=false WINEDEBUG=

test: install
	# Host-native unit tests, built for this machine's native arch (no Rosetta).
	# The windows workspace singles out mtld3d-core (its other members are
	# PE-only and can't build for the host target) and must override its i686
	# default; the unix workspace already defaults to the host, so just run all
	# of it.
	cd windows && cargo nextest run -p mtld3d-core -p mtld3d-types --target $(UNIX_NATIVE_TARGET)
	cd unix && cargo nextest run
	# Pre-boot a persistent wineserver so individual e2e test processes attach
	# to it instead of each paying boot cost (and briefly holding its stdio).
	# Both lines detach stdio: the persistent server (and the winedevice.exe
	# residents wineboot leaves behind) would otherwise inherit make's
	# stdout/stderr and hold a consumer pipe open forever — `make test | ...`
	# then never sees EOF even though make itself exited.
	-wineserver -p >/dev/null 2>&1
	-wine wineboot >/dev/null 2>&1
	cd windows && $(MTLD3D_TEST_ENV) CARGO_TARGET_I686_PC_WINDOWS_MSVC_RUNNER=wine \
		cargo nextest run -p mtld3d-tests --target $(PE_i386)
	cd windows && $(MTLD3D_TEST_ENV) CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER=wine \
		cargo nextest run -p mtld3d-tests --target $(PE_x64)

# d3d9 conformance (NOT part of `make test`): run Wine's upstream d3d9 test exe
# from WINE_BUILD against our installed builtin d3d9.dll, then diff per-site
# failure counts against the checked-in baseline. Many subtests fail by design
# — see unix/conformance/CONFORMANCE.md. WINE_BUILD must point at a Wine build
# tree that has built dlls/d3d9/tests (it ships d3d9_test.exe per arch); the
# runner finds the wine loader via the global WINE_SDK and its baseline.txt in
# the crate dir.
conformance: install
	test -n "$(WINE_BUILD)" || { echo "WINE_BUILD is not set — point it at a Wine build tree with dlls/d3d9/tests built" >&2; exit 2; }
	cd unix && cargo run --profile $(PROFILE) -p mtld3d-conformance -- --wine-build $(WINE_BUILD)

conformance-baseline: install
	test -n "$(WINE_BUILD)" || { echo "WINE_BUILD is not set — point it at a Wine build tree with dlls/d3d9/tests built" >&2; exit 2; }
	cd unix && cargo run --profile $(PROFILE) -p mtld3d-conformance -- --update-baseline --wine-build $(WINE_BUILD)

# Flap characterization: run ONE subtest/arch REPEAT times and print a per-site
# flap report (which sites fire deterministically vs flutter run-to-run) — the
# evidence for tagging a site `flaky` in baseline.txt. Tune with ONLY (device|
# visual|stateblock|d3d9ex), ARCH (i686|x86_64), REPEAT (default 20).
ONLY ?= device
ARCH ?= i686
REPEAT ?= 20
conformance-isolate: install
	test -n "$(WINE_BUILD)" || { echo "WINE_BUILD is not set — point it at a Wine build tree with dlls/d3d9/tests built" >&2; exit 2; }
	cd unix && cargo run --profile $(PROFILE) -p mtld3d-conformance -- --wine-build $(WINE_BUILD) --only $(ONLY) --arch $(ARCH) --repeat $(REPEAT)

fmt:
	cd windows && cargo +nightly fmt
	cd unix && cargo +nightly fmt

clippy:
	# No --all-targets on the whole-workspace PE runs: that would build every
	# member's test targets for PE, including mtld3d-core's apple-only objc2
	# dev-deps (the SM3 corpus test), which hard `compile_error!` off Apple.
	# Lib/bin only here; test targets are linted per-crate below.
	cd windows && cargo clippy --target $(PE_i386) -- -D warnings
	cd windows && cargo clippy --target $(PE_x64)  -- -D warnings
	cd windows && cargo clippy -p mtld3d-core --target $(UNIX_NATIVE_TARGET) --all-targets -- -D warnings
	# mtld3d-tests' integration tests aren't covered by the whole-workspace
	# lib/bin runs; lint all its targets on both PE archs (no apple dev-deps).
	cd windows && cargo clippy -p mtld3d-tests --target $(PE_i386) --all-targets -- -D warnings
	cd windows && cargo clippy -p mtld3d-tests --target $(PE_x64)  --all-targets  -- -D warnings
	cd unix && cargo clippy --all-targets -- -D warnings

# The conventions clippy can't express: doc-comment shape, the Clone/Copy derive
# inventory, and the handful of patterns that are banned or confined to a known
# set of files. See docs/CONVENTIONS.md § Mechanical audit.
audit:
	./scripts/audit.sh

# rustdoc's own lints, which no other target sees: broken and private intra-doc
# links, malformed HTML in doc comments. `audit` gates the *shape* of a doc block
# and clippy gates its prose; only rustdoc knows whether its links resolve.
#
# The windows workspace is documented for a PE target, not the host: d3d9 and the
# shim are `cdylib`s with raw-dylib imports and only build for *-pc-windows-msvc,
# so a host run would silently skip them. i686 covers every member.
doc:
	cd windows && RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --target $(PE_i386)
	cd unix && RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

# One command to run before every commit: formatting, the full clippy sweep, the
# conventions audit, and the doc build. fmt-check first (fast, fails early on
# drift); clippy reuses the target above; audit is pure grep; doc last.
check:
	cd windows && cargo +nightly fmt --check
	cd unix && cargo +nightly fmt --check
	$(MAKE) clippy
	$(MAKE) audit
	$(MAKE) doc

clean:
	cd windows && cargo clean
	cd unix && cargo clean

upgrade:
	cd windows && cargo update
	cd unix && cargo update

upgrade-incompat:
	cd windows && cargo upgrade --incompatible && cargo update
	cd unix && cargo upgrade --incompatible && cargo update

setup:
	@echo "==> brew: install/upgrade llvm and lld"
	brew install llvm lld
	brew upgrade llvm lld
	@echo "==> rustup: add cross-compile targets"
	rustup target add i686-pc-windows-msvc x86_64-pc-windows-msvc x86_64-apple-darwin
	@echo "==> cargo: install/upgrade xwin and cargo-edit"
	cargo install xwin cargo-edit
	@echo "==> /opt/xwin: ensure user-writable"
	@if mkdir -p /opt/xwin 2>/dev/null && [ -w /opt/xwin ]; then \
		echo "    /opt/xwin already user-writable"; \
	else \
		echo ""; \
		echo "    /opt/xwin will hold the splatted Windows SDK (~3 GB)."; \
		echo "    /opt is root-owned on macOS, so sudo is required to create the directory"; \
		echo "    and chown it to $$USER so 'xwin splat' (and future re-splats) can write."; \
		echo ""; \
		sudo mkdir -p /opt/xwin && sudo chown $$USER /opt/xwin; \
	fi
	@echo "==> xwin: compare upstream manifest to local cache ($(XWIN_CACHE))"
	@upstream=$$(xwin --accept-license --arch x86,x86_64 --cache-dir $(XWIN_CACHE) list 2>/dev/null | grep -oE 'Microsoft\.VC\.[0-9.]+\.CRT|Win11SDK_[0-9.]+' | sort -u); \
	cached=$$(ls $(XWIN_CACHE)/dl/ 2>/dev/null | grep -oE 'Microsoft\.VC\.[0-9.]+\.CRT|Win11SDK_[0-9.]+' | sort -u); \
	if [ -n "$$cached" ] && [ "$$upstream" = "$$cached" ] && [ -d /opt/xwin/crt ] && [ -d /opt/xwin/sdk ]; then \
		echo "    up to date — skipping splat"; \
		echo "$$cached" | sed 's/^/      /'; \
	else \
		if [ -n "$$cached" ] && [ "$$upstream" != "$$cached" ]; then \
			echo "    upgrade available — wiping cache and /opt/xwin"; \
			echo "      cached:   $$(echo $$cached | tr '\n' ' ')"; \
			echo "      upstream: $$(echo $$upstream | tr '\n' ' ')"; \
			rm -rf $(XWIN_CACHE) /opt/xwin/crt /opt/xwin/sdk; \
		elif [ -z "$$cached" ]; then \
			echo "    no local cache — first-time download"; \
		else \
			echo "    /opt/xwin missing or stale — re-splat from cache"; \
		fi; \
		xwin --accept-license --arch x86,x86_64 --cache-dir $(XWIN_CACHE) splat --output /opt/xwin; \
	fi
