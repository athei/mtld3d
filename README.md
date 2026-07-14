# mtld3d

Direct3D 9 translation layer for Wine on macOS, backed by Metal.

mtld3d replaces Wine's built-in `d3d9.dll` with an implementation that translates D3D9 calls through Wine's PE/Unix boundary into Metal command buffers on the host. The pure-Rust core (`mtld3d-core`) handles DXSO → MSL shader translation, render-pass scheduling, and fixed-function state.

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

For threading, the PE/Unix boundary contract, perf instrumentation, and the shader/heap debugging toolkits, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

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
