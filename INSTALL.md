# Installing mtld3d

This is the installation guide for the release bundle (`mtld3d.tar.xz`). It
covers stock Wine installations and CrossOver bottles. Building from source
and the developer `make install` flow are covered in the source repository's
`README.md`.

## Bundle contents

```
wine/                       lib/wine-shaped tree, every PE builtin-marked
  i386-windows/
    d3d9.dll                Direct3D 9 implementation (builtin-marked)
    mtld3d.dll              PE half of the unix-call bridge (builtin-marked)
    mtld3d.fake.dll         prefix marker for the custom builtin name
  x86_64-windows/           the same three files, 64-bit
  x86_64-unix/
    mtld3d.so               Metal-side unix library
native/                     unmarked d3d9.dll for the DLL-override route
  i386-windows/d3d9.dll
  x86_64-windows/d3d9.dll
mtld3d.conf                 sample configuration, self-documenting
INSTALL.md                  this file
LICENSE
```

The two `d3d9.dll` variants are the same binary in two loader flavors:

- `wine/…/d3d9.dll` carries Wine's *builtin* signature. Dropped into a Wine
  installation's `lib/wine/` directories it **replaces the stock d3d9
  builtin** for that whole Wine tree. The signature is also why it cannot be
  used as an override: Wine never executes a builtin-marked PE found outside
  its builtin search path.
- `native/…/d3d9.dll` is an ordinary native PE, loaded through a
  `d3d9=native` DLL override. The Wine installation stays untouched.

Common to both routes: `mtld3d.dll` + `mtld3d.so` are a custom-named Wine
builtin pair — the PE half can only reach its unix half when loaded as a
builtin, so there is no native variant of it. And because Wine resolves
builtin *names* through the prefix's system directories (wineboot stamps
placeholders there for the names it knows — `mtld3d` is not one of them),
every prefix needs the `mtld3d.fake.dll` markers copied in once.

## Requirements

- **Apple Silicon macOS 15** or newer.
- **Rosetta 2** (`softwareupdate --install-rosetta`) — the unix library is
  x86_64 Mach-O, matching Wine's own binaries.
- A Wine with the current WoW64 loader: **Wine 8.0 or newer**, or
  **CrossOver 24 or newer**.
- A **64-bit prefix / bottle** — 32-bit games run in it through WoW64.

## x87 performance

D3D9-era games do their floating-point math in x87 instructions, which
Rosetta 2 translates slowly. For full performance, run the game together with
[x87sidecar](https://github.com/athei/x87sidecar), a JIT that replaces
Rosetta's x87 handling. Its cooperative attach mode requires a Wine that
performs the sidecar handshake at startup: the Wine builds from
[wine-build](https://github.com/athei/wine-build) carry that patch, which
lets the x87sidecar binary work without any entitlements.

## Choosing a route

**Builtin** — for a Wine installation you own (your own build, an app-bundled
Wine, a package install). Simplest to operate: no registry override, no
per-game files, and every d3d9 application in that Wine tree uses mtld3d.
The costs: it modifies the installation, so a Wine update or reinstall wipes
it (re-copy afterwards), and other d3d9 applications can no longer reach the
stock implementation. Not possible on CrossOver — `CrossOver.app` is replaced
wholesale on every update.

**Native override** — required on CrossOver, and the right choice on stock
Wine when the stock d3d9 should keep serving other applications. The costs: a
registry override per prefix plus a `d3d9.dll` copy per game (or per prefix).

## Stock Wine, builtin route

`$WINE` is the installation root — the directory containing `lib/wine/`.

```sh
tar -xf mtld3d.tar.xz

# Replaces lib/wine's d3d9.dll and adds the mtld3d builtin pair.
cp -R wine/* "$WINE/lib/wine/"

# One-time prefix markers for the custom builtin name (d3d9 needs none —
# wineboot stamps its placeholder into every prefix).
cp wine/i386-windows/mtld3d.fake.dll   "$WINEPREFIX/drive_c/windows/syswow64/mtld3d.dll"
cp wine/x86_64-windows/mtld3d.fake.dll "$WINEPREFIX/drive_c/windows/system32/mtld3d.dll"

# Optional: runtime configuration next to the game executable.
cp mtld3d.conf "/path/to/MyGame/"
```

Re-run the `lib/wine` copy after any Wine update or reinstall.

## Stock Wine, native-override route

```sh
tar -xf mtld3d.tar.xz

# Only the mtld3d builtin pair goes into the installation — the stock
# d3d9.dll stays in place.
cp wine/i386-windows/mtld3d.dll   "$WINE/lib/wine/i386-windows/"
cp wine/x86_64-windows/mtld3d.dll "$WINE/lib/wine/x86_64-windows/"
cp wine/x86_64-unix/mtld3d.so     "$WINE/lib/wine/x86_64-unix/"

# One-time prefix markers, as in the builtin route.
cp wine/i386-windows/mtld3d.fake.dll   "$WINEPREFIX/drive_c/windows/syswow64/mtld3d.dll"
cp wine/x86_64-windows/mtld3d.fake.dll "$WINEPREFIX/drive_c/windows/system32/mtld3d.dll"

# Native d3d9.dll next to the game executable — pick the game's arch.
cp native/i386-windows/d3d9.dll "/path/to/MyGame/"     # 32-bit game
#cp native/x86_64-windows/d3d9.dll "/path/to/MyGame/"  # 64-bit game

# Optional: runtime configuration next to the game executable.
cp mtld3d.conf "/path/to/MyGame/"

# DLL override so Wine loads the native d3d9.dll instead of its builtin.
wine reg add 'HKCU\Software\Wine\DllOverrides' /v d3d9 /d native /f
```

Instead of the game directory, the native `d3d9.dll` can go into the prefix —
`drive_c/windows/syswow64/` for 32-bit games, `drive_c/windows/system32/` for
64-bit — where it covers every application in the prefix. A game-directory
copy wins when both exist. The prefix copy survives wineboot prefix updates,
which only replace placeholder files, never real PEs.

## CrossOver

The setup is self-contained in the bottle and survives CrossOver updates:
`d3d9.dll` is loaded as a *native* DLL from the game directory via a DLL
override, and the `mtld3d.dll` / `mtld3d.so` builtin pair is supplied through
the bottle's DLL search path. Nothing is written into `CrossOver.app`.

```sh
tar -xf mtld3d.tar.xz

BOTTLE="$HOME/Library/Application Support/CrossOver/Bottles/MyBottle"
GAME_DIR="$BOTTLE/drive_c/Program Files/MyGame"   # dir holding the game .exe

# The builtins (both PE arches + the unix side), kept inside the bottle.
# (The builtin-marked d3d9.dll comes along but is inert here — a marked PE
# on the search path can never shadow CrossOver's own builtin.)
cp -R wine "$BOTTLE/mtld3d"

# Prefix markers so Wine resolves the custom builtin name (Wine looks
# builtin names up in the prefix's system dirs, not on the search path).
cp "$BOTTLE/mtld3d/i386-windows/mtld3d.fake.dll"   "$BOTTLE/drive_c/windows/syswow64/mtld3d.dll"
cp "$BOTTLE/mtld3d/x86_64-windows/mtld3d.fake.dll" "$BOTTLE/drive_c/windows/system32/mtld3d.dll"

# Native d3d9.dll next to the game executable — pick the game's arch.
cp native/i386-windows/d3d9.dll "$GAME_DIR/"     # 32-bit game
#cp native/x86_64-windows/d3d9.dll "$GAME_DIR/"  # 64-bit game

# Optional: runtime configuration next to the game executable.
cp mtld3d.conf "$GAME_DIR/"

# DLL override so Wine loads the native d3d9.dll instead of its builtin.
CX_WINE="/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin/wine"
"$CX_WINE" --bottle MyBottle reg add 'HKCU\Software\Wine\DllOverrides' /v d3d9 /d native /f
```

The native `d3d9.dll` can alternatively go into the bottle's
`drive_c/windows/syswow64/` (32-bit) or `system32/` (64-bit) to cover every
application in the bottle, as described in the stock-Wine section above.

Finally, add the `mtld3d` directory to the bottle's DLL search path in
`$BOTTLE/cxbottle.conf` (create the `[Wine]` section if it doesn't exist):

```ini
[Wine]
"DllPath" = "${CX_ROOT}/lib/wine/x86_64-windows:${CX_ROOT}/lib/wine/i386-windows:${CX_ROOT}/lib/wine:${WINEPREFIX}/mtld3d"
```

Two CrossOver quirks make this exact form necessary. First, a `DllPath`
value replaces the launcher-computed `WINEDLLPATH` wholesale. Wine itself
would not care — ntdll always searches the directory it was loaded from
before any `WINEDLLPATH` entry — but CrossOver's launcher script does its
own lookups over this value (`winewrapper.exe`, which it prepends to every
launch, is located through it), so CrossOver's own directories must stay
listed. Second, a `WINEDLLPATH` entry under `[EnvironmentVariables]` would
be overwritten by the launcher, so it must be the `[Wine]` key.

Notes:

- **Graphics setting**: leave the bottle's graphics backend on its default.
  The registry override makes Wine load the native `d3d9.dll` regardless of
  the selected backend — that switch only redirects Wine's *builtin* DLL
  search, which the native override bypasses. Other Direct3D versions keep
  following the bottle's backend selection.
- **Logging / env**: per-bottle `RUST_LOG` or `MTLD3D_CONFIG` go under
  `[EnvironmentVariables]` in the same `cxbottle.conf`.
- **CrossOver updates** replace `CrossOver.app` but not bottles, so this
  setup persists. If a version upgrade migrates the bottle and refreshes
  `drive_c/windows`, re-copy the two prefix markers.

## Configuration and logging

`mtld3d.conf` is read from the directory of the running `.exe`; the bundled
sample documents every option with its default. A missing file is fine —
defaults apply. Every key can also be set at launch through the
`MTLD3D_CONFIG` environment variable (semicolon-separated `key=value`
entries; env wins over the file).

Logging is controlled by `RUST_LOG`; `RUST_LOG=mtld3d=warn` is the single
switch for the whole project. On stock Wine, export both in the environment
that launches the game; on CrossOver, set them under
`[EnvironmentVariables]` in the bottle's `cxbottle.conf`.
