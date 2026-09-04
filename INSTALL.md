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
  x86_64-windows/           the same two files, 64-bit
  x86_64-unix/
    mtld3d.so               Metal-side unix library
  aarch64-unix/
    mtld3d.so               the same library for an arm64 Wine
native/                     unmarked d3d9.dll for the DLL-override route
  i386-windows/d3d9.dll
  x86_64-windows/d3d9.dll
prefix-markers/             for a prefix wineboot never stamped (see below)
  syswow64/mtld3d.dll       stub, copy into the prefix dir of the same name
  system32/mtld3d.dll       stub, 64-bit
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

The `.so` ships for two architectures. Wine loads the one under the directory
matching the arch of the Wine build itself: an x86_64 Wine takes
`x86_64-unix/` and ignores `aarch64-unix/`, an arm64 Wine the other way
round. Nothing to choose: copy both, or the one your Wine needs. The PE side
is x86 in either case.

Common to both routes: `mtld3d.dll` + `mtld3d.so` are a custom-named Wine
builtin pair — the PE half can only reach its unix half when loaded as a
builtin, so there is no native variant of it. And Wine resolves builtin
*names* through the prefix's system directories, not through `lib/wine`: a
builtin only loads if a marker for its name is already sitting in `system32`
or `syswow64`.

`wineboot` writes those markers when it creates a prefix, one for every
builtin it finds in `lib/wine` at that moment. Custom names are not special
here; `mtld3d` gets a marker for free as long as it was installed **before**
the prefix was created.

Installing by hand is usually the other case: the prefix already exists, so it
never saw `mtld3d`. That is what `prefix-markers/` is for. The two directories
are named after their destination and the stubs already carry the right name,
so each is a plain copy with no rename. `wineboot -u` would do the same job at
the cost of a full prefix update. Nothing in `prefix-markers/` ever belongs in
`lib/wine`.

The CrossOver route below always needs them. There `mtld3d` reaches Wine
through the bottle's DLL search path and never enters CrossOver's own
`lib/wine`, so no `wineboot` will ever stamp a marker for it.

## Requirements

What the test suites run on in CI:

- **macOS 15 or macOS 26**, on Apple Silicon or Intel.
- A Wine from [wine-build](https://github.com/athei/wine-build), the release
  CI pins, which is based on **CrossOver 26**. CrossOver 27 with its arm64 Wine
  has been tested by hand. Older Wine or CrossOver releases are not expected to
  work.
- A **64-bit prefix / bottle** — 32-bit games run in it through WoW64.
- **Rosetta 2** (`softwareupdate --install-rosetta`) for an x86_64 Wine, which
  is what most builds are: the game and the whole PE side are x86. An arm64
  Wine brings its own x86 translation (FEX) and does not need it.

## x87 performance

D3D9-era games do their floating-point math in x87 instructions, which
Rosetta 2 translates slowly. Under an x86_64 Wine, run the game together with
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

# One-time prefix markers, only needed because this prefix already existed.
# Skip both lines if the prefix is created after the copy above.
cp prefix-markers/syswow64/mtld3d.dll "$WINEPREFIX/drive_c/windows/syswow64/"
cp prefix-markers/system32/mtld3d.dll "$WINEPREFIX/drive_c/windows/system32/"

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
cp wine/x86_64-unix/mtld3d.so     "$WINE/lib/wine/x86_64-unix/"    # x86_64 Wine
#cp wine/aarch64-unix/mtld3d.so   "$WINE/lib/wine/aarch64-unix/"   # arm64 Wine

# One-time prefix markers, as in the builtin route.
cp prefix-markers/syswow64/mtld3d.dll "$WINEPREFIX/drive_c/windows/syswow64/"
cp prefix-markers/system32/mtld3d.dll "$WINEPREFIX/drive_c/windows/system32/"

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
cp prefix-markers/syswow64/mtld3d.dll "$BOTTLE/drive_c/windows/syswow64/"
cp prefix-markers/system32/mtld3d.dll "$BOTTLE/drive_c/windows/system32/"

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

- **Graphics setting**: any selection is fine, including *Auto*. The switch
  (and the per-application database that *Auto* consults) only redirects
  Wine's *builtin* DLL search, which the `d3d9=native` override bypasses:
  d3d9 stays with mtld3d in every position. Other Direct3D versions keep
  following the bottle's backend selection.
- **If d3d9 ever loads wrong anyway**: CrossOver's compatibility database
  can ship per-game DLL overrides that outrank the bottle's registry
  override. No game is known to carry one for d3d9; the tell would be
  wined3d `GLSL` fixme lines in the log and no `mtld3d` lines, and
  `WINEDEBUG=+cxcompatdb` names any applied rule.
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

## Troubleshooting

**The two lines that say mtld3d is live.** Every run writes these into its
log file (`mtld3d-logs/<exe>-<pid>.log` next to the executable) before the
game draws anything, and no `RUST_LOG` setting is needed to see them:

```
[mtld3d::shim] mtld3d.dll <version> <id>, unix call initialized
[mtld3d::d3d9] d3d9.dll <version> <id> loaded at <address>
```

If they are absent, `d3d9.dll` never mapped, and every other symptom follows
from that. Check the prefix markers first, then the override.

**World of Warcraft 1.12 dies at startup with `ERROR #132`.** The report reads
`0xC0000005 (ACCESS_VIOLATION) at 0107:0063A915`, "referenced memory at
`0x00000054`", with `EAX=00000000`. This is a load failure, not a rendering
bug, and the game never reached its first frame.

The client loads `d3d9.dll` through `LoadLibrary` during its hardware survey,
long before it creates a device, and calls `Direct3DCreate9` and
`GetDeviceCaps` to classify the GPU. When that probe fails it finds no row in
its video-hardware table and dereferences the null result. So the crash means
Wine could not load `d3d9.dll` at all. Verified against both failure shapes:
missing prefix marker and missing `d3d9.dll` produce this identical report.

Check, in order:

- `mtld3d.dll` exists in the prefix's `syswow64` (32-bit game) or `system32`
  (64-bit game). This is the marker step in the route sections above, and it is
  the usual cause: `d3d9.dll` imports `mtld3d.dll`, so a missing marker makes
  `d3d9.dll` itself unloadable.
- For the native-override route, `d3d9` is set to `native` under
  `HKCU\Software\Wine\DllOverrides`, and the `d3d9.dll` next to the game came
  from `native/`, not from `wine/`. A builtin-marked PE never executes its own
  bytes.
- The unix side is installed for the arch of the Wine build, not of the game.
- On an ARM64 Wine, or on macOS older than 15.4, use 0.6.0 or newer. Earlier
  releases carry BMI instructions that neither `xtajit` nor pre-15.4 Rosetta
  can execute, so module init aborts and the load fails the same way.

The game's own `Logs/gx.log` is not useful here: the crash happens before that
file is opened, so it is empty either way. An empty `Loaded Modules` section in
the report only means the game directory has no usable `dbghelp.dll`.

## Fullscreen

A fullscreen game sets the display mode it picked, as on Windows, and gets a
borderless window covering the monitor. To keep that mode-set virtual, so the
desktop keeps its resolution and other windows stay where they are, set Wine's
`EmulateModeset` key in the prefix (on CrossOver, through the bottle's `wine`
as in the section above):

```sh
wine reg add 'HKCU\Software\Wine\X11 Driver' /v EmulateModeset /d Y /f
```

win32u reads the key whatever the driver, once per Wine session, so set it
before launching the game and let the previous session exit first. With it the
game window is scaled onto the physical display, letterboxed if the aspect
differs, and mouse input is mapped into the mode. Without it, Wine's mac
driver switches the whole desktop to the game's mode.

The resolution picked in the game's video options therefore sizes the frame.
`render.scale` in `mtld3d.conf` multiplies on top of it, rendering fewer
pixels and upscaling the result to the screen.

The resolution list a game sees carries sizes of the display's own aspect
only, largest first and at most 15 per colour format, because Wine's full list
overflows menus built for a driver's short one. Any mode Wine accepts stays
settable whether listed or not. A request that matches no mode, such as a size
a game derived from its own window, follows the window instead. A fullscreen
game is never told it lost its device on a focus change: the desktop mode
comes back on deactivation and the game's mode is set again on activation.
