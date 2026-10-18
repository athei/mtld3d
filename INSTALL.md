# Installing mtld3d

This is the installation guide for the release bundle (`mtld3d.tar.xz`). It
covers stock Wine installations and CrossOver bottles. Building from source
and the developer `make install` flow are covered in the source repository's
`docs/BUILDING.md`.

## Bundle contents

```
wine/                       lib/wine-shaped tree, every PE builtin-marked
  i386-windows/
    d3d9.dll                Direct3D 9 implementation (builtin-marked)
    mtld3d.dll              PE half of the unix-call bridge (builtin-marked)
  x86_64-windows/           the same two files, 64-bit, for an x86_64 Wine
  aarch64-windows/          the same two files as ARM64X images, for an arm64 Wine
  x86_64-unix/
    mtld3d.so               Metal-side unix library
  aarch64-unix/
    mtld3d.so               the same library for an arm64 Wine
native/                     unmarked d3d9.dll for the DLL-override route
  i386-windows/d3d9.dll
  x86_64-windows/d3d9.dll
  aarch64-windows/d3d9.dll  ARM64X, for a 64-bit game on an arm64 Wine
prefix-markers/             for a prefix wineboot never stamped (see below)
  syswow64/mtld3d.dll       stub, copy into the prefix dir of the same name
  system32/mtld3d.dll       stub, 64-bit, for an x86_64 Wine (AMD64)
  system32-arm64/mtld3d.dll stub, 64-bit, for an arm64 Wine (hybrid ARM64X);
                            copy it into system32
mtld3d.conf                 sample configuration, self-documenting
INSTALL.md                  this file
LICENSE
```

Only a bundle built against an arm64 Wine has the `aarch64-windows` entries
of `wine/` and `native/`. Releases after 0.13.0 are built that way; 0.13.0
and older releases, and any bundle built against an x86_64 Wine, have
neither, and on an arm64 Wine their x86_64 pair takes that place (see
below). `prefix-markers/system32-arm64/` comes in releases after 0.14.0,
built against an arm64 Wine.

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
round. Nothing to choose: copy both, or the one your Wine needs.

The 64-bit PE pair ships in two builds, and each Wine takes one of them:

- An x86_64 Wine takes the x86_64 pair, `wine/x86_64-windows/`. It runs the
  whole PE side as x86 code, through Rosetta on Apple Silicon, and never
  reads an `aarch64-windows` directory.
- An arm64 Wine takes the ARM64X pair, `wine/aarch64-windows/`, when the
  bundle has it. There an x64 program runs as an ARM64EC process, which loads
  its builtins from `aarch64-windows`, never from `x86_64-windows`, and the
  ARM64X images carry an ARM64EC half that the process runs as native code,
  where the x86_64 pair would be translated. The x86_64 pair also loads
  there, emulated, which is the route for a bundle without
  `wine/aarch64-windows/`: the x86_64 pair goes into that same
  `aarch64-windows` directory. The two share the directory and the file
  names, so only one can be installed, and the ARM64X pair is the faster one.

32-bit games load the i386 pair, `wine/i386-windows/`, on either Wine.

Where the files go also depends on the layout of the Wine. A Wine from
[wine-build](https://github.com/athei/wine-build), for either arch, keeps
each Direct3D implementation in a tree of its own and loads mtld3d from
`lib/wine/d3d9/mtld3d`, whose subdirectories mirror those of `lib/wine`; its
default directories hold only placeholders, which stay as they are. Any other
Wine loads mtld3d from `lib/wine` itself. On an arm64 Wine, copy the 64-bit
pair into the `aarch64-windows` of the tree mtld3d loads from, never over the
placeholders in a wine-build Wine's default `lib/wine/aarch64-windows`. Leave
those placeholders as they are: an x64 process loads the ARM64X pair behind
either kind the Wine may ship, AMD64 or hybrid (ARM64X). `make install`
re-stamps them the way wine-build's bundle does, hybrid for the ARM64X pair,
so that a native ARM64 program can load the names too, and AMD64 for the
x86_64 pair; it needs wine-build's arm64-11.19-2 or later, whose winebuild
writes hybrid placeholders. A prefix takes a changed placeholder on its next
`wineboot -u`. The routes below spell out both layouts. The source tree's
`make install` does all of this for a wine-build Wine (see
`docs/BUILDING.md`).

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
never saw `mtld3d`. That is what `prefix-markers/` is for. The directories
are named after their destination and the stubs already carry the right name,
so each is a plain copy with no rename. The 64-bit stub comes in two machines,
because each Wine's loader takes a different one: `system32/` holds an AMD64
stub for an x86_64 Wine, and `system32-arm64/` a hybrid (ARM64X) one for an
arm64 Wine, which both its x64 programs and its native ARM64 programs accept;
an x86_64 Wine refuses that one, and a native ARM64 program refuses the AMD64
one. Both go into the prefix's `system32`. `wineboot -u` would do the same
job at the cost of a full prefix update, and a wine-build Wine's `wineboot`
stamps markers of the right machine by itself. Nothing in `prefix-markers/`
ever belongs in `lib/wine`.

The CrossOver route below always needs them. There `mtld3d` reaches Wine
through the bottle's DLL search path and never enters CrossOver's own
`lib/wine`, so no `wineboot` will ever stamp a marker for it.

## Requirements

- macOS 15 or macOS 26, on Apple Silicon or Intel. CI covers macOS 15 and 26
  on Apple Silicon and macOS 15 on Intel, because no macOS 26 Intel runners
  exist. Intel on macOS 26 is expected to work.
- A Wine from [wine-build](https://github.com/athei/wine-build). Its arm64
  Wine is the primary target; an x86_64 Wine from wine-build, based on
  CrossOver 26, which an Intel Mac runs and CI pins, takes the x86_64 pair.
  Older Wine or CrossOver releases are not expected to work.
- macOS 26.4 or later for the arm64 Wine, whose binaries are built for it.
  On an Apple Silicon Mac below that, such as one on macOS 15, the x86_64
  Wine under Rosetta 2 is the route.
- A **64-bit prefix / bottle**; 32-bit games run in it through WoW64.
- **Rosetta 2** (`softwareupdate --install-rosetta`) for an x86_64 Wine, under
  which the game and the whole PE side are x86. An arm64 Wine brings its own
  x86 translation (FEX) and does not need it.

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

# Where this Wine loads mtld3d from: its own tree in a wine-build Wine,
# lib/wine itself otherwise.
T="$WINE/lib/wine/d3d9/mtld3d"; [ -d "$T" ] || T="$WINE/lib/wine"

# Replaces the d3d9.dll there and adds the mtld3d builtin pair. arm64 Wine:
cp wine/i386-windows/*.dll     "$T/i386-windows/"
cp wine/aarch64-windows/*.dll  "$T/aarch64-windows/"
cp wine/aarch64-unix/mtld3d.so "$T/aarch64-unix/"
# arm64 Wine, a bundle without wine/aarch64-windows: the x86_64 pair instead.
#cp wine/x86_64-windows/*.dll   "$T/aarch64-windows/"
# x86_64 Wine, instead:
#cp wine/i386-windows/*.dll    "$T/i386-windows/"
#cp wine/x86_64-windows/*.dll  "$T/x86_64-windows/"
#cp wine/x86_64-unix/mtld3d.so "$T/x86_64-unix/"

# One-time prefix markers, only needed because this prefix already existed.
# Skip both lines if the prefix is created after the copy above. arm64 Wine:
cp prefix-markers/syswow64/mtld3d.dll       "$WINEPREFIX/drive_c/windows/syswow64/"
cp prefix-markers/system32-arm64/mtld3d.dll "$WINEPREFIX/drive_c/windows/system32/"
# x86_64 Wine, or a bundle without system32-arm64, the AMD64 stub instead:
#cp prefix-markers/system32/mtld3d.dll      "$WINEPREFIX/drive_c/windows/system32/"

# Optional: runtime configuration next to the game executable.
cp mtld3d.conf "/path/to/MyGame/"
```

Re-run the `lib/wine` copy after any Wine update or reinstall.

## Stock Wine, native-override route

```sh
tar -xf mtld3d.tar.xz

# The mtld3d builtin pair goes into the installation. Where it goes, as in
# the builtin route:
T="$WINE/lib/wine/d3d9/mtld3d"; [ -d "$T" ] || T="$WINE/lib/wine"
# A wine-build Wine's tree T also holds a builtin d3d9.dll of the mtld3d it
# shipped with, which every process without the override loads: copy it from
# this bundle too (all of wine/<arch>/*.dll), so each process gets a d3d9.dll
# and an mtld3d.dll of one build. Any other Wine's lib/wine holds Wine's own
# d3d9.dll, which stays: copy only mtld3d.dll there (wine/<arch>/mtld3d.dll).
[ "$T" = "$WINE/lib/wine" ] && P=mtld3d || P='*'
# arm64 Wine:
cp wine/i386-windows/$P.dll     "$T/i386-windows/"
cp wine/aarch64-windows/$P.dll  "$T/aarch64-windows/"
cp wine/aarch64-unix/mtld3d.so  "$T/aarch64-unix/"
# arm64 Wine, a bundle without wine/aarch64-windows: the x86_64 pair instead.
#cp wine/x86_64-windows/$P.dll  "$T/aarch64-windows/"
# x86_64 Wine, instead:
#cp wine/i386-windows/$P.dll    "$T/i386-windows/"
#cp wine/x86_64-windows/$P.dll  "$T/x86_64-windows/"
#cp wine/x86_64-unix/mtld3d.so  "$T/x86_64-unix/"

# One-time prefix markers, as in the builtin route. arm64 Wine:
cp prefix-markers/syswow64/mtld3d.dll       "$WINEPREFIX/drive_c/windows/syswow64/"
cp prefix-markers/system32-arm64/mtld3d.dll "$WINEPREFIX/drive_c/windows/system32/"
# x86_64 Wine, or a bundle without system32-arm64, the AMD64 stub instead:
#cp prefix-markers/system32/mtld3d.dll      "$WINEPREFIX/drive_c/windows/system32/"

# Native d3d9.dll next to the game executable: pick the game's arch, and for
# a 64-bit game the build whose mtld3d.dll went in above.
cp native/i386-windows/d3d9.dll "/path/to/MyGame/"      # 32-bit game
#cp native/aarch64-windows/d3d9.dll "/path/to/MyGame/"  # 64-bit game, arm64 Wine
#cp native/x86_64-windows/d3d9.dll "/path/to/MyGame/"   # 64-bit game, x86_64 Wine,
#                                                       # or arm64 Wine without aarch64-windows

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

In a wine-build Wine the native `d3d9.dll` reaches the builtin `mtld3d.dll`
only in a process whose builtin search includes the tree `T`: the Wine's
compatdb.so prepends `T` for a process it selects mtld3d as the d3d9 of, and
it does so by default for every x86 process, 32-bit and 64-bit, and, from
arm64-11.19-2 on, for every native ARM64 process too. A process compatdb
gives another d3d9 (a per-game rule) does not search `T`, finds no
`mtld3d.dll` and fails to load the native `d3d9.dll`.

### On an arm64 Wine

A 64-bit game on an arm64 Wine runs as an ARM64EC process, so the route
takes the ARM64X builds for its 64-bit half: the builtin `mtld3d.dll` from
`wine/aarch64-windows/` in the tree's `aarch64-windows`, and the unmarked
`d3d9.dll` from `native/aarch64-windows/` next to the game or in the
prefix's `drive_c/windows/system32/`, with the tree's builtin `d3d9.dll`
replaced from `wine/aarch64-windows/` as above. The game then runs both as
native code. A bundle without the `aarch64-windows` entries, one built
against an x86_64 Wine, serves the same route with its x86_64 builds
instead: the pair from `wine/x86_64-windows/` in the tree's
`aarch64-windows` and the `d3d9.dll` from `native/x86_64-windows/`, all
translated. The files in the tree and the native `d3d9.dll` come from one
build either way. A 32-bit game takes the i386 files, as on an x86_64 Wine.
The 64-bit prefix marker is the hybrid one from `system32-arm64/`, which an
x64 process accepts as it does the AMD64 one and a native ARM64 process
accepts too; a bundle without it leaves the AMD64 one from `system32/`, which
serves x64 programs only. On a wine-build arm64 Wine the `mtld3d.dll` goes
into `lib/wine/d3d9/mtld3d` (the `T` above), never over the placeholders in
its default `lib/wine/aarch64-windows`.

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
[mtld3d::shim] mtld3d.dll <version> <id>, unix call initialized as <arch>
[mtld3d::d3d9] d3d9.dll <version> <id> loaded at <address> as <arch>
```

`<arch>` is the build the process runs: `i686` for a 32-bit game, `x86_64`
for the x86_64 build, `arm64ec` for the ARM64X build in an x64 game on an
arm64 Wine, and `aarch64` in a native ARM64 program. The unix side's own
line, `mtld3d.so <version> <id> initialized as <arch>`, names its arch, and
reads `as x86_64, under Rosetta 2` when it runs through Rosetta 2.

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

On a prefix whose system DPI is set to a value other than 96, Wine scales the
window size and the mouse coordinates of a game that does not declare itself
DPI-aware by 96 over that DPI, but not the display mode list. In fullscreen
the game then lays out its interface in a fraction of the screen, its clicks
land away from the drawn pointer, and the pointer cannot reach the far edges.
mtld3d logs a warning that names the mismatch when such a game goes
fullscreen. The remedy is to mark the executable DPI-aware in the prefix,
under its file name without the path:

```sh
wine reg add 'HKCU\Software\Microsoft\Windows NT\CurrentVersion\AppCompatFlags\Layers' /v hl2.exe /d '~ HIGHDPIAWARE' /f
```

With `/ve` in place of `/v hl2.exe` the value becomes the key's default and
applies to every program in the prefix; that default is a Wine extension,
which Windows does not read. The flag is read when a process starts, so
restart the game after setting it.

The resolution picked in the game's video options therefore sizes the frame.
`render.scale` in `mtld3d.conf` multiplies on top of it, rendering fewer
pixels and upscaling the result to the screen.

The resolution list a game sees describes the primary display. The list the
adapter reports carries the sizes that fill the display, largest first (those
that, scaled onto it, leave a bar of less than one pixel), and then the
standard sizes of another shape that Wine lists for it (2560x1440, 1920x1080,
1600x900, 1280x720, 1024x768, 800x600 and 640x480), largest first, which are
letterboxed in fullscreen. It holds at most 15 sizes per colour format,
because Wine's full list overflows menus built for a driver's short one. A
game that builds its menu from Wine's own display-mode list is shown the same
sizes in Wine's order, and a game that sorts its menu shows its own order. Any
mode Wine accepts stays settable whether listed or not, except the sizes some
Wine builds abort on (`docs/STATUS.md`, Kept divergences). A request that
matches no mode, such as a size a game derived from its own window, follows
the window instead. A fullscreen game is never told it lost its device on a
focus change: the desktop mode comes back on deactivation and the game's mode
is set again on activation.
