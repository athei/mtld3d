# Installing mtld3d

Installation guide for the release bundle (`mtld3d.tar.xz`), on stock Wine and
CrossOver bottles. Building from source and `make install` are in the source
repository's `docs/BUILDING.md`.

## Requirements

- macOS 15 or macOS 26, on Apple Silicon or Intel. CI covers all but macOS 26
  on Intel, which has no runners and is expected to work.
- A Wine from [wine-build](https://github.com/athei/wine-build). Its arm64
  Wine is the primary target, and its x86_64 Wine, based on CrossOver 26, which
  an Intel Mac runs and CI pins, takes the x86_64 build. Older Wine or
  CrossOver releases are not expected to work.
- macOS 26.4 or later for the arm64 Wine, whose binaries are built for it. An
  Apple Silicon Mac below that takes the x86_64 Wine under Rosetta 2.
- A 64-bit prefix or bottle. 32-bit games run in it through WoW64.
- Rosetta 2 (`softwareupdate --install-rosetta`) for an x86_64 Wine. An arm64
  Wine brings its own x86 translation (FEX).
- [x87sidecar](https://github.com/athei/x87sidecar) under an x86_64 Wine, a JIT
  that replaces Rosetta's slow handling of the x87 math in D3D9-era games. Its
  cooperative attach mode needs the startup handshake that wine-build's Wines
  carry, and then no entitlements.

## Choosing a route

mtld3d replaces Wine's d3d9 in one of two ways: as a builtin in the Wine tree,
or as a native `d3d9.dll` in the prefix, selected by a DLL override.

| | Builtin | Native override |
| --- | --- | --- |
| Use for | A Wine installation you own: your own build, an app-bundled Wine, a package install | Stock Wine, when the stock d3d9 should keep serving other prefixes |
| Covers | Every d3d9 application in that Wine tree | Every application of its arch in the prefix |
| Cost | A Wine update or reinstall wipes it. Other d3d9 applications lose the stock implementation | A registry override plus a `d3d9.dll` copy per prefix |
| CrossOver | Not possible: `CrossOver.app` is replaced on every update | Required |

## Stock Wine, shared steps

Both stock routes start here and run in one terminal, because they share its
variables. Export `WINE`, the directory containing `lib/wine/`, and
`WINEPREFIX`.

In the bundle's directory, extract it:

```sh
tar -xf mtld3d.tar.xz
```

`ls "$WINE/lib/wine"` shows `aarch64-unix` on an arm64 Wine and `x86_64-unix`
on an x86_64 one. `ls wine prefix-markers` shows what the bundle has. Run one
of these four blocks.

An arm64 Wine with a bundle that has `prefix-markers/system32-arm64`:

```sh
PE64=aarch64-windows DIR64=aarch64-windows UNIX=aarch64-unix MARKER64=system32-arm64
```

An arm64 Wine with a bundle that has `wine/aarch64-windows` but no
`prefix-markers/system32-arm64`, such as 0.14.0:

```sh
PE64=aarch64-windows DIR64=aarch64-windows UNIX=aarch64-unix MARKER64=system32
```

An arm64 Wine with a bundle without `wine/aarch64-windows`:

```sh
PE64=x86_64-windows DIR64=aarch64-windows UNIX=aarch64-unix MARKER64=system32
```

An x86_64 Wine:

```sh
PE64=x86_64-windows DIR64=x86_64-windows UNIX=x86_64-unix MARKER64=system32
```

Then copy the builtin pair into `T`, the tree this Wine loads mtld3d from:

```sh
T="$WINE/lib/wine/d3d9/mtld3d"; [ -d "$T" ] || T="$WINE/lib/wine"
WIN="$WINEPREFIX/drive_c/windows"
cp wine/i386-windows/mtld3d.dll "$T/i386-windows/"
cp "wine/$PE64/mtld3d.dll" "$T/$DIR64/"
cp "wine/$UNIX/mtld3d.so" "$T/$UNIX/"
```

If the prefix existed before this install, add its
[markers](#prefix-markers):

```sh
cp prefix-markers/syswow64/mtld3d.dll "$WIN/syswow64/"
cp "prefix-markers/$MARKER64/mtld3d.dll" "$WIN/system32/"
```

To configure the game, copy the sample next to its executable. On a later
run, skip this: it overwrites your edits.

```sh
cp mtld3d.conf "/path/to/MyGame/"
```

## Stock Wine, builtin route

After the shared steps, replace the `d3d9.dll` in `T`. After any Wine update or
reinstall, run the shared steps and this block again.

```sh
cp wine/i386-windows/d3d9.dll "$T/i386-windows/"
cp "wine/$PE64/d3d9.dll" "$T/$DIR64/"
```

## Stock Wine, native-override route

After the shared steps, run this block. A wine-build tree `T` also holds a
builtin `d3d9.dll`, which every process without the override loads, so it is
replaced to keep each process on one build; any other Wine's own `d3d9.dll`
stays. `-n` keeps the first backup when this runs again, and `rm -f` removes
a game-folder copy, which would win over the prefix copy.

```sh
if [ "$T" != "$WINE/lib/wine" ]; then
  cp wine/i386-windows/d3d9.dll "$T/i386-windows/"
  cp "wine/$PE64/d3d9.dll" "$T/$DIR64/"
fi
cp -n "$WIN/system32/d3d9.dll" "$WIN/system32/d3d9.dll.orig" || true
cp -n "$WIN/syswow64/d3d9.dll" "$WIN/syswow64/d3d9.dll.orig" || true
rm -f "/path/to/MyGame/d3d9.dll"
```

Copy the native `d3d9.dll` into the prefix's system directory, never the game
folder. Run both for games of both arches. For a 32-bit game:

```sh
cp native/i386-windows/d3d9.dll "$WIN/syswow64/"
```

For a 64-bit game:

```sh
cp "native/$PE64/d3d9.dll" "$WIN/system32/"
```

Then set the DLL override, so Wine loads the native `d3d9.dll` instead of its
builtin:

```sh
wine reg add 'HKCU\Software\Wine\DllOverrides' /v d3d9 /d native /f
```

The native `d3d9.dll` goes into the prefix's system directory because Steam's
overlay looks d3d9 up at its system path, and Wine matches that path only
against a module loaded from there. The prefix copy survives wineboot prefix
updates, which only replace placeholder files, never real PEs.

With the `d3d9=native` override set, a process of the other arch finds only
the placeholder, which `native` refuses, so d3d9 fails to load in it. A prefix
that runs both 32-bit and 64-bit d3d9 games needs both copies.

After an mtld3d upgrade or a Wine update or reinstall, run the shared steps and
this route's first block again and re-copy `d3d9.dll` into the prefix, all from
one build.

On a wine-build Wine, a process that a per-game compatdb rule gives another
d3d9 does not search `T`, finds no `mtld3d.dll` and fails to load the native
`d3d9.dll`.

## CrossOver

`d3d9.dll` loads as a native DLL from the bottle's system directory, and the
builtin pair comes through the bottle's DLL search path. Nothing is written
into `CrossOver.app`.

Set `MyBottle` to your bottle and `GAME_DIR` to the directory holding the game
`.exe`. The builtin-marked `d3d9.dll` that `cp -R` brings along is inert: it
can never shadow CrossOver's own builtin.

```sh
tar -xf mtld3d.tar.xz
BOTTLE="$HOME/Library/Application Support/CrossOver/Bottles/MyBottle"
GAME_DIR="$BOTTLE/drive_c/Program Files/MyGame"
WIN="$BOTTLE/drive_c/windows"
cp -R wine/. "$BOTTLE/mtld3d"
cp prefix-markers/syswow64/mtld3d.dll "$WIN/syswow64/"
cp prefix-markers/system32/mtld3d.dll "$WIN/system32/"
cp -n "$WIN/system32/d3d9.dll" "$WIN/system32/d3d9.dll.orig" || true
cp -n "$WIN/syswow64/d3d9.dll" "$WIN/syswow64/d3d9.dll.orig" || true
rm -f "$GAME_DIR/d3d9.dll"
```

To configure the game:

```sh
cp mtld3d.conf "$GAME_DIR/"
```

Copy the native `d3d9.dll` into the bottle's system directory, never the game
folder, for the same reasons as on stock Wine. A bottle that runs both 32-bit
and 64-bit d3d9 games needs both copies. For a 32-bit game:

```sh
cp native/i386-windows/d3d9.dll "$WIN/syswow64/"
```

For a 64-bit game:

```sh
cp native/x86_64-windows/d3d9.dll "$WIN/system32/"
```

Then set the DLL override:

```sh
CX_WINE="/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin/wine"
"$CX_WINE" --bottle MyBottle reg add 'HKCU\Software\Wine\DllOverrides' /v d3d9 /d native /f
```

Finally, add the `mtld3d` directory to the bottle's DLL search path in
`$BOTTLE/cxbottle.conf` (create the `[Wine]` section if it doesn't exist):

```ini
[Wine]
"DllPath" = "${CX_ROOT}/lib/wine/x86_64-windows:${CX_ROOT}/lib/wine/i386-windows:${CX_ROOT}/lib/wine:${WINEPREFIX}/mtld3d"
```

`DllPath` replaces the launcher's `WINEDLLPATH`. ntdll still searches its own
directory first, but the launcher finds `winewrapper.exe`, which it prepends to
every launch, through this value, so CrossOver's own directories must stay
listed. A `WINEDLLPATH` under `[EnvironmentVariables]` would be overwritten, so
it must be the `[Wine]` key.

- Any Graphics setting is fine, including *Auto*: the switch and the database
  *Auto* consults only redirect Wine's builtin search, which `d3d9=native`
  bypasses. Other Direct3D versions follow the bottle's backend selection.
- CrossOver's compatibility database can ship per-game DLL overrides that
  outrank the bottle's, though no game is known to carry one for d3d9. The sign
  is wined3d `GLSL` fixme lines in the log and no `mtld3d` lines.
  `WINEDEBUG=+cxcompatdb` names any applied rule.
- CrossOver updates keep bottles. If one migrates the bottle and refreshes
  `drive_c/windows`, re-copy the two prefix markers and `d3d9.dll`.
- To upgrade mtld3d, re-copy `d3d9.dll` and `$BOTTLE/mtld3d`, all from one
  build.

## Configuration and logging

`mtld3d.conf` is read from the directory of the running `.exe`, and a missing
file means defaults. The bundled sample documents every option. Every key can
also be set through the `MTLD3D_CONFIG` environment variable
(semicolon-separated `key=value` entries), which wins over the file.

`RUST_LOG=mtld3d=warn` is the single logging switch for the whole project. On
stock Wine, export both in the environment that launches the game. On
CrossOver, set them under `[EnvironmentVariables]` in `cxbottle.conf`.

## Troubleshooting

### The two lines that say mtld3d is live

Every run writes these to `mtld3d-logs/<exe>-<pid>.log` next to the executable
before the game draws anything, with no `RUST_LOG` setting needed:

```
[mtld3d::shim] mtld3d.dll <version> <id>, unix call initialized as <arch>
[mtld3d::d3d9] d3d9.dll <version> <id> loaded at <address> as <arch>
```

| `<arch>` | Build the process runs |
| --- | --- |
| `i686` | 32-bit game |
| `x86_64` | The x86_64 build |
| `arm64ec` | The ARM64X build in an x64 game on an arm64 Wine |
| `aarch64` | A native ARM64 program |

The unix side's own line, `mtld3d.so <version> <id> initialized as <arch>`,
reads `as x86_64, under Rosetta 2` when it runs through Rosetta 2.

If they are absent, `d3d9.dll` never mapped, and every other symptom follows
from that. Check the prefix markers first, then the override.

### World of Warcraft 1.12 dies at startup with `ERROR #132`

The report reads `0xC0000005 (ACCESS_VIOLATION) at 0107:0063A915`, "referenced
memory at `0x00000054`", with `EAX=00000000`. This is a load failure: a missing
prefix marker and a missing `d3d9.dll` produce this identical report. The
client's hardware survey calls `LoadLibrary`, `Direct3DCreate9` and
`GetDeviceCaps` before it creates a device, and a failed probe dereferences a
null video-hardware row.

Check, in order:

1. `mtld3d.dll` is in the prefix's `syswow64` (32-bit game) or `system32`
   (64-bit game). This marker step is the usual cause: `d3d9.dll` imports it.
2. On the native-override route, `d3d9` is `native` under
   `HKCU\Software\Wine\DllOverrides`, and the `d3d9.dll` in that directory
   came from `native/`, not `wine/`.
3. The unix side is installed for the arch of the Wine build, not of the game.
4. On an ARM64 Wine, or on macOS older than 15.4, use 0.6.0 or newer. Earlier
   releases carry BMI instructions that neither `xtajit` nor pre-15.4 Rosetta
   can execute, so the load fails the same way.

The game's `Logs/gx.log` stays empty in this crash. An empty `Loaded Modules`
section only means the game directory has no usable `dbghelp.dll`.

### The Steam overlay does nothing

The sign is no `GameOverlay: started` line for the game in
`Steam/logs/console_log.txt`, because `gameoverlayui.exe` or
`gameoverlayui64.exe` never starts. On the native-override route, put
`d3d9.dll` in the prefix's system directory and leave no copy in the game
folder.

## Fullscreen

By default Wine's mac driver switches the whole desktop to the game's mode. To
keep the desktop's resolution and window positions, make the mode-set virtual
(on CrossOver, through the bottle's `wine`):

```sh
wine reg add 'HKCU\Software\Wine\X11 Driver' /v EmulateModeset /d Y /f
```

win32u reads the key whatever the driver, once per Wine session, so set it
before launching the game and let the previous session exit first. The game
window is then scaled onto the display, letterboxed if the aspect differs, with
mouse input mapped into the mode.

### System DPI other than 96

Wine scales the window size and mouse coordinates of a game that does not
declare itself DPI-aware by 96 over that DPI, but not the display mode list.
In fullscreen the game fills a fraction of the screen, its clicks miss the
drawn pointer, and the pointer cannot reach the far edges. mtld3d logs a
warning that names the mismatch.

Mark the executable DPI-aware in the prefix, under its file name without the
path, and restart the game:

```sh
wine reg add 'HKCU\Software\Microsoft\Windows NT\CurrentVersion\AppCompatFlags\Layers' /v hl2.exe /d '~ HIGHDPIAWARE' /f
```

With `/ve` in place of `/v hl2.exe` it applies to every program in the prefix.
That default is a Wine extension, which Windows does not read.

### Resolutions

The game's video options size the frame. `render.scale` in `mtld3d.conf`
multiplies on top of it, rendering fewer pixels and upscaling.

- The adapter lists sizes smallest first. A menu built from Wine's own
  display-mode list keeps Wine's order.
- Wine lists forty-odd sizes on a Retina display, which can overflow a menu of
  fixed length. `display.maxResolutions` in `mtld3d.conf` then offers only the
  largest. The World of Warcraft profile sets it to 32.
- Any mode Wine accepts stays settable, listed or not, except the sizes some
  Wine builds abort on (`docs/STATUS.md`, Kept divergences).
- A request that matches no mode follows the window.
- A fullscreen game keeps its device on a focus change: the desktop mode comes
  back on deactivation and the game's mode on activation.

## Bundle contents

```
wine/                  lib/wine-shaped tree, every PE builtin-marked
  i386-windows/        d3d9.dll, Direct3D 9
                       mtld3d.dll, PE half of the unix-call bridge
  x86_64-windows/      the same two, for an x86_64 Wine
  aarch64-windows/     the same two as ARM64X images, for an arm64 Wine
  x86_64-unix/         mtld3d.so, Metal-side unix library
  aarch64-unix/        mtld3d.so for an arm64 Wine
native/                unmarked d3d9.dll for the DLL-override route
  i386-windows/
  x86_64-windows/
  aarch64-windows/     ARM64X
prefix-markers/        mtld3d.dll stubs for a prefix wineboot never stamped
  syswow64/
  system32/
  system32-arm64/
mtld3d.conf            sample configuration
INSTALL.md             this file
LICENSE
```

Releases after 0.13.0 are built against an arm64 Wine and have the
`aarch64-windows` entries. Those after 0.14.0 also have
`prefix-markers/system32-arm64/`. 0.13.0 and older releases, and any bundle
built against an x86_64 Wine, have neither.

`d3d9.dll` is one binary in two forms:

- `wine/<arch>/d3d9.dll` carries Wine's builtin signature. In a Wine tree's
  `lib/wine` it replaces the stock d3d9 for that whole tree. Wine never
  executes it outside its builtin search path, so it cannot be an override.
- `native/<arch>/d3d9.dll` is an ordinary native PE, loaded through a
  `d3d9=native` override.

`mtld3d.dll` has no native form: it reaches `mtld3d.so`, its custom-named
builtin pair, only when loaded as a builtin.

### Where the files go

| Wine | Game | Build in `wine/` and `native/` | Into `T/` | Native `d3d9.dll` into |
| --- | --- | --- | --- | --- |
| Either | 32-bit | `i386-windows` | `i386-windows` | `syswow64` |
| arm64 | 64-bit | `aarch64-windows` | `aarch64-windows` | `system32` |
| arm64, bundle without `aarch64-windows` | 64-bit | `x86_64-windows` | `aarch64-windows` | `system32` |
| x86_64 | 64-bit | `x86_64-windows` | `x86_64-windows` | `system32` |

`mtld3d.so` goes from `wine/<arch>-unix` to the same directory of `T`, for the
arch of the Wine, not of the game. `T` is `$WINE/lib/wine/d3d9/mtld3d` in a
wine-build Wine of either arch, whose subdirectories mirror `lib/wine`, and
`$WINE/lib/wine` in any other Wine.

- A wine-build Wine's default `lib/wine` directories hold only placeholders.
  Never copy over them: an x64 process loads the ARM64X pair behind either
  kind the Wine may ship, AMD64 or hybrid (ARM64X).
- An arm64 Wine loads x64 builtins from `aarch64-windows`, never from
  `x86_64-windows`, so only one 64-bit pair can be installed. The ARM64X pair
  is faster: it runs as native code, where the x86_64 pair is translated.
- The source tree's `make install` does all of this for a wine-build Wine and
  re-stamps the placeholders (see `docs/BUILDING.md`).

### Prefix markers

A builtin loads only if the prefix's `system32` or `syswow64` holds a marker
for its name. `wineboot` writes one for every builtin in `lib/wine` when it
creates a prefix, so a prefix created after the install needs nothing. An
existing prefix takes the stubs below, or a `wineboot -u`, which is a full
prefix update and on a wine-build Wine stamps markers of the right machine.

| Stub in `prefix-markers/` | Into | Machine | Works for |
| --- | --- | --- | --- |
| `syswow64/mtld3d.dll` | `syswow64` | 32-bit | 32-bit programs on either Wine |
| `system32-arm64/mtld3d.dll` | `system32` | hybrid (ARM64X) | x64 and native ARM64 programs on an arm64 Wine. An x86_64 Wine refuses it |
| `system32/mtld3d.dll` | `system32` | AMD64 | An x86_64 Wine. On an arm64 Wine, x64 programs only, for a bundle without `system32-arm64` |

Nothing in `prefix-markers/` belongs in `lib/wine`. CrossOver always needs the
stubs: `mtld3d` never enters its `lib/wine`, so no `wineboot` stamps a marker.
