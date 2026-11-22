# Installing mtld3d

Installation guide for the release bundle (`mtld3d.tar.xz`), on stock Wine and
CrossOver Preview bottles. Building from source and `make install` are in the
source repository's `docs/BUILDING.md`. An older bundle installs by the
`INSTALL.md` shipped in it.

## Requirements

- A Mac with Apple Silicon. On any other Metal device `CreateDevice` fails
  with `D3DERR_NOTAVAILABLE`.
- macOS 26.4 or later, the oldest macOS the arm64 Wine runs on.
- An arm64 Wine, which brings its own x86 translation: the Wine engine of
  [Mach9](https://github.com/athei/mach9) v0.4.0, the release CI pins, or an
  arm64 bottle of CrossOver Preview. An x86_64 Wine cannot load mtld3d.
- A 64-bit prefix or bottle. 32-bit games run in it through WoW64.

## Choosing a route

mtld3d replaces Wine's d3d9 in one of two ways: as a builtin in the Wine tree,
or as a native `d3d9.dll` in the prefix, selected by a DLL override.

| | Builtin | Native override |
| --- | --- | --- |
| Use for | A Wine installation you own: your own build, an app-bundled Wine, or a packaged Wine if it is an arm64 build | Stock Wine, when the stock d3d9 should keep serving other prefixes |
| Covers | Every d3d9 application in that Wine tree | Every application of its arch in the prefix |
| Cost | A Wine update or reinstall wipes it. Other d3d9 applications lose the stock implementation | A registry override plus a `d3d9.dll` copy per prefix |
| CrossOver Preview | Not possible: `CrossOver Preview.app` is replaced on every update | Required |

## Stock Wine, shared steps

Both stock routes start here and run in one terminal, because they share its
variables. Export `WINE`, the directory containing `lib/wine/`, and
`WINEPREFIX`.

In the bundle's directory, extract it:

```sh
tar -xf mtld3d.tar.xz
```

`ls "$WINE/lib/wine"` shows `aarch64-unix` on an arm64 Wine. A Wine whose
`lib/wine` holds `x86_64-unix` and no `aarch64-unix` is an x86_64 Wine, which
cannot load mtld3d.

Copy the builtin pair into `T`, the tree this Wine loads mtld3d from:

```sh
T="$WINE/lib/wine/d3d9/mtld3d"; [ -d "$T" ] || T="$WINE/lib/wine"
WIN="$WINEPREFIX/drive_c/windows"
cp wine/i386-windows/mtld3d.dll "$T/i386-windows/"
cp wine/aarch64-windows/mtld3d.dll "$T/aarch64-windows/"
cp wine/aarch64-unix/mtld3d.so "$T/aarch64-unix/"
```

If the prefix existed before this install, add its
[markers](#prefix-markers):

```sh
cp prefix-markers/syswow64/mtld3d.dll "$WIN/syswow64/"
cp prefix-markers/system32-arm64/mtld3d.dll "$WIN/system32/"
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
cp wine/aarch64-windows/d3d9.dll "$T/aarch64-windows/"
```

## Stock Wine, native-override route

After the shared steps, run this block. A Mach9 tree `T` also holds a
builtin `d3d9.dll`, which every process without the override loads, so it is
replaced to keep each process on one build; any other Wine's own `d3d9.dll`
stays. `-n` keeps the first backup when this runs again, and `rm -f` removes
a game-folder copy, which would win over the prefix copy.

```sh
if [ "$T" != "$WINE/lib/wine" ]; then
  cp wine/i386-windows/d3d9.dll "$T/i386-windows/"
  cp wine/aarch64-windows/d3d9.dll "$T/aarch64-windows/"
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
cp native/aarch64-windows/d3d9.dll "$WIN/system32/"
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

On a Mach9 Wine, a process that a per-game compatdb rule gives another
d3d9 does not search `T`, finds no `mtld3d.dll` and fails to load the native
`d3d9.dll`.

## CrossOver Preview

mtld3d runs in an arm64 bottle of CrossOver Preview, where an x64 game loads
the ARM64X pair. `d3d9.dll` loads as a native DLL from the bottle's system
directory, and the builtin pair comes from a directory in the bottle that the
game's launch adds to the DLL search path. Nothing is written into
`CrossOver Preview.app`.

The CrossOver Preview window shows a bottle's architecture as *Bottle Type*
under *Advanced Settings*, and mtld3d needs `arm64`, which runs CrossOver's
arm64 Wine. It is the `WineArch` key under `[Bottle]` in the bottle's
`cxbottle.conf`. An existing arm64 bottle also needs the *CrossOver Preview*
switch on under *Advanced Settings*, the `Preview` key under `[Bottle]`:
CrossOver Preview's `wine` refuses to start a bottle without it. Without an
arm64 bottle, choose *File > New Bottle…* and a *New Bottle Type* that ends in
*(arm64)*, such as *Windows 10 (arm64)*. The *(Intel)* types run CrossOver's
x86_64 Wine, which cannot load mtld3d.

CrossOver Preview's `win32u` lacks the 16-bit ratio fix, so with
`EmulateModeset` on some fullscreen sizes end the game; *Sizes that end the
game under `EmulateModeset`* under *Fullscreen* names them.

All blocks below run in one terminal, in the bundle's directory, because they
share its variables. Set `CX` and `NAME` first. `MyBottle` is a placeholder
for the bottle's name, as the window lists it under *Bottles*. Keep the
quotes: a name can hold spaces, as the default *New Bottle* does.

```sh
CX="/Applications/CrossOver Preview.app/Contents/SharedSupport/CrossOver"
NAME="MyBottle"
```

Instead of *File > New Bottle…*, the command line creates the same bottle,
from `win10_64` or another 64-bit template:

```sh
"$CX/bin/cxbottle" --create --bottle "$NAME" --template win10_64 --winearch arm64
```

Set `GAME_DIR` to the directory holding the game `.exe`. The builtin-marked
`d3d9.dll` that `cp -R` brings along is inert: it can never shadow
CrossOver's own builtin.

```sh
tar -xf mtld3d.tar.xz
BOTTLE="$HOME/Library/Application Support/CrossOver/Bottles/$NAME"
GAME_DIR="$BOTTLE/drive_c/Program Files/MyGame"
WIN="$BOTTLE/drive_c/windows"
cp -R wine/. "$BOTTLE/mtld3d"
cp prefix-markers/syswow64/mtld3d.dll "$WIN/syswow64/"
cp prefix-markers/system32-arm64/mtld3d.dll "$WIN/system32/"
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
cp native/aarch64-windows/d3d9.dll "$WIN/system32/"
```

Then set the DLL override:

```sh
"$CX/bin/wine" --bottle "$NAME" reg add 'HKCU\Software\Wine\DllOverrides' /v d3d9 /d native /f
```

### Launching the game

mtld3d's files live in the bottle, not in the app, so Wine finds `mtld3d.dll`
and `mtld3d.so` only through `WINEDLLPATH`, and every launch has to add the
bottle's `mtld3d` directory to it. CrossOver's `wine` script sets
`WINEDLLPATH` to CrossOver's own directories on every launch, after it applies
`[EnvironmentVariables]` from `cxbottle.conf`. Only a launch's *Environment
variables* field (the `--env` option), or the `CX_ENV` variable when that field
is empty, is applied after it. `cxbottle.conf` has no key that adds a
directory, and a `WINEDLLPATH` under `HKCU\Environment` in the bottle never
reaches the loader. CrossOver's directories stay first in the value: they hold
its builtins, and its `wine` script keeps `$CX/lib/wine` there so the loader
finds `ntdll.so`.

Print the line that sets the value:

```sh
DLLS="$CX/lib/wine/aarch64-windows:$CX/lib/wine/i386-windows:$CX/lib/wine:$BOTTLE/mtld3d"
echo "WINEDLLPATH='$DLLS'"
```

Every process the game starts inherits the value. A launch without it gets
CrossOver's value instead, finds no `mtld3d.dll` and fails to load the native
`d3d9.dll`.

#### Double-click on a launcher

To give every launch from CrossOver Preview the value, put it into `CX_ENV`
for the login session. Then quit CrossOver Preview and open it again: it reads
the variable only when it starts.

```sh
launchctl setenv CX_ENV "WINEDLLPATH='$DLLS'"
```

A double-click on a launcher in the CrossOver Preview window then loads
mtld3d, and so does opening the launcher's app in `~/Applications/CrossOver`
or *Bottle > Run Command…* with an empty *Environment variables* field.

- `CX_ENV` applies to every bottle. Launches in an arm64 bottle without
  mtld3d, and in an *(Intel)* bottle, behave as they do without it.
- It holds one bottle's `mtld3d` directory. Launch a game in another bottle
  with the *Environment variables* field, as below.
- Any text in a launch's *Environment variables* field replaces `CX_ENV`. To
  add a variable to such a launch, enter the printed line first, then a space
  and the variable.

`launchctl setenv` lasts until logout. A LaunchAgent sets `CX_ENV` at every
login. Loading it sets it now as well, and CrossOver Preview needs the same
restart. To run the block again, for another bottle for example, run the
removal block below first: `launchctl bootstrap` fails while the agent is
loaded.

```sh
mkdir -p "$HOME/Library/LaunchAgents"
AGENT="$HOME/Library/LaunchAgents/local.mtld3d.cx-env.plist"
plutil -create xml1 "$AGENT"
plutil -insert Label -string local.mtld3d.cx-env "$AGENT"
plutil -insert RunAtLoad -bool YES "$AGENT"
plutil -insert ProgramArguments -array "$AGENT"
for arg in /bin/launchctl setenv CX_ENV "WINEDLLPATH='$DLLS'"; do
  plutil -insert ProgramArguments -string "$arg" -append "$AGENT"
done
launchctl bootstrap "gui/$(id -u)" "$AGENT"
```

To remove the LaunchAgent and `CX_ENV`, run this, then quit CrossOver Preview
and open it again:

```sh
launchctl bootout "gui/$(id -u)/local.mtld3d.cx-env"
rm -f "$HOME/Library/LaunchAgents/local.mtld3d.cx-env.plist"
launchctl unsetenv CX_ENV
```

#### Launching with the value each time

Without `CX_ENV`, a double-click on a launcher does not load mtld3d. Paste the
printed line, quotes included, into the *Environment variables* field of the
launch instead:

- For an installed program, select its launcher and click *Run With Options*,
  or right-click it and choose *Run with Options…*, then click *Run*.
- For any `.exe`, choose *Bottle > Run Command…* and enter its Windows path
  in *Command*, in double quotes and followed by its arguments, such as
  `"C:\Program Files\MyGame\MyGame.exe"`, then click *Run*.

Neither window keeps the field: *Run With Options* opens empty the next time,
and a launcher made with *Save Command as a Launcher* starts without it.

From the terminal, launch the game, or the launcher that starts it, with the
same value:

```sh
"$CX/bin/wine" --bottle "$NAME" --env "WINEDLLPATH='$DLLS'" "$GAME_DIR/MyGame.exe"
```

To see why a launch did not load mtld3d, tick *Create log file* in either
window and enter `+loaddll` under *Additional Wine logging channels*, or add
`--debugmsg +loaddll` to a terminal launch. The log, which the window writes
to `~/Library/Logs/CrossOver Preview`, names the cause:
`Library mtld3d.dll (which is needed by L"C:\\windows\\system32\\d3d9.dll") not found`.
A 32-bit game prints the same `system32` path, which WoW64 maps to
`syswow64`.

- Any Graphics setting is fine, including *Auto*: the switch and the database
  *Auto* consults only redirect Wine's builtin search, which `d3d9=native`
  bypasses. Other Direct3D versions follow the bottle's backend selection.
- CrossOver's compatibility database can ship per-game DLL overrides that
  outrank the bottle's, though no game is known to carry one for d3d9. The sign
  is wined3d `GLSL` fixme lines in the log and no `mtld3d` lines. Adding
  `--cx-log "$HOME/cx.log" --debugmsg +cxcompatdb` to the launch names any
  applied rule in that file.
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
CrossOver Preview, set them under `[EnvironmentVariables]` in `cxbottle.conf`,
or after the `WINEDLLPATH` line in a launch's *Environment variables* field.

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
| `arm64ec` | The ARM64X build in an x64 game |
| `aarch64` | A native ARM64 program |

The unix side writes its own line, `mtld3d.so <version> <id> initialized as
aarch64`.

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
3. `mtld3d.so` is in `aarch64-unix`, whatever the arch of the game.
4. On CrossOver Preview, the launch added the bottle's `mtld3d` directory to
   `WINEDLLPATH`, as in [Launching the game](#launching-the-game). A
   double-click on a launcher does so only while `CX_ENV` is set.

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
(on CrossOver Preview, through the bottle's `wine`):

```sh
wine reg add 'HKCU\Software\Wine\X11 Driver' /v EmulateModeset /d Y /f
```

win32u reads the key whatever the driver, once per Wine session, so set it
before launching the game and let the previous session exit first. The game
window is then scaled onto the display, letterboxed if the aspect differs, with
mouse input mapped into the mode.

### Sizes that end the game under `EmulateModeset`

With `EmulateModeset` on, a mode-set can abort the game in Wine's `win32u` on
a Wine without its 16-bit ratio fix: CrossOver Preview, upstream Wine while
[MR !12724](https://gitlab.winehq.org/wine/wine/-/merge_requests/12724) is
open, and Mach9 before v0.4.0. mtld3d lists these sizes all the same.

- After an emulated mode-set `win32u` rescales the monitor by the system DPI
  times the display's size over the mode's size, on each axis, reduces each
  ratio, and asserts that its terms fit in 16 bits.
- The sizes that fail are those whose reduced ratio does not fit. On a
  3456x2234 panel at *More Space*, a 4112x2658 desktop at 192 dpi, they
  include the panel's own 3456x2234 and the Default 1728x1117.
- Without `EmulateModeset` the mac driver switches the display itself, the
  ratio reduces to the DPI alone, and the assert always holds.
- To avoid it, pick another size in the game, or run a Mach9 release from
  v0.4.0 on.

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
- On a scaled desktop, such as *More Space*, Wine lists sizes up to the
  desktop's backing store, larger than the panel. The adapter leaves out
  every size larger than the panel but the desktop's own.
- Any other mode Wine accepts stays settable, listed or not.
- A request that matches no mode follows the window.
- A fullscreen game keeps its device on a focus change: the desktop mode comes
  back on deactivation and the game's mode on activation.

## Bundle contents

```
wine/                  lib/wine-shaped tree, every PE builtin-marked
  i386-windows/        d3d9.dll, Direct3D 9
                       mtld3d.dll, PE half of the unix-call bridge
  aarch64-windows/     the same two as ARM64X images, for x64 games
  aarch64-unix/        mtld3d.so, Metal-side unix library
native/                unmarked d3d9.dll for the DLL-override route
  i386-windows/
  aarch64-windows/     ARM64X
prefix-markers/        mtld3d.dll stubs for a prefix wineboot never stamped
  syswow64/
  system32-arm64/
mtld3d.conf            sample configuration
INSTALL.md             this file
LICENSE
```

`d3d9.dll` is one binary in two forms:

- `wine/<arch>/d3d9.dll` carries Wine's builtin signature. In a Wine tree's
  `lib/wine` it replaces the stock d3d9 for that whole tree. Wine never
  executes it outside its builtin search path, so it cannot be an override.
- `native/<arch>/d3d9.dll` is an ordinary native PE, loaded through a
  `d3d9=native` override.

`mtld3d.dll` has no native form: it reaches `mtld3d.so`, its custom-named
builtin pair, only when loaded as a builtin.

### Where the files go

| Game | Build in `wine/` and `native/` | Into `T/` | Native `d3d9.dll` into |
| --- | --- | --- | --- |
| 32-bit | `i386-windows` | `i386-windows` | `syswow64` |
| 64-bit | `aarch64-windows` | `aarch64-windows` | `system32` |

`mtld3d.so` goes from `wine/aarch64-unix` to the same directory of `T`,
whatever the arch of the game. `T` is `$WINE/lib/wine/d3d9/mtld3d` in a Mach9
Wine, whose subdirectories mirror `lib/wine`, and `$WINE/lib/wine` in any
other Wine.

- A Mach9 Wine's default `lib/wine` directories hold only placeholders.
  Never copy over them: an x64 process loads the ARM64X pair behind either
  kind the Wine may ship, AMD64 or hybrid (ARM64X).
- An arm64 Wine loads x64 builtins from `aarch64-windows`, never from
  `x86_64-windows`, and runs the ARM64X pair's ARM64EC half as native code.
- The source tree's `make install` does all of this for a Mach9 Wine and
  re-stamps the placeholders (see `docs/BUILDING.md`).

### Prefix markers

A builtin loads only if the prefix's `system32` or `syswow64` holds a marker
for its name. `wineboot` writes one for every builtin in `lib/wine` when it
creates a prefix, so a prefix created after the install needs nothing. An
existing prefix takes the stubs below, or a `wineboot -u`, which is a full
prefix update and on a Mach9 Wine stamps markers of the right machine.

| Stub in `prefix-markers/` | Into | Machine | Works for |
| --- | --- | --- | --- |
| `syswow64/mtld3d.dll` | `syswow64` | 32-bit | 32-bit programs |
| `system32-arm64/mtld3d.dll` | `system32` | hybrid (ARM64X) | x64 and native ARM64 programs |

Nothing in `prefix-markers/` belongs in `lib/wine`. CrossOver Preview always
needs the stubs: `mtld3d` never enters its `lib/wine`, so no `wineboot` stamps
a marker.
