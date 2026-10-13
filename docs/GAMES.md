# Tested games

The games mtld3d has been run with, and how far each one gets. A profile
named in the table is a built-in set of options the game gets without any
configuration; [`app_profile.rs`](../windows/core/src/app_profile.rs) gives
the reason for each one.

| Game | Status |
| --- | --- |
| World of Warcraft 1.12 | Plays, `wow` profile, the primary target |
| World of Warcraft 3.3.5a | Plays, `wow` profile |
| Half-Life 2 | Plays |
| Team Fortress 2 | Plays, 64-bit, D3D9 renderer (launched without `-vulkan`) |
| Grand Theft Auto IV | Plays, `gta-iv` profile |
| Call of Duty: Modern Warfare 2 | Plays, 64-bit |
| Assassin's Creed II | Plays, with EaglePatch |
| Age of Empires II: HD Edition | Plays |
| Kane & Lynch: Dead Men | Plays |
| LEGO Indiana Jones 2: The Adventure Continues | Plays |
| League of Legends 4.20 | Plays |
| Need for Speed: Underground | Plays, 32-bit, with the Widescreen Fix |
| Need for Speed: Underground 2 | Plays, with the Widescreen Fix, ExtraOptions and XtendedInput |
| Need for Speed: Most Wanted (2005) | Plays, with the Widescreen Fix and ExtraOptions |
| Need for Speed: Carbon | Plays, with the Widescreen Fix |
| NosTale | Plays, 32-bit, windowed and DPI-unaware |
| Halo: Combat Evolved (PC, 2003) | Plays, launched with `WINE_LARGE_ADDRESS_AWARE=0` |
| Halo 2 | Renders, `halo2` profile |
| 3DMark05 | Runs end to end |
| Unigine Tropics | Runs |
| Gunmetal | Starts and benchmarks |

Half-Life 2 and Team Fortress 2 call `Direct3DCreate9Ex`. Since v0.13.0
([#1088](https://github.com/athei/mtld3d/pull/1088)) they run on the D3D9Ex
device, and both rows were verified on it. `-nod3d9ex` on the command line
keeps them on plain D3D9. Every other game in the table that was checked
creates a plain device.

NosTale needs v0.13.0 or later. Before it, its characters drew with other
sprites' images.

NosTale draws its text with GDI into slots sized for 96 DPI, so it has to
run DPI-unaware. On a prefix above 96 DPI, a DPI-aware NosTale draws its
text larger. The slots clip it, and the text overflows its boxes. Do not
give its executable the `HIGHDPIAWARE` flag that
[`INSTALL.md`](../INSTALL.md#fullscreen) recommends for fullscreen, and do
not set that flag as the prefix default with `/ve`.

Play NosTale in a window. Its fullscreen mode keeps a windowed device and
changes the display mode. Under `EmulateModeset` on a prefix above 96 DPI,
its window then comes out larger than the screen.

## Reporting a game

A game that fails or renders wrongly is tracked as a
[`game-compat`](https://github.com/athei/mtld3d/labels/game-compat) issue in
the [tracker](https://github.com/athei/mtld3d/issues); reports are welcome.
Name the game, its version and whether it is 32-bit or 64-bit, the mtld3d
release and the Wine or CrossOver version, and attach the log from
`mtld3d-logs` next to the game's executable. A report of wrong rendering is
easiest to act on with a Ctrl+Shift+P capture taken at the moment it
shows
([`ARCHITECTURE.md`](ARCHITECTURE.md#ctrlshiftp-three-frame-dump-and-gpu-capture)
says how).
