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
| Halo 2 | Renders, `halo2` profile |
| 3DMark05 | Runs end to end |
| Unigine Tropics | Runs |
| Gunmetal | Starts and benchmarks |

## Reporting a game

A game that fails or renders wrongly is tracked as a
[`game-compat`](https://github.com/athei/mtld3d/labels/game-compat) issue in
the [tracker](https://github.com/athei/mtld3d/issues); reports are welcome.
Name the game, its version and whether it is 32-bit or 64-bit, the mtld3d
release and the Wine or CrossOver version, and attach the log from
`mtld3d-logs` next to the game's executable. A report of wrong rendering is
easiest to act on with an F12 capture taken at the moment it shows
([`ARCHITECTURE.md`](ARCHITECTURE.md#f12-three-frame-dump-and-gpu-capture)
says how).
