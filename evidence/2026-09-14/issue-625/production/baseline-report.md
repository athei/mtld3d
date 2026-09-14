# Production serial baseline

The PE probe loads the isolated i686 production DLL and x86_64 Unix library through Wine/Rosetta. Source is cdf836dbe97d9db46d3ed33a8d7cda3a7c37fc66 with an unreferenced startup_work core helper present; shader_prewarm and encoder runtime source remain unchanged. Installed and build-output binary hashes are retained in baseline-stamp.json. The d3d9 install step changes the build-output file, so its installed hash is recorded separately.

Each of 18 fresh processes uses an independent copied cache. Exact inputs are original copied game caches with stale programmable MSL; refresh append and compaction are included in the runtime startup log interval. Novel inputs regenerate the same MSL bodies and assign reversible new entry names and matching recipe references, with current emitter fingerprints and no DXSO. They expose first use of new source identities, not guaranteed cold driver caches. No shared cache was purged.

The startup interval is the production prewarm body from loading through completed native shader/PSO creation, sibling resolution and compaction, immediately before payload transfer. First readback measures CreateDevice through completed readback and includes the encoder barrier plus additional creation/render/readback overhead. Both pixels match, a second clear/readback leaves the cache unchanged, and final device release returns zero in every accepted sample. This clear-only probe does not exercise the game shaders as draws. The two existing pipeline-cache replay end-to-end tests passed separately.

| Corpus | Exact samples, startup ms | Novel samples, startup ms | Pipelines / sibling mappings |
|---|---|---|---|
| wow | 1621, 223, 222 | 1521, 1531, 1535 | 356 / 97 |
| hl2 | 2357, 292, 294 | 2206, 2225, 2206 | 216 / 0 |
| gtaiv | 9284, 848, 849 | 8713, 8732, 8746 | 405 / 6 |

The first exact process has unknown driver state and is reported separately from subsequent warmed processes. The source-identity sequence is serial-baseline first and candidate later, so any later comparison must retain the temporal ordering limitation. Individual process resource costs and first-readback intervals are in baseline-rows.jsonl, with raw logs per run. Compiler service resources and concurrent multi-device resource scaling are not measured.

One initial invocation exited before DLL load because WINEMSYNC did not match the already-running isolated server. That setup failure is retained as excluded-msync output; all accepted samples explicitly use WINEMSYNC=1.
