# Bounded production startup prewarm

Eight bounded callers reduce the measured production prewarm interval on these workloads. Four randomized baseline/candidate pairs per corpus give median reductions of 70.1% for WoW, 74.6% for Half-Life 2 and 79.7% for GTA IV. This is a synthetic first-use source-identity experiment using real cache bodies and recipes, not a guaranteed cold driver cache or a gameplay/FPS measurement.

## Method and build identity

Apple M5 Max, 18 CPU cores, 64 GiB, macOS 27.0 build 26A428. Both arms use optimized PERF=1 PROD=1 i686 PE DLLs with the x86_64 Unix library under Wine/Rosetta. Baseline source is cdf836dbe97d9db46d3ed33a8d7cda3a7c37fc66 with an unreferenced helper module; candidate is 3f0f1fbb8ceeb7cf1a1a49980aa0ce3fb3bb1a32 plus the retained candidate-source.diff. The dependency delta includes the separately merged query changes and prewarm spawn-failure recovery; it is retained as dependency-source.diff. The latter moves initial thread startup ownership but leaves the serial compilation body unchanged. Installed binaries, source identities, exact commands and hashes are in baseline-stamp.json and candidate-stamp.json; both saved installed binary sets are retained. The same release-built probe executable is used in every arm.

Every process owns an independent cache copy beside its probe executable. Exact inputs retain original bytes and stale programmable MSL; regeneration appends and compaction are included. Novel inputs have semantically identical regenerated shader bodies with unique derived entry names, matching rewritten recipe references, current emitter fingerprints and no DXSO. Each rename is reversed and checked against the original body during preparation. Immutable input.bin copies and namespaces are retained. No original game cache or shared driver cache is modified. The driver may canonicalize equivalent sources, so unique names do not prove a cold cache.

The first matrix ran 18 serial samples before implementation and 18 candidate samples afterwards. The stronger follow-up alternates saved baseline/candidate binaries with randomized arm order in four paired blocks per corpus (24 processes); every process has a fresh namespace. All native work and prior builds/tests had ended before each exclusive measurement window. A separate two-process referenced-shader failure check is excluded from timing comparisons. Only the owned private SDK was switched between saved binaries, then restored to the candidate.

The production startup log measures prewarm loading, regeneration/persistence, library and PSO completion, sibling mapping and compaction immediately before payload transfer. First readback measures CreateDevice through the encoder startup barrier and completed readback, including other creation/render/readback costs. Two clear colors and final release are checked in every process; the second readback must leave cache bytes unchanged. This probe does not draw the imported game shaders. Existing pipeline replay end-to-end tests separately verify healthy warm shader/PSO rendering and no appended warm-hit records.

## Alternating first-use source identities

All intervals are milliseconds. Ranges show the four individual samples, not confidence intervals.

| Corpus | Serial prewarm median (range) | Bounded prewarm median (range) | Reduction | First readback, serial / bounded |
|---|---:|---:|---:|---:|
| wow | 1565.0 (1543-1570) | 468.5 (466-471) | 70.1% | 1908.5 / 810.1 |
| hl2 | 2248.5 (2238-2262) | 570.0 (566-572) | 74.6% | 2594.0 / 913.1 |
| gtaiv | 8833.5 (8786-8913) | 1791.0 (1785-1807) | 79.7% | 9177.3 / 2137.2 |

The warmed exact-input observations are much smaller and were not interleaved: using samples 1 and 2 from each arm, median prewarm intervals are 222.5 / 218.0 ms (WoW), 293.0 / 288.0 ms (HL2), and 848.5 / 819.0 ms (GTA IV). The first serial exact sample had unknown driver state and took 1621, 2357 and 9284 ms respectively; those values are not compared to later warmed candidate samples. Individual results and resources remain in baseline-rows.jsonl and candidate-rows.jsonl.

## Cost and accepted results

Whole-process time -l resource medians below exclude compiler-service processes and concurrent multi-device scaling. The CPU values are process user/system time, not the sum of native call latencies. Involuntary context switches increase with the extra callers.

| Corpus / arm | Max RSS MiB | User / system seconds | Involuntary switches | Sum Metal library call ms | Sum Metal PSO-build call ms |
|---|---:|---:|---:|---:|---:|
| wow / baseline | 146.08 | 0.330 / 0.115 | 2452.5 | 640.6 | 781.5 |
| wow / candidate | 148.46 | 0.350 / 0.130 | 3278.0 | 1029.3 | 1530.3 |
| hl2 / baseline | 147.83 | 0.370 / 0.120 | 2758.5 | 1097.4 | 981.7 |
| hl2 / candidate | 150.54 | 0.390 / 0.130 | 3474.5 | 1593.0 | 1487.5 |
| gtaiv / baseline | 177.11 | 0.690 / 0.190 | 6172.0 | 5079.6 | 3381.0 |
| gtaiv / candidate | 179.72 | 0.745 / 0.215 | 7617.0 | 6725.0 | 4493.3 |

Summed native call latencies rise while elapsed startup falls. These durations include waiting inside concurrent native calls and do not isolate driver locks or CPU work. PSO parent totals and their native subtimers must not be added together. The full phase totals, call/failure counts, process resource outputs and logs are retained per process. Per-phase wall times were measured in the earlier native proxy; the production code retains its existing total-startup elapsed timer rather than adding permanent instrumentation.

| Corpus | Successful libraries | Successful PSOs | No-color mappings |
|---|---:|---:|---:|
| WoW | 99 | 356 | 97 |
| Half-Life 2 | 117 | 216 | 0 |
| GTA IV | 420 | 405 | 6 |

These counts match in every one of the 60 healthy production processes (18 baseline, 18 candidate, 24 alternating). Native phase failure counts are zero. Because every usable library/recipe succeeds, the accepted set is the full prepared input set; the earlier native matrix also compared explicit canonical successful-key hashes. The separate PE failure fixtures replace the first recipe's VS with invalid MSL. Both arms retain 98 libraries, 317 PSOs and 85 mappings, omit the same 39 dependent recipes, release the startup barrier, render both clear checks and keep the post-startup cache stable. Failure logs are retained separately and excluded from timing summaries.

## Implementation and verification

The coordinator and at most seven temporary workers claim jobs from one atomic index. Available CPU count and batch length reduce the cap on smaller hosts. Each device owns its batches and cancellation state. Failed worker creation reduces concurrency while the coordinator and admitted workers finish remaining jobs. Cancellation stops admission, waits for admitted work, and preserves completed result ownership. Scoped handles are dropped; installed Rust scoped-thread source confirms completion uses its internal counter/park path, avoiding the explicit Win32 thread-handle join that Wine can invalidate.

MSL regeneration and appends remain serial. All library results are installed before dependent PSO resolution. PSOs are admitted once per resolved runtime key, including a failed attempt; unlike the old success-map check, duplicate recipes do not retry that key within the same startup. The runtime key already carries functions, declaration identity, stream layout and normalized pipeline state. Retained disk recipes and gameplay misses may retry later. Mapping assembly and compaction follow the completed native batches, and cancellation skips compaction. The existing encoder startup barrier and gameplay compilation paths are unchanged.

make fmt and make check passed. One initial check found a needless borrow in the new scoped spawner; it was fixed before the green check. Full make ISOLATED=1 PERF=1 PROD=1 test passed: 1222 core/workspace tests, 404 Unix tests, and 575 reported end-to-end tests on each PE architecture, with zero failed or not-run tests. The six new deterministic scheduling tests cover exactly-once/order, cap, first and partial worker rejection, pre-cancellation, in-flight cancellation with an independent device, and ownership release. Failure-timeout waits and release-on-drop cleanup prevent admission regressions from hanging the cancellation test. Both existing cache replay tests pass. Full matched-profile conformance passes for i686 and x86_64 with no regressions against the repository baselines; expected conformance failures remain classified rather than counted as all tests passing.

No game launch, gameplay FPS measurement, imported-shader draw equivalence, driver-service resource accounting or concurrent-device performance result is claimed. The local evidence supports the fixed eight-caller cap; it does not prove the optimal cap on every platform. Source and API behavior outside startup prewarm remain out of scope.

Conformance counters: i686 device 735 classified failures against ceiling 758, visual 221/221, stateblock 0/0 and d3d9ex 0/0; x86_64 device 733/756, visual 219/219, stateblock 0/0 and d3d9ex 0/0. Every group has zero crashes and an accepted comparison.

Validated source commit: a9462ce831c0d86eee9ab49fd8640b6141bbbf64. The worktree is clean. Measurements used the identical source before commit, with the pre-commit installed binary hashes retained; no source changed after the green gates.
