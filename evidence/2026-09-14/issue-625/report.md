# Startup prewarm measurements

Measurements on Apple M5 Max (18 physical/logical CPU cores, 64 GiB), macOS 27.0 build 26A428, from source cdf836dbe97d9db46d3ed33a8d7cda3a7c37fc66. All runs used a native optimized standalone harness, with no game or Wine launch. The orchestrator held unrelated builds/tests and GPU inspection for both measurement windows.

This report records the native experiment before production edits. It justified the bounded prototype; subsequent production measurements and final verification are recorded in production/report.md. The native proxy alone does not establish a production startup improvement, gameplay behavior, rendering equivalence or cancellation correctness.

## Workloads and method

The copied current-format caches contain 99 shader records and 356 pipeline recipes for WoW, 117/216 for Half-Life 2 and 420/405 for GTA IV after the core reader's validation and deduplication. These are the usable records under this build, not all records ever seen by the games. All accepted programmable entries had stale MSL and regenerated successfully from retained DXSO. Fixed-function stale records and dependent recipes are filtered by the production reader before these counts. All snapshots indicate compaction is needed.

The input hashes and exact native source-copy hashes are in manifest.json. Core cache parsing, MSL regeneration, pipeline snapshots and translation come directly from the assigned worktree. Native shader/pipeline/handle helpers are unchanged copies; two texture conversion helpers are extracted unchanged. The harness keeps a canonical device retain, completes all shader workers before any PSO job, deduplicates runtime pipeline keys before scheduling, joins all workers before assembling the ready payload, and releases all retained objects after the run. Every compile uses a per-call autorelease pool, matching the runtime thunk wrapper.

Proxy-ready elapsed time starts after native device creation and includes cache read/parse, serial MSL regeneration, worker scheduling, both compilation phases and identity/sibling assembly. It excludes persistent append/compaction, Wine thunk calls, production encoder installation and any frame submission. The production barrier includes those excluded operations. Native phase totals sum durations spent inside concurrent calls and are not elapsed startup time.

Workers 1 runs directly in the coordinator; workers 2, 4 and 8 use joined scoped native threads and an atomic work index. Each process uses a new device handle. The scratch harness is an experiment, not a proposed production thread-pool implementation.

## Exact-input primed driver condition

Seven randomized blocks per corpus, with each block running all four worker counts in randomized order after a serial priming pass. Input bytes and emitted source identities are identical across worker counts. No Apple cache was purged. Medians and observed ranges below are milliseconds to proxy-ready.

| Corpus | Serial | 2 workers | 4 workers | 8 workers |
| --- | ---: | ---: | ---: | ---: |
| wow | 9.281 (9.228-9.404) | 8.829 (8.775-9.590) | 7.669 (7.619-7.764) | 7.351 (7.236-7.653) |
| hl2 | 12.077 (12.014-12.589) | 10.845 (10.673-10.924) | 9.878 (9.727-10.036) | 9.823 (9.630-10.199) |
| gtaiv | 49.172 (48.784-49.401) | 44.194 (43.820-44.841) | 41.861 (41.571-42.729) | 41.579 (40.911-42.313) |

Four workers save approximately 1.6 ms, 2.2 ms and 7.3 ms respectively. Eight workers add little in this condition. Library compilation wall time is roughly flat while PSO wall time falls; summed call times and context switches rise. These warm savings alone would not justify adding startup concurrency.

The initial non-comparison passes measured approximately 2.059 seconds for HL2 and 8.198 seconds for GTA IV. Their driver-cache state is unknown, so they cannot be used as serial baselines against the later primed runs. The first WoW pilot is excluded: a scratch logger compile error left an older binary with disabled native counters, and the runner rejected its zero phase totals. The failed build, pilot and subsequent corrected build logs are retained. Accepted comparisons all require positive phase counters. The corrected WoW priming pass followed that pilot and is not cold evidence.

## Novel entry-point identities

Four randomized blocks per corpus, each running workers 1, 2, 4 and 8 with a unique entry-point namespace saved in run-plan.json. Only entry-point names change; shader bodies, compiler options, logical records and pipeline recipes are identical. The harness reverses each replacement and asserts it reproduces the original MSL before compiling. This is a synthetic identity perturbation of actual game cache contents. It is not a guaranteed cold-driver experiment, because the driver may canonicalize equivalent sources. All accepted logical key sets are compared in the original namespace.

| Corpus | Serial | 2 workers | 4 workers | 8 workers |
| --- | ---: | ---: | ---: | ---: |
| wow | 1402.4 (1355.4-1442.1) | 857.5 (828.2-871.6) | 505.9 (499.5-511.1) | 325.3 (323.0-328.1) |
| hl2 | 2045.5 (2025.7-2117.4) | 1136.4 (1133.8-1149.3) | 642.0 (638.6-645.0) | 406.4 (403.3-408.1) |
| gtaiv | 8235.3 (8123.6-8764.3) | 4486.6 (4391.5-4525.8) | 2443.0 (2414.0-2524.4) | 1438.8 (1434.0-1448.8) |

Four workers reduce the median proxy interval by approximately 64%, 69% and 70%; eight reduce it by 77%, 80% and 83%. These percentages describe this native experiment, not game startup or frame rate. Serial and four-worker observed ranges do not overlap for any corpus. Both library and PSO wall phases benefit.

| Corpus / workers | Library wall ms | PSO wall ms | Sum library-call ms | Sum PSO-build ms | Median process RSS MiB | Involuntary switches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wow / 1 | 625.5 | 773.9 | 624.7 | 772.5 | 29.82 | 1272.5 |
| wow / 2 | 350.6 | 504.0 | 695.7 | 1005.4 | 29.76 | 1425.0 |
| wow / 4 | 198.6 | 304.3 | 780.6 | 1213.8 | 30.63 | 1577.5 |
| wow / 8 | 131.3 | 191.4 | 1015.6 | 1500.5 | 31.20 | 2072.0 |
| hl2 / 1 | 1074.2 | 966.2 | 1073.1 | 964.9 | 29.75 | 1567.0 |
| hl2 / 2 | 589.2 | 544.5 | 1173.9 | 1084.6 | 30.23 | 1657.5 |
| hl2 / 4 | 330.0 | 306.8 | 1292.0 | 1221.3 | 30.59 | 1807.0 |
| hl2 / 8 | 207.4 | 193.2 | 1589.7 | 1484.1 | 31.35 | 2346.5 |
| gtaiv / 1 | 4926.2 | 3279.1 | 4921.5 | 3276.1 | 56.52 | 5016.5 |
| gtaiv / 2 | 2677.4 | 1774.0 | 5335.0 | 3540.9 | 57.60 | 5112.0 |
| gtaiv / 4 | 1441.5 | 971.5 | 5739.0 | 3846.3 | 59.18 | 5387.5 |
| gtaiv / 8 | 850.6 | 559.8 | 6722.4 | 4386.8 | 60.06 | 6772.5 |

The larger call-duration sums are consistent with overlap and waiting/competition inside native compilation; they do not establish a particular driver lock or CPU bottleneck. Process CPU, RSS, footprint, page faults and context switches come from /usr/bin/time -l in each raw stderr. Driver compiler-service CPU and memory are outside those per-process measurements, so the table does not bound total machine resource cost. No thread count is yet selected for production, and these measurements cover one Apple-family device, not Intel/AMD.

## Accepted work and failure behavior

All 84 exact-input primed runs and all 48 novel-identity runs produced identical successful logical library/PSO sets for their corpus. Counts are WoW 99/356 with 97 no-color sibling mappings, HL2 117/216 with zero mappings and GTA IV 420/405 with six mappings. No dependency skips, regeneration failures, library failures or PSO failures occurred in these successful groups. The stored key hashes identify accepted work, not rendered pixels.

A separate failure group replaces the vertex shader referenced by the first WoW recipe with invalid MSL. All four worker counts produce 98 libraries and 317 PSOs, omit the same 39 dependent recipes, and agree on key hashes. Native objects are released after joined workers. An earlier check selected shader index zero and removed no recipe; that nondiscriminating check is retained but is not the evidence for dependency failure handling. Neither check validates production device cancellation or failed channel-send cleanup.

## Remaining work before a production change

A production prototype must preserve the one-shot encoder barrier, deduplication, library-before-PSO ordering, device lifetime, cancellation between admitted jobs, partial failures and orderly handle cleanup. It must keep regenerated append order before compaction and prevent cache writes from racing the encoder's writer. Worker-creation failure needs a defined fallback; no compiler pool may outlive prewarm cancellation. These are implementation obligations, not validated outcomes of this native proxy.

The full production interval should then be measured through an isolated non-game harness using copied caches, and successful prewarmed combinations checked for identical rendering and no additional live compilation. The final source candidate needs make fmt, make check, make ISOLATED=1 test and make ISOLATED=1 conformance. No such gates were run for this artifact-only investigation, and there is no commit or PR candidate.

## Rules check

The work follows CONTRIBUTING.md by measuring before source edits and preserving raw outcomes, ranges and exclusions. All work is inside ignored repository-local evidence and scratch directories. No shared SDK or original game cache was modified.

Under CONVENTIONS.md this introduces no production static, config key, environment variable, wire field, cache schema, dependency, derive, lint suppression or duplicate helper. Exact native copies and a process-local logger exist only in the standalone scratch experiment; their unused-item warnings do not describe a green repository gate. ARCHITECTURE.md's API/encoder/submit paths and the actual startup barrier remain unchanged. The measured native phase sums are kept separate from proxy wall time.

## Separate source finding

On thread-creation failure, shader_prewarm::spawn discards the error through .ok() without sending an empty payload. EncoderThread retains another prewarm_tx sender, so encoder_thread_main's blocking receive cannot observe channel closure and remains before frame/shutdown intake. This is a conditional source-derived deadlock, not a reproduced failure and not a performance measurement. It was handed to the orchestrator for independent review and duplicate checking; no fix is included here.

## Evidence and processes

- matrix-20260914-073503: accepted primed results, individual stdout/stderr, environment and process inventories.
- novel-20260914-073819: novel-identity results, persisted randomized run plan, individual stdout/stderr and referenced failure results.
- summary.json: medians, ranges, phase totals and resource measurements.
- manifest.json: input and copied native source hashes.
- plan.md: reviewed experiment boundaries and Rules check.
- harness-source: final scratch program and runners, including the added novel-identity path. The accepted primed program predates that path; its initial build logs and result schema identify the earlier configuration.

All started build and measurement processes have exited. No process is transferred or left running. The assigned repository worktree is clean at its original commit.
