# Issue 625 measurement plan

Compare serial native startup compilation against bounded parallel native compilation on copied current-format WoW, Half-Life 2 and GTA IV caches. The assigned source is cdf836dbe97d9db46d3ed33a8d7cda3a7c37fc66.

The standalone native harness uses the repository's core cache reader, regeneration and pipeline-translation logic. Its native shader, pipeline and handle modules are exact copies of the assigned source; the two texture conversion functions are extracted unchanged. It retains one device through all joined workers and holds all library/function handles until dependent pipeline creation completes. Each native operation has the same autorelease-pool boundary as the runtime thunk.

WoW contains 99 deduplicated shaders and 356 recipes, Half-Life 2 contains 117 shaders and 216 recipes, and GTA IV contains 420 shaders and 405 recipes. Each copied corpus has stale programmable MSL, all of which regenerates successfully with the assigned source. Regeneration stays serial for this first experiment.

Run each corpus once under unknown driver-cache conditions, then seven randomized blocks comparing workers 1, 2, 4 and 8 in fresh processes after that priming pass. Do not purge or alter shared Apple driver caches. No cache-cold claim is available. Record total proxy-ready elapsed time separately from library and pipeline wall times and from summed native phase durations. Exclude device creation, which precedes production prewarm, from proxy-ready time. Include file reading, parsing, serial MSL regeneration, scheduling and result assembly. Measure process maximum RSS, CPU and context switches with time -l; compiler service resource costs are outside that process measurement.

Compare the sorted successful library and pipeline identity hashes, counts, dependency skips and no-color sibling mappings for every run. Inject one shader failure in a separate untimed correctness group to verify the same dependent recipes are omitted at every worker count. Worker pools fully join before the next dependency phase and before releasing the startup-ready proxy. Run only in an orchestrator-approved exclusive timing window.

The harness does not execute Wine thunks, install the production encoder payload, persist regenerated cache entries, compact the cache, render or run gameplay. Its result establishes native throughput only. Production cancellation and failed-send cleanup require review and tests if the measurements support an implementation. No game launches, shared SDK mutation, binary archives or background gameplay compilation are in scope.

## Rules check

CONTRIBUTING.md: measurement precedes production changes. The copied caches and all artifacts are under the primary repository's ignored .codex directory. No install-bearing make is run. A production change would require fmt, check, isolated test and conformance before commit, followed by review on the final candidate.

CONVENTIONS.md: no production statics, config keys, environment variables, wire fields, schemas, dependencies, derives, lint suppressions or duplicated helpers are introduced. The temporary measurement program uses existing dependency families and exact native source copies for fidelity. Its logger is process-local instrumentation. No runtime code is changed based on an unmeasured hypothesis.

ARCHITECTURE.md: library-before-pipeline dependencies and complete-before-frame-intake ordering remain explicit. Read-only input loading prevents cache invalidation or compaction from touching the original game files. Separate native phase totals cannot be added as if they were startup elapsed time. The production API, encoder and submit overlap is unchanged.

## Follow-up: novel entry-point identities

The primed matrix measures milliseconds of native work and does not answer the original first-use cost. The approved follow-up gives every process a unique suffix on each shader entry point, preserving the original shader body, compiler options, logical shader records and pipeline recipes. Every substitution is checked by reversing it to the original MSL text before compilation. Four randomized blocks per corpus compare workers 1, 2, 4 and 8, each with a new namespace saved in the run plan. Accepted library and PSO identities are reported in their original logical key space and must agree.

This is a synthetic identity perturbation of real cache contents, not proof of a cold driver cache. Apple may canonicalize semantically identical sources. No shared driver cache is purged. The operation is intended to expose first-use compilation cost under matched source complexity and is labelled separately from the exact-byte primed matrix. A separate failure check selects the first pipeline recipe's vertex shader, ensuring a failed compile actually removes dependent recipes. Rules check is unchanged: this adds only ignored measurement artifacts and no production implementation.
