# Issue 625 evidence handoff

Completed by PR #676, merge57467572dc51c0c64e69519d1f225936385d0d54. Read production/report.md first, then report.md for the native proxy and production-plan.md for the design. Startup remains a complete barrier before frame intake. Four alternating pairs per corpus measured serial/bounded medians1565/468.5ms (WoW),2248.5/570ms (HL2),8833.5/1791ms (GTA IV). These are novel source identities, not guaranteed cold driver caches or game FPS.

The archive includes all public timing rows/stdout/stderr, failure fixtures' logs, baseline/candidate stamps and source/dependency diffs, six-test scheduling evidence, optimized gates, final standard gates, native harness and PE probe/generator sources. All60 healthy PE processes matched counts/readbacks/cache stability; both failed-shader runs omitted the same39 dependent PSOs. Existing replay e2e tests cover warm shader/PSO rendering; the clear/readback probe does not draw the imported game shaders. Compiler-service resources, simultaneous-device scaling and other hardware remain unmeasured.

Copied game caches and generated shader-input derivatives are not redistributed. Their paths, sizes and hashes are in WITHHELD.json; original source copies remain under the primary repository's .codex/evidence/issue-loop-20260914/agent-625/inputs. A later run needs locally held matching caches to reproduce exact counts, or must label replacement inputs as a new experiment. Rebuildable binaries are also omitted; installed/build hashes remain in stamps. Five unrelated host process inventories are omitted.

Reproduction source is in reproduction-source/, with the original native snapshot additionally in harness-source/. Retarget absolute Cargo path dependencies to a named repository worktree, and script constants to a new harness and evidence directory. Rebuild optimized PERF=1 PROD=1 binaries for both arms, verify installed hashes (winebuild alters the d3d9 DOS stub), and stop on build failure before running any probe. The original serial code is cdf836d; the candidate commit is a9462ce on3f0f1fb, now merged as5746757. The separately landed dependency delta is recorded. Use the native harness with `<cache> <workers> [failure-selector] [namespace]`; prepare rewrites matching entry identities for production fixtures. Match run-plan.json and alternating-plan.json where present, preserve fresh namespaces and randomized order, and allocate an exclusive local timing window. No cache purge or game launch is required.

The archive contains retained source, reports and raw evidence, including unsuccessful attempts where they informed the conclusion. Historical reports preserve their original investigation-time wording. This handoff records the final outcome. Local paths in logs and stamps are provenance, not downloadable links. Worktrees named in those paths have been cleaned; create a new named worktree and new evidence directory when reproducing.

Download the archive with the sibling `fetch_evidence.py`, the full commit SHA from the issue comment, this issue number and a new output directory:

```sh
python3 fetch_evidence.py <EVIDENCE_COMMIT_SHA> 625 /absolute/new/output
```

The downloader verifies the complete archive hash before extraction and every published payload hash afterwards. `ARCHIVE.json` records archive size, hash and ordered 32 MiB chunks; `ARCHIVE-SHA256SUMS` allows independent chunk verification. `PUBLIC-SHA256SUMS` covers the exported payload. `WITHHELD.json` records any omitted file with its size, SHA256 and reason. Original manifests are retained as provenance and may also name withheld files.
