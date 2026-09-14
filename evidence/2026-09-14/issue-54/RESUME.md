# Issue 54 evidence handoff

Closed as not planned after maintainer review. Retain the current HDR shader. Hardware sRGB saved 5-11% of the measured pass but changed near-black precision and unequal-extent filtering. Tested half-curve savings were negligible; half-PQ was slower with large error. Do not resume this optimization without new evidence or an explicit changed accuracy/routing decision.

Read REPORT.md and REPRODUCE.md. The archive contains the complete hdr-five-variants.gputrace directory, gpudebug profiles, compiler/resource inspection, extracted PNGs/heatmap, all retained pixel outputs, both timing runs, analysis, Swift probe and five Metal source variants. The trace is synthetic, with no game assets. Two unrelated full-host process inventories were withheld.

For a fresh exact-variant reproduction, copy Probe.swift and the five .metal sources into a new evidence directory, build there with `xcrun swiftc -O Probe.swift -o probe`, and follow the correctness/timing/capture commands in REPRODUCE.md. The archived variants are already generated, so the generator step is unnecessary. To regenerate variants against repository source instead, retarget the two paths in generate_variants.py to a new worktree and output directory. Keep outputs beside the copied probe source because it resolves #filePath. Never rerun into the archive. Capture/profile overhead is excluded from reported uncaptured timings.

The archive contains retained source, reports and raw evidence, including unsuccessful attempts where they informed the conclusion. Historical reports preserve their original investigation-time wording. This handoff records the final outcome. Local paths in logs and stamps are provenance, not downloadable links. Worktrees named in those paths have been cleaned; create a new named worktree and new evidence directory when reproducing.

Download the archive with the sibling `fetch_evidence.py`, the full commit SHA from the issue comment, this issue number and a new output directory:

```sh
python3 fetch_evidence.py <EVIDENCE_COMMIT_SHA> 54 /absolute/new/output
```

The downloader verifies the complete archive hash before extraction and every published payload hash afterwards. `ARCHIVE.json` records archive size, hash and ordered 32 MiB chunks; `ARCHIVE-SHA256SUMS` allows independent chunk verification. `PUBLIC-SHA256SUMS` covers the exported payload. `WITHHELD.json` records any omitted file with its size, SHA256 and reason. Original manifests are retained as provenance and may also name withheld files.
