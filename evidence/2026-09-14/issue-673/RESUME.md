# Issue 673 evidence handoff

Completed by PR #675, merge3f0f1fbb8ceeb7cf1a1a49980aa0ce3fb3bb1a32. The worker owns the sole prewarm-result sender, so a rejected spawn releases encoder startup. Recovery disables persistent writes because no cache file was validated. Success ordering and cancellation-before-cleanup remain intact.

Read review.md and gate-results.txt. Four host tests exercise rejected spawn, startup ordering, cancellation and None versus validated empty payload. The retained-sender red proof is a mutation of the new helper restoring the old ownership mechanism, not execution on unmodified baseline. Mutation patch and red/green logs are retained. Worker and final check/test/conformance passed. Allocator probes' exit5-after-reported-results notes were accepted by the existing runner; this is not a general waiver for nonzero exits.

Main CI attempt1 run34791459053 had one Intel x86_64 visual leg report kIOAccelCommandBufferCallbackErrorHang and stop without a verdict. Exact job log and raw artifact are in ci/. Attempt2 passed unchanged on a fresh runner. This observed transition does not prove a source fix for the GPU hang. Full per-attempt job records are in the sibling ci-observations.json.

The archive contains retained source, reports and raw evidence, including unsuccessful attempts where they informed the conclusion. Historical reports preserve their original investigation-time wording. This handoff records the final outcome. Local paths in logs and stamps are provenance, not downloadable links. Worktrees named in those paths have been cleaned; create a new named worktree and new evidence directory when reproducing.

Download the archive with the sibling `fetch_evidence.py`, the full commit SHA from the issue comment, this issue number and a new output directory:

```sh
python3 fetch_evidence.py <EVIDENCE_COMMIT_SHA> 673 /absolute/new/output
```

The downloader verifies the complete archive hash before extraction and every published payload hash afterwards. `ARCHIVE.json` records archive size, hash and ordered 32 MiB chunks; `ARCHIVE-SHA256SUMS` allows independent chunk verification. `PUBLIC-SHA256SUMS` covers the exported payload. `WITHHELD.json` records any omitted file with its size, SHA256 and reason. Original manifests are retained as provenance and may also name withheld files.
