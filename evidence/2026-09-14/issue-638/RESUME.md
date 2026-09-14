# Issue 638 evidence handoff

Completed by PR #674, merge1809d4a0ba65b743b2b85a617234af1021e02b0c. Comments and documentation now explain that Metal GPU hazard tracking cannot protect CPU writes into backing still used by queued draws when query.flushImmediate reports completion early. Runtime behavior is unchanged. The shipped WoW profile enables this option; measured loading benefit does not prove unsafe CPU reuse cannot occur.

Read source-verification.md and gate-status.md; full worker and final check/test logs are archived. Conformance was not required for the comment-only diff. Source commit026dd5c2 was reviewed and locally gated before merge. Main CI passed. No new live-game safety claim is made.

The archive contains retained source, reports and raw evidence, including unsuccessful attempts where they informed the conclusion. Historical reports preserve their original investigation-time wording. This handoff records the final outcome. Local paths in logs and stamps are provenance, not downloadable links. Worktrees named in those paths have been cleaned; create a new named worktree and new evidence directory when reproducing.

Download the archive with the sibling `fetch_evidence.py`, the full commit SHA from the issue comment, this issue number and a new output directory:

```sh
python3 fetch_evidence.py <EVIDENCE_COMMIT_SHA> 638 /absolute/new/output
```

The downloader verifies the complete archive hash before extraction and every published payload hash afterwards. `ARCHIVE.json` records archive size, hash and ordered 32 MiB chunks; `ARCHIVE-SHA256SUMS` allows independent chunk verification. `PUBLIC-SHA256SUMS` covers the exported payload. `WITHHELD.json` records any omitted file with its size, SHA256 and reason. Original manifests are retained as provenance and may also name withheld files.
