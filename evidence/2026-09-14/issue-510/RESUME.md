# Issue 510 evidence handoff

Open, awaiting affected-device evidence. Review assessment.md, issue-510-body.md, ci-ledger.json and audit-summary.txt. The body was published during the loop despite the historical worker assessment saying proposed/unpublished.

97 ordinary main runs and 116 Intel i686 job attempts were audited. Two all-black-row failures remained after #522: run34166547160 attempt1 at485f3d5 and run34479501158 attempt1 at45bad5d. The first passed unchanged on retry; its currently downloadable layer artifact only contains the successful retry. The latter retains the original failing process10634 layer/stderr/system evidence. Three earlier same-process paravirtual child-resource errors do not identify the MSAA resource or establish a shared cause with #477.

Next useful evidence is an affected-device recurrence with the existing command diagnostic target from docs/ARCHITECTURE.md enabled, capturing render-pass/resolve placement, copy/readback completion, resource identity and system GPU errors for the same PID/build. No demonstrated ordering defect was found. Do not add unconditional waits or dispatch speculative CI based only on the black row. This investigation ran no new GPU workload or local rendering gate.

The archive contains retained source, reports and raw evidence, including unsuccessful attempts where they informed the conclusion. Historical reports preserve their original investigation-time wording. This handoff records the final outcome. Local paths in logs and stamps are provenance, not downloadable links. Worktrees named in those paths have been cleaned; create a new named worktree and new evidence directory when reproducing.

Download the archive with the sibling `fetch_evidence.py`, the full commit SHA from the issue comment, this issue number and a new output directory:

```sh
python3 fetch_evidence.py <EVIDENCE_COMMIT_SHA> 510 /absolute/new/output
```

The downloader verifies the complete archive hash before extraction and every published payload hash afterwards. `ARCHIVE.json` records archive size, hash and ordered 32 MiB chunks; `ARCHIVE-SHA256SUMS` allows independent chunk verification. `PUBLIC-SHA256SUMS` covers the exported payload. `WITHHELD.json` records any omitted file with its size, SHA256 and reason. Original manifests are retained as provenance and may also name withheld files.
