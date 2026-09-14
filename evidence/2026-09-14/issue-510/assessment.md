# Issue 510 assessment

Reviewed cdf836dbe97d9db46d3ed33a8d7cda3a7c37fc66 in the assigned issue-510 worktree. No source change, commit, PR, installation, build, GPU workload, or CI dispatch was made. The local host is Apple M5 Max. The investigation remains open for affected-device producer/copy diagnostics.

## Confirmed evidence

PR #522, merged as 99855175e806b3d67547379b49431cbc9c489871, added the resolve-retirement workaround. It is still present. Post-mitigation all-black-row failures were recovered from ordinary main CI on 485f3d5 (September 7, run 34166547160 attempt 1) and 45bad5d (September 10, run 34479501158 attempt 1). The September 7 retry passes the same commit. A complete fix is not established.

The September 10 failing process is 10634. Its preserved stderr line 147 names the test and line 148 contains 640 opaque-black pixels. The corresponding layer log stamps all three loaded libraries as 45bad5d. It has no failed readback or GPU command-buffer failure record. The only backbuffer failure is the deliberate 65535x65535 negative test, not #477's valid request rejected by Metal. A previous optional sRGB-view refusal is at layer-log line 1546.

The same process's system log has three AppleParavirtGPUMetal child-resource failures at lines 10484, 10495 and 10507 (13:09:44.605/.769/.923 UTC), each adjacent to a kernel object-already-exists record. The MSAA device's creation is at 13:09:52. System collection completed with exit 0. No GPU-hang signature was found. These records establish earlier same-process driver errors, not that their unknown resources are the MSAA target. No shared cause with #477 is established.

The current downloadable September 7 layer-log artifact contains only the successful attempt 2 files. The original failure's job log retains the assertion and summary but cannot establish the original producer's completion state. This limitation is explicit in the proposed issue body.

## Source review

- `windows/core/src/passes.rs:4659`: the last pass for each multisampled attachment is assigned its resolve; `note_msaa_read` at 4776 covers a mid-submission read.
- `windows/d3d9/src/device.rs:7504`: StretchRect notes the MSAA read and invokes retirement for a source with an MSAA companion at 7510.
- `windows/d3d9/src/encoder.rs:2618`: draining async submissions waits for their payloads to return. `wait_for_resolve_retire` at 8539 drains first and targets `current_submit_seq - 1`.
- `unix/unix/src/metal/command.rs:220`: retirement waits for the registered command buffer and records failures. Managed readback at 3259 allocates Managed storage, synchronizes it at 3408, waits at 3413 and checks completion at 3414 onward.
- `unix/unix/src/metal/device.rs:138`: the existing Paravirtual device-name predicate sets the retirement cap. Forced Intel configuration leaves this cap alone; `windows/core/src/gpu_caps/tests.rs:79` and 94 pin that contract.
- `docs/ARCHITECTURE.md:346`: the existing command diagnostic target covers the missing render-pass, copy, and completion evidence. No duplicate diagnostic was added.

No demonstrable ordering defect was found in the reviewed path. Source intent cannot prove the failing producer executed correctly. The all-black read alone does not distinguish missing drawing, stale resolve content, or a driver resource failure. Additional unconditional waiting would be speculative.

## Rules check

Investigation only: no statics, configuration keys, wire fields, dependencies, derives, lint suppressions, duplicated functionality, rendering behavior, baseline, or test changes. The assigned worktree remained clean. CONTRIBUTING.md, CONVENTIONS.md, ARCHITECTURE.md and conformance guidance require a fail-before/pass-after proof plus full check/test/conformance for a rendering fix; none is claimed here. Those gates were not run because there is no candidate change and the local GPU cannot establish the reported paravirtual failure. Existing source and retained CI were used instead.

## Deliverables

`issue-510-body.md` is the proposed replacement issue body, not published. `ci-ledger.json` retains run/job/attempt identities and extracted target results. `main-runs.json`, `jobs-*.json`, and `job-*.log` preserve the GitHub read results. `artifacts-34479501158` and `system-34479501158` preserve the September 10 failure. `artifacts-34166547160` preserves exactly what remains downloadable for September 7. `issue-510.json`, `issue-477.json`, and `pr-522.json` capture the current discussion and earlier mitigation description. A final audit summary and process status are recorded separately.
