# Issue-loop evidence, September 14, 2026

Published evidence for issues [54](issue-54/RESUME.md), [510](issue-510/RESUME.md), [625](issue-625/RESUME.md), [638](issue-638/RESUME.md) and [673](issue-673/RESUME.md). Issue477 shares the limited cross-issue evidence in510; no common cause was established. Each handoff describes the result, limitations and next useful action.

This branch preserves evidence only and is not intended to merge into main. Issue comments link the immutable evidence commit. No CI run or release is needed for this archive. Main runtime code is unchanged by this evidence commit.

Each issue directory contains readable reports, a handoff, checksums and a compressed raw-evidence archive. The HDR archive includes the complete GPU trace and pixel outputs and is split into32MiB files. Use fetch_evidence.py with the full pinned commit SHA to download, assemble, verify and extract into a new directory. Python3.9 or later with standard-library lzma is sufficient.

Game caches/shader inputs, rebuildable binaries and unrelated local process inventories are omitted with explicit per-file hashes and reasons. No original retained local evidence was deleted. Reproduction requirements for omitted inputs are in the corresponding handoff.

ci-observations.json retains exact merge SHAs, runs, attempts and all job outcomes. At the publication refresh, PR674 and PR676 main CI passed; PR675 passed unchanged on attempt2 after an attempt1 GPU-hang no-verdict. These observations do not turn a failed or truncated attempt into a passing result.
