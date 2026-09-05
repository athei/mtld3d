# Contributing

Contributions are welcome, written by hand or with an agent. This file is the
operating manual: what to read before changing anything, which gates have to be
green, how to read their output, and what makes a pull request land on the first
review instead of the third.

It states no rule twice. Every rule has exactly one home, and this file points
at it.

## Read these first

| File | What it owns |
| --- | --- |
| [`README.md`](README.md) | The goal, the requirements, what plays, and where everything else lives. |
| [`docs/STATUS.md`](docs/STATUS.md) | What is implemented, what is not yet, what never will be, and the divergences kept on purpose. |
| [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) | Every code rule: module layout, visibility, data-structure discipline, unsafe discipline, doc-comment shape, dependencies. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The boundary contract: thunk versus command, stable backing for pointers the unix side dereferences, typed wire values, labelling Metal objects, the threading model, the perf counters and how to read them. |
| [`unix/conformance/CONFORMANCE.md`](unix/conformance/CONFORMANCE.md) | The conformance suite: how the runner and the baseline work, what each classification means, and the current per-site audit with its rationales. |
| [`windows/tests/COVERAGE.md`](windows/tests/COVERAGE.md) | What the end-to-end suite covers, and the stubs whose contract a test pins on purpose. |
| [`mtld3d.conf`](mtld3d.conf) | Every runtime option, its default, and a short why. |

Read the conformance document before changing anything in the render, state or
shader-emission path. It is where the reasoning behind the current behaviour
lives, and a change that looks like a fix is often a divergence that was
measured and kept.

## The gates

Two commands, both green before you commit:

- **`make check`** is `cargo fmt --check`, clippy with `nursery` and `pedantic`,
  `make audit`, and `make doc`. Only the check legs deny warnings, so a plain
  `cargo clippy` in an editor reports without failing. Each audit finding names
  the section of `docs/CONVENTIONS.md` behind it; read that section rather than
  pattern-matching your way past the grep.
- **`make test`** is the host-native unit tests plus the end-to-end suite under
  Wine, one leg per PE architecture. Conformance is not part of it, on purpose:
  many of its checks fail by design, so it gates on a regression against a
  baseline instead of on zero failures.

`make fmt` uses nightly rustfmt. If a toolchain bump reformats files you never
touched, that churn is its own pull request, not a hand-revert and not a passenger
in yours.

Every test leg installs the build into the Wine tree `WINE_SDK` names (and into
`WINE_INSTALL_DIR` when set) before it runs, so two checkouts testing at once
overwrite each other's `d3d9.dll` and `mtld3d.so`, and a game launched from that
tree meanwhile runs whichever build landed last. `make test ISOLATED=1` avoids
both: it clones the SDK and the ambient prefix once into `.wine-isolated/`
inside the checkout (APFS clones, so neither costs space or a prefix boot) and
points the tools, the install and the prefix at the clones. Use it whenever
another worktree may be testing or a game is running; a plain `make install`
still targets the shared trees on purpose, since that is how the game gets a
build. The clones and the persistent wineserver of the private prefix stay
behind for the next run; `make clean-isolated` takes them down, and
`make clean-isolated-all` does so for every worktree of the repository. A failing run
whose log shows a `d3d9.dll v` stamp that is not your checkout's is that
collision, not a regression.

Both agent runners configured in this tree already print the conventions digest
at session start and run `scripts/audit.sh --file` after every edit, so a
violation surfaces while you write rather than at commit time. The digest at
`.claude/conventions-digest.md` is generated: regenerate it in the same change
that touches `docs/CONVENTIONS.md`.

## Reading a test run

The end-to-end suite is three test binaries per architecture, and the runner
in `unix/e2e` runs each one once under Wine with every test of the binary on
`JOBS` threads of that process (one at a time by default; the Makefile says
what a higher `JOBS` waits for). It prints one `PASS`/`FAIL`/`SKIP` line per test and a
summary that counts every test, so the summary is the thing to read; a
failure is fatal to the default run (`FAIL_FAST=0` reports the whole suite).
One trap remains: a pipeline reports the last stage's status, so
`make test | tee log` returns the exit code of `tee`. Capture with a plain
redirect, `make test > out.log 2>&1`, and judge the run by the runner's
summary on both architectures. mtld3d's log of each test process is a file,
`<binary>-<pid>.log` under `mtld3d-logs` next to the test executable in
`windows/target`, one per process, so one file carries the whole suite's log.

Two things are worth knowing when a test process looks wrong. `d3d9.dll`
terminates the process from its `DLL_PROCESS_DETACH` once a device exists
(it cannot survive the allocator's thread-local teardown on Wine's 1 MB
main-thread stack), and that exit carries code 0 whatever libtest was
exiting with; the harness's panic hook (`windows/tests/src/win32.rs`)
terminates the process with libtest's failure code at the first failed
assertion instead, after the default hook has printed the report that names
the test. The tests in flight go down with the process: the runner marks the
named test failed and runs the rest again in a fresh process, and a crash or
a hang (no result for `TIMEOUT` seconds) is charged the same way, through a
one-thread re-run of the tests that were in flight when nothing names the
culprit. So a failure costs one result and one extra process, and the
`processes` count in the summary says how many the run took: six is a clean
`make test`.

## Which suite is right when they disagree

Wine's d3d9 test suite is the spec oracle. The end-to-end suite is our own
regression harness, so its assertions can encode our past bugs rather than D3D9's
behaviour.

When a conformance-improving change breaks an end-to-end test, the first question
is whether the assertion is wrong, not whether the change should be reverted. If
the new behaviour is what D3D9 does, update or delete the assertion and keep the
fix. Revert only when the change itself is wrong, for example when it rejects an
operation that is valid.

## Speed, conformance, and divergences

Speed is the goal and conformance serves it, so a divergence that buys frame time
and breaks no game is allowed to stay. What is not allowed is a silent one.

A kept divergence is a decision with three obligations: it gets its line in
the kept-divergence list of `docs/STATUS.md`, its rationale goes in the Kept
divergences section of `CONFORMANCE.md` (plus the cluster prose where Wine's
suite has a site for it), and where a knob makes sense it is revertible from
`mtld3d.conf`. "It matches what we ship today" is not a justification for a
change, because what we ship today may itself be the divergence. Prove a claim
about observable behaviour with a test that fails before the change and passes
after, and say in the pull request which way it went.

## Reference implementations

Before inventing a render-state heuristic, a per-shader allowlist, or any
game-shaped special case, read what an established implementation does. DXVK's
`src/d3d9/` (including its shader translator under `dxso/`) is the correctness
reference and has been hardened against thousands of D3D9 titles; dxmt is the
D3D9-on-Metal reference for the workloads it has been validated against; Wine's
own wined3d is the baseline that runs on the same machine, so a behaviour we get
wrong and it gets right is a regression by definition.

If one of them ships clean where we do not, the gap is usually a structural
feature we have not ported, not a quirk of one game. Port the structure. When
our architecture forces a deviation, say so explicitly in the pull request
instead of quietly simplifying.

## Working on conformance

```sh
make conformance                       # both architectures, diffed against the baseline
make conformance-i686                  # one architecture, one runner process
make conformance-isolate ONLY=visual ARCH=x86_64 REPEAT=1
make conformance-isolate ONLY=device ARCH=i686 REPEAT=20 VARIANT=intel LOG=debug
make conformance-baseline-i686         # re-record this architecture's entries
```

The rules that are easy to get wrong:

- Classifications live only in `CONFORMANCE.md`, on its `Sites:` lines.
  `baseline.txt` is machine-owned counts and crash state; the parser rejects
  class tokens there.
- A classification records the nature of a divergence, never its difficulty or
  how much a game cares. A hard-to-fix real defect is still real.
- Re-record the baseline in the same change as the fix that moves the counts, and
  check the diff: a re-record drops flaky-pinned sites that happened to read zero
  in that run.
- Derive the reason for a failing site from the upstream test source and from the
  raw actual-versus-expected values, which `MTLD3D_CONFORMANCE_RAW_DIR=<dir>`
  keeps. A site name is not a description of what the test exercises.
- Run gating runs with clean shader caches, and never re-record prefix drift: the
  Makefile pins the prefix display state before every run for a reason.

A shader-emission or shared-crate change has wide, subtle fallout. Run the whole
suite before committing, not just the subtest you were working on.

## Companion edits that belong in the same change

Each of these rots silently when it is left for later:

- A new render-state, texture-stage-state or sampler-state consumer moves its
  slot to consumed in the matching classifier, and flips the matching caps bit.
  The classifier warnings are only useful while every warning is a real gap.
- A change to shader emission bumps the shader-cache schema version. An unchanged
  cache key serves stale MSL, and the result looks like a rendering bug.
- A new config key ships with its dispatch arm, its unit test, and its entry in
  the `mtld3d.conf` sample with the default and a short why.
- A new built-in app profile ships with the rationale for every key it sets as
  the comment on its entry in `windows/core/src/app_profile.rs`, a test that
  resolves it from the version strings the shipped binary actually carries, and
  its line in the README profile list. A profile that pins no version field is
  not a profile, it is a name collision waiting to happen.
- A new `Clone` or `Copy` derive updates `scripts/derive_inventory.txt`
  (`scripts/audit.sh --update-derives`).

## Standing rules worth knowing before you write code

The full set is in `docs/CONVENTIONS.md`. These are the ones a newcomer trips:

- No new `MTLD3D_*` environment variable. A runtime knob is a `mtld3d.conf` key;
  a diagnostic is a narrowed `mtld3d::*` log target consumed through `RUST_LOG`.
- No fourth `#[allow]`. The tree carries exactly three, and complexity lints are
  never suppressed: introduce a parameter struct instead.
- No silent failures. Every stub, fallback and catch-all arm logs once.
- No `pub(crate)`, no `mod.rs`, no type aliases, no glob imports, no raw
  Objective-C selectors.
- Hash maps use `FxHashMap`; content hashing uses xxh3.
- Pure logic belongs in `mtld3d-core`; `windows/d3d9` is COM wiring. A COM
  wrapper carries a vtable pointer, a refcount and an opaque inner pointer, and
  every other field lives on the inner struct.
- Every integer with symbolic meaning that crosses the boundary is a typed value
  in `unix/shared`, never a bare `u32` and never a locally restated constant.
- Comments state the invariant, not the history that produced it: no incident
  provenance, no upstream test-file citations outside `unix/conformance/`, no
  absolute paths from someone's machine. The audit greps for these.
- No em dashes in prose, comments, commit messages or documentation.

## Pull requests

`main` is protected, so everything lands through a pull request, maintainers
included, and pull requests are squash-merged.

That makes the pull request, not the commit, the unit of review and the unit of
history. One pull request is one clearly defined change, and the squashed commit
is what has to stay bisectable on `main`. No drive-bys: an unrelated cleanup, a
rename or a reformat picked up along the way goes in its own pull request. It
keeps the review small and keeps `main` at one change per commit.

Commits inside the branch are discarded by the squash, so they need no polish.
The description is what survives, and for anything non-trivial it carries:

1. The user-visible symptom, quoting the exact log line or error where there is
   one.
2. The root cause, the mechanism, and say so when it is conjecture.
3. The fix, in its minimum conceptual steps.
4. At least one considered alternative, including the dead ends.
5. Verification: the commands you ran and what a reviewer should look for.

CI compiles on two machines and replays everywhere else. One job builds the
stage (`make stage`: both PE arches, both unix `.so` builds, the e2e test
binaries, the e2e and conformance runners for both host arches) and another
runs every lint, doc and unit leg as steps of one job before building the
bundle. The test machines carry no toolchain: they install the stage
(`STAGE=<dir>`) and run the end-to-end and conformance suites on three
images: the newest macOS on arm64, the oldest macOS mtld3d supports on arm64,
and the Intel image, whose device has no unified memory and none of the
packed 16-bit formats, so it runs the Intel/AMD code paths for real. One
more end-to-end leg runs the suite at `render.scale = 0.75`, the evidence
that the coordinates the tests assert on stay in the space D3D9 reports when
the frame is rasterized smaller; a test that needs single-pixel resolution
asks `render_scale_is_identity()` and pins its exact shape at the identity
rather than failing that leg. Every image gates. The Intel image reads the conformance baseline's `@mac2` entries,
which only it can record: dispatch the workflow with `record_intel_baseline`
and commit the `@mac2` sections from the `baseline-mac2-<arch>` artifacts
(`unix/conformance/CONFORMANCE.md` has the procedure). A conformance subtest
that dies on one image now and then is caught by dispatching with
`conformance_repeat=<n>`, which runs it that many times on every image and
uploads each run's raw output, ending in how the process ended, with the
layer's debug log beside it. Run `make conformance`
locally when your change touches the render or shader-emission path. The
manual
`probe-metal` job is how an image is checked before it is added, and it runs
under the forced Intel answers, because its device filters 32-bit floats
where a real Intel/AMD Mac's driver does not. On an Apple-family machine the
forced answers (`make test INTEL=1`, `make conformance-intel`) are the way to
run the Intel paths without the hardware.

The end-to-end legs run one test at a time in CI on purpose (`JOBS=1`),
because parallel device creation aborts on a runner. A flake there is not
fixed by re-enabling parallelism.

## What sends a pull request back

- `make check` or `make test` is red, or the run was judged by the summary.
- A new lint suppression, or a new environment variable.
- A conformance regression with no re-recorded baseline and no rationale.
- An end-to-end assertion deleted without saying which behaviour it contradicted.
- A behaviour change that diverges from D3D9 without being written down as a
  decision.
- A missing companion edit: classifier arm, caps bit, cache-schema bump, config
  sample entry, derive inventory.
- Incident provenance or a private path in a comment.
- Two unrelated changes in one branch.

## Issues and labels

The GitHub tracker is the backlog. A finding worth keeping, whether it comes
out of a review, a session, or a game report, becomes an issue the moment it is
not being fixed on the spot: one issue per finding, verified against the source
before filing.

Every open issue carries one type label, one priority label, and, where the
work can be estimated, one effort label:

- Type: `bug` is wrong behaviour in an implemented path; `enhancement` is a new
  capability or an unimplemented part of the D3D9 surface; `performance` is
  speed or memory with no behaviour change; `game-compat` is a specific game
  failing or misbehaving; `infra` is build, CI, test harness, or tooling.
- Priority: `P1` is next in line, a user-visible breakage or a live correctness
  hazard; `P2` is real and expected to get done; `P3` is speculative,
  nice-to-have, or waiting on its trigger.
- Effort: `effort/S` is hours; `effort/M` is a normal single-PR change;
  `effort/L` is multi-day or needs a design first.

Two modifiers exist. `blocked` marks an issue waiting on an external trigger,
a machine we do not have, or an upstream change, with the body naming exactly
what it waits for; a blocked issue keeps its priority, the label only says why
it is not moving. `needs-repro` marks an external report that has not been
reproduced or root-caused locally yet.

File the issue with its labels attached. When a claim in an issue body stops
being true, edit the body rather than correcting it in a comment trail.
