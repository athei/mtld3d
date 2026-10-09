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
| [`README.md`](README.md) | The goal, the features, the requirements, and where everything else lives. |
| [`docs/BUILDING.md`](docs/BUILDING.md) | Building from source, installing into a Wine tree, the gates and tests, the arm64 Wine switches, benchmarks and the release bundle. |
| [`docs/GAMES.md`](docs/GAMES.md) | The games tested so far and how far each one gets. |
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

- `make check` is `cargo fmt --check`, clippy with `nursery` and `pedantic`,
  `make audit`, the Makefile's own tests (`test-isolation` and
  `test-e2e-discovery`), and `make doc`. Only the check legs deny warnings, so
  a plain `cargo clippy` in an editor reports without failing. Each audit
  finding names the section of `docs/CONVENTIONS.md` behind it; read that
  section rather than pattern-matching your way past the grep.
- `make test` is the host-native unit tests plus the end-to-end suite under
  Wine, one leg per PE architecture. Conformance is not part of it, on purpose:
  many of its checks fail by design, so it gates on a regression against a
  baseline instead of on zero failures.

Two opt-in switches (see [`docs/BUILDING.md`](docs/BUILDING.md#arm64-wine))
add legs that run under an arm64 Wine, the one `WINE_ARM64` names. `ARM64=1`
adds `test-e2e-i686-arm64` and `test-e2e-x86_64-arm64` to `make test`, the
shipping builds under that Wine, and `EC=1` adds `test-e2e-arm64x`, the x64
suite against the ARM64X build, beside a `make check` leg that lints its two
halves. `make conformance` gets the matching `conformance-*-arm64` and
`conformance-arm64x` legs (`unix/conformance/CONFORMANCE.md`, "The
arm64-runtime legs"). Each of them installs into a private clone of
`WINE_ARM64`, creates a fresh prefix for it (a prefix only knows the builtins
that were installed when it was made) and runs one test at a time. The
end-to-end ones leave out two tests with the runner's `--skip`,
`window_lifecycle::devices_and_windows_come_and_go_on_several_threads_at_once`
and
`device::concurrent_retargets_deliver_every_window_message_to_its_own_device`,
which deadlock in that Wine's winemac; the Makefile says why beside the skip.
The environment variables the legs read are listed in
[`docs/BUILDING.md`](docs/BUILDING.md#which-target-reads-which-path).

`make fmt` uses nightly rustfmt. If a toolchain bump reformats files you never
touched, that churn is its own pull request, not a hand-revert and not a passenger
in yours.

Every test leg installs the build into the Wine tree `WINE_SDK` names (and into
`WINE_INSTALL_DIR` when set) before it runs, so two checkouts testing at once
overwrite each other's `d3d9.dll` and `mtld3d.so`, and a game launched from that
tree meanwhile runs whichever build landed last. `make test ISOLATED=1` avoids
both: it clones the SDK and the ambient prefix once into `.wine-isolated/`
inside the checkout (APFS clones, so neither costs space or a prefix boot) and
points the tools, the install and the prefix at the clones. Each clone is one
`clonefile(2)` on the directory (the Makefile's `clone_tree`), so it costs a
call rather than a walk of the file count, and falls back to `cp -c -R` and then
to `cp -R` when the source sits on another volume or on a volume that is not
APFS. Use it whenever another worktree may be testing or a game is running; a
plain `make install` still targets the shared trees on purpose, since that is
how the game gets a build. The display mode is the one thing a clone cannot
separate: the tests that change or read it take turns across every run on the
machine, as [`windows/tests/COVERAGE.md`](windows/tests/COVERAGE.md) describes.
The clones and the persistent wineserver of the
private prefix stay behind for the next run; `make clean-isolated` takes down
the ones in the checkout you are in, and `make clean-isolated-orphans` the ones
a removed checkout left behind. Either one ends the whole Wine session rather
than its server alone: the service processes a prefix keeps (`services.exe`,
the two `winedevice.exe`, `plugplay.exe`, `svchost.exe`, `rpcss.exe`) outlive a
server that is merely signalled, and nothing in their name says which checkout
they belong to, so what is left over after the server is gone is found by the
paths it holds open. `clean-isolated-orphans` finds the environments three ways:
the clones beside the checkouts, the record every isolated checkout writes into
the directory the worktrees share, and any process still running out of one. So
neither a removed directory, nor a checkout kept somewhere else, nor a reboot
that took the servers hides one. A directory that is still a checkout is left
alone there: its environment may be mid-run, and it is that checkout's own
`make clean-isolated` to take down. A failing run whose log shows a `d3d9.dll v`
stamp that is not your checkout's is that collision, not a regression.

`scripts/audit.sh --file <path>` runs the per-file subset of the audit, so an
agent runner or an editor can call it after every edit and surface a violation
while you write rather than at commit time. Such a hook, and any digest of the
conventions an agent runner prints at session start, is optional local setup:
the tree tracks neither, nothing keeps a digest in step with
`docs/CONVENTIONS.md`, and `make check` is the gate either way.

## Reading a test run

The end-to-end suite is six test binaries per architecture, and the runner
in `unix/e2e` runs each one once under Wine with every test of the binary on
`JOBS` threads of that process (four at a time by default; the Makefile says
what that assumes of the Wine it runs under). It prints one
`PASS`/`FAIL`/`SKIP` line per test and a summary that counts every test, so
the summary is the thing to read; a failure is fatal to the default run
(`FAIL_FAST=0` reports the whole suite). One trap remains: a pipeline
reports the last stage's status, so `make test | tee log` returns the exit
code of `tee`. Capture with a plain redirect, `make test > out.log 2>&1`,
and judge the run by the runner's summary on both architectures. mtld3d's
log of each test process is a file, `<binary>-<pid>.log` under `mtld3d-logs`
next to the test executable in `windows/target`, one per process, so one
file carries the whole suite's log.

A process that ends cleanly is still checked against libtest's own account.
Its `test result:` line counts the results it printed, and a runner tally
short of that count means a result never arrived, whatever tore it off the
pipe. The note is `libtest counted <n> results and the runner read <m>; no
outcome for: <names>`, the named tests run again in a fresh process, and the
summary's total therefore never falls short of the set the run was asked for.
A second process that reports none of them again fails them rather than
looping.

A process that ends with tests unaccounted for leaves its whole stderr in the
same directory as `<binary>-<pid>.stderr`, and every line the runner prints
about that process names the file. What it prints inline is the last fifteen
lines the process itself wrote: Wine's `fixme` lines and everything its
`dbghelp` channel prints are left out, because dozens of them surround each
backtrace and a raw tail rarely reaches back past them to the message that
says what failed. The layer's own log of that process is the other account,
and the only one of a death the layer's crash handler ended: its fatal banner,
registers and stack go there and never to stderr. The runner moves that log
to `<binary>-<pid>.layer-log` beside the stderr, because the layer keeps only
its ten newest logs and the next run would remove it, and quotes it in the
same note from the banner on when there is one. Every test process runs
under Apple's Main Thread Checker (`debug.mainThreadChecker` in the suite's
config, `docs/ARCHITECTURE.md` says how), so a kept `.stderr` that carries a
`Main Thread Checker: UI API called on a background thread:` line names the
AppKit call the layer made off the main thread and the thread that made it;
that call is the failure, whatever the process printed after it.

Process cleanup has one two-second budget from the first termination or cleanup
attempt, including signal-error retries and the final reap. The leader stays
unreaped until the last group signal. If cleanup cannot establish exit and reap,
the runner reports the unresolved PID and exits immediately with infrastructure
code 2. A survivor may still be running: macOS adopts the remaining children and
owns their eventual reap. No background thread survives the runner to do that
work. This bounds the runner's wait/retry loops, subject to scheduling, syscalls
and diagnostic output; the kernel can also delay process exit while closing
file descriptors. No further binary runs after this failure.

An explicit driver GPU-hang report ends the leg with exit code 3 and no
verdict, even under `FAIL_FAST=0` or when the process itself exits cleanly.
The runner watches stderr while the process runs, checks the layer log before
it can launch another process, and keeps both accounts of the initiating
process. Assertions after that report are not measurements of the source: the
later results may reflect the hosted GPU's failed state. CI re-runs such a leg
on a fresh machine; "Pull requests" below says when.

When every test is accounted for but the process ends abnormally, the runner
keeps its full captured stdout, stderr and exit status together in
`<binary>-<pid>.process-log` and names it in the note. This preserves the
existing test verdicts and does not retry completed tests. The streams are
captured decoded text, not a byte-exact pipe recording. A retention error is
reported in the note without changing those verdicts. Clean successful and
expected self-exiting processes create no such file. As with stderr, the
newest ten bundles remain in the selected log directory.

In CI every end-to-end leg uploads all four kinds of file as its
`e2e-logs-<image>-<arch>` artifact on every run, kept fourteen days; the
directory holds the ten newest of each, so a leg that restarted more than ten
processes hands back its last ten. They are the layer's side of a red leg,
which the job log lacks: when one image reads back black on test after test
while its sibling legs are green, re-run the failed jobs first (`gh run rerun
<run-id> --failed`, which lands on a fresh machine) and read the command
buffer errors in the artifact's log, since a hosted runner's GPU can fail that
way with no hang line in the job log and nothing else tells it from a
regression. The Intel image's GPU, the paravirtual `AppleParavirtGPUMetal`,
refuses a texture or a view now and then while several threads create
and release them, with its kernel logging `addObject: Object already exists`
in the system log, so on a paravirtual device the layer asks again for up to
a quarter of a second and logs each create it recovers that way. A
`CreateBackbuffer` failure on the Intel image is therefore read from two
lines. The unix line names the request and the device; a sane request (the
window's size, `BGRA8Unorm`, a sample count the device answered for)
together with the layer's `still refused after` line for it is that runner
fault outlasting the retries, so re-run the failed jobs and add the leg to
the issue that tracks the fault. A line naming a zero dimension or a null
handle is the layer's own bug.

Two things are worth knowing when a test process looks wrong. Once a device
has existed, `d3d9.dll` terminates the process from the
`DLL_PROCESS_DETACH` of its exit (it cannot survive the allocator's
thread-local teardown on Wine's 1 MB main-thread stack), so a test binary's
exit status is whatever that `TerminateProcess` carries: the status the
process asked to exit with, and 0 for a process that never asked. A
`FreeLibrary` after a device does not reach that detach: the first
`CreateDevice` pins the image, so it stays mapped and the process carries
on. The harness's panic hook
(`windows/tests/src/win32.rs`) does not wait for libtest to reach its own
exit and terminates with libtest's failure code at the first failed
assertion, after the default hook has printed the report that names the
test. The tests in flight go down with the process: the runner marks the
named test failed and runs the rest again in a fresh process, and a crash or
a hang (no result for `TIMEOUT` seconds) is charged the same way, through a
one-thread re-run of the tests that were in flight when nothing names the
culprit. So a failure costs one result and one extra process, and the
`processes` count in the summary says how many the run took: twelve is a
clean `make test`.

Explicit test selections, including recovery rounds and filtered initial runs,
are split into batches that fit the Windows command-line limit. The budget
counts UTF-16 units after quoting, with flags, separators, the terminator and
an allowance for Wine's executable-path mapping. Every deferred name keeps
its intended thread width, so a serial recovery stays serial across batches.
Each batch is an actual process with its own summary and retained failure
logs; the original failure remains failed. Fail-fast and a GPU-hang report
stop before later batches. A single name that cannot fit reports a runner
error instead of repeatedly launching an impossible command.

The harness defaults to hidden borderless windows. Wine builds a framed
window's title bar and controls on the AppKit main thread, so creating and
destroying them for every rendering test serializes a parallel run.
Tests of window-style changes use `HarnessConfig::window_style` with
`WindowStyle::Framed`, as does the window-lifecycle stress test. Keep that
choice explicit when a test needs the non-client frame.

## Benchmarks

The benchmark launchers set `RUST_LOG=info` and `__CX_UNIX_RUST_LOG=info`
for timed Wine processes, independent of the shell's filters: startup identity
and perf records are required inputs, even when no warning occurs. Untimed
shape processes set both variables to the pass-trace filter described below.
Ordinary test runs keep the caller's logging settings.

The synthetic benchmarks are `#[ignore]`d tests of the end-to-end binary, so
`make test` lists them as ignored and never runs them. `make bench` runs them
once against this checkout and writes a report per benchmark. `make bench-ab
BASE=<ref>` is the one that answers whether a change made things slower: it
builds `BASE` in a worktree of its own and this checkout, each into its own
isolated Wine tree, runs every benchmark against both in alternating order
for `RUNS` rounds (five by default, three at least), and judges each metric
pair by pair against the noise those rounds show. A leg's round is one
process running every benchmark in libtest's order, which is the same in
both legs, so what one benchmark leaves in the process (the process-wide
pipeline cache, the page-box pool) reaches the next alike on both sides of
each pair; the memory rows a comparison judges are each benchmark's growth
from a sample taken before its interface. That growth is clamped at zero,
and memory an earlier benchmark frees late, inside a later one, can hide
some of the later one's growth, which one process a round cannot avoid.
The two runs of one benchmark in a round are therefore a whole round
process apart, about 33 s with the `wow` set, rather than back to back, and
with an odd `RUNS` one leg goes first in one round more than the other (3
to 2 with the default five). Before each round's process the run measures
for half a second the CPU its busiest other processes take, and the report
warns about every round that started on a busy machine (one other process
at a quarter of a core, all of them at half, or macOS throttling for heat);
run those again. `BENCH_WAIT_IDLE=120` optionally waits up to 120 seconds
before each timed process for three consecutive quiet half-second CPU
samples, using those same thresholds. The default `0` keeps advisory
sampling. A failed CPU sample cannot count as quiet: both process reads
must succeed and include the sampler itself. Some macOS hosts do not expose
`kernel_task` to `ps`; its CPU share then remains unavailable, explicitly
recorded in each idle sample, and only the foreign-process thresholds can
be checked. Each attempt stays in
the round's `idle-<binary>/sample-*.txt` files (or `idle-host/`), including
rejected samples, and `idle-<binary>.txt` records the wait and its outcome.
The final accepted sample is the ordinary `machine-<binary>.txt` that the
report reads. A timeout ends the run with exit 2 and no performance verdict,
preserving completed rounds. This checks the machine before a process
starts; it cannot promise that background work stays idle during it. The
runner exposes the same option as `bench-ab --wait-idle <seconds>`, alongside
its explicit Wine, prefix, stamp and binary paths, so a rebuilt runner can
compare existing installed product builds without rebuilding those legs.
`--base-config` and `--cand-config` replace the shared `--config` for the
named leg, including an explicitly empty value; the other leg keeps the
shared config. These apply to timed tests and their untimed shape runs,
with the run's own `log.dir` appended last. For a configuration comparison
of one clean build, `--allow-same-image` permits the same recorded image
identities on both legs; use complete config strings and retain their
runtime logs so the intended difference can be checked. The host emitter
does not load the renderer and does not use these runtime settings. The runner
exits 1 on a regression and 2 when the run itself cannot be trusted, which
includes the two legs running different Wines. The runs and the report stay
in a directory under the main checkout's `.codex/evidence/bench-ab`, and
`make bench-compare AB_DIR=<dir>` judges one again into a report of its own,
for instance with `ACCEPT=<metric>,...` naming an exact count (a draw count,
a pass count) that the change is meant to move. `make clean-bench-ab`
removes the kept base worktrees. The metrics a benchmark writes include the
layer's own counters, read from the `perf-kv` line of its perf windows as
`perf.*` (`bench.rs` gives the rules): the per-frame counts of work the API
calls fix, such as `perf.draws_pf` and `perf.passes_pf`, are the exact ones.

`ARM64=1` and `EC=1` (see [`docs/BUILDING.md`](docs/BUILDING.md#arm64-wine))
move `make bench` and `make bench-ab` onto the arm64 Wine that `WINE_ARM64`
names: `ARM64=1` runs the `ARCH` build there, `EC=1` the ARM64X build under
the x86_64 benchmark binary, and each leg of `bench-ab` gets a private clone
of that Wine with a fresh prefix, the way the arm64 test legs do. A benchmark
measures one layout, so the two switches are not given together, and
`EC=1 make bench-ab` needs a `BASE` whose Makefile has `windows-arm64x`: it
stops with a message naming `BASE` otherwise. `make bench-variants` compares
layouts instead of commits: this checkout's x86_64 build on the SDK's Wine
against the same build on `WINE_ARM64` (`x86_64-sdk-vs-arm64`), and with
`EC=1` the x86_64 build against the ARM64X one on `WINE_ARM64`
(`x86_64-vs-arm64x`). `ARCH=i686` runs the first pair for the i686 build
instead (`i686-sdk-vs-arm64`), the path World of Warcraft 1.12 and 3.3.5a
take on an arm64 Wine, whose x86 translator runs our 32-bit DLLs; there is no
ARM64X pair for it, since a 32-bit process never loads the ARM64X build. Every
pair is a `bench-ab` run of its own whose report names each leg's runtime and
DLL variant. Such a run has one commit in both legs, so a binary the two
layouts share loads as one image, which the comparison notes instead of
refusing, and the first pair runs two Wines. Read that first pair with care:
it changes the host arch and the Wine build at once (the SDK is a patched
CrossOver 26, `WINE_ARM64` a stock CrossOver 27), so its differences are not
the arch's alone, for either arch. The ARM64X pair runs one Wine in both legs
and changes only our DLLs, so it measures what the ARM64X build buys an x64
game.

On an arm64 Wine the benchmarks' clock is FEX's translation of `rdtsc`, a
counter read that is not ordered after earlier stores (an `mfence` does not
order it either), so stores still draining when a timed call starts are
charged to that call. A row that times one call alone between writes is
then not the call alone: `texture_streaming`'s `lockrect.p50` and
`lockrect.p99`, the new textures' `LockRect` rows, carry part of the
previous level's fill, and move with a change that only makes that write
slower. Judge those on an arm64 leg by `fill.p50` and `fill.p99`, which time
each level's `LockRect`, write and `UnlockRect` as one interval. A texture's
levels are filled one after another, so a drain that misses one level's
interval lands in the next level's, and only the last level's can leave
the row.

The per-call rows of `api_call_cost` (`ns_per_call.*`) time single calls of
8 to 85 ns, and they move with where the linker places the layer's
functions as well as with what the functions do: single setter rows have
moved by 30 to 50 % in runs whose change did not touch the call. A
comparison therefore fails a setter row only past 50 %, and a smaller
change to the cost of a call is for the frame benchmarks to show. The draw
rows (`ns_per_call.draw_*`) move far less and fail past 15 %. An A/A run
(`BASE=HEAD`) builds one commit twice with the same code placement, so it
does not reproduce this; it has still moved single setter rows by about
30 % (`5f274cd66021`, `aad71d015386`), for a reason not established (data
placement is a candidate). `RATIO_FLOOR_PER_CALL` in
`unix/e2e/src/bench/compare.rs` gives the measurements.

An exact count a change moved on purpose fails every comparison against a
base from before that change, so such a run names it in `ACCEPT`, and only
that run: the next change to the count fails again. Against v0.11.0 or an
older base, `cold_start`'s `prewarm.shaders` reads 98 in the base and 92 in
the candidate. Since #1050 an unlit fixed-function draw no longer keys
`D3DRS_SPECULARENABLE` into its vertex shader (`build_vs_flags` in
`windows/core/src/ff_state.rs`), and the benchmark's 24 fixed-function
combinations toggle specular on 12 unlit ones, so six pairs of them share
one shader. Nothing goes unprepared: both legs pre-warm 196 pipelines with
none failed or skipped, and every shader the candidate's cache holds is
pre-warmed. Such a run takes `ACCEPT=prewarm.shaders`.

`make bench-host` is the one benchmark that needs no Wine: it times DXSO
parsing and MSL emission on this machine over two synthetic corpora and any
shader cache `BENCH_CORPUS` names, and writes its metrics into the `host`
directory beside the reports of `make bench`. `make bench-ab` runs it too, in
rounds of its own before the others, the run's first benchmark processes
(only the short Wine process that lists the benchmarks comes before), so no
benchmark's Wine process can still be exiting on the cores it times: each
leg builds and runs its own
tree's emitter, both read the same `BENCH_CORPUS`, and its MSL byte counts
are exact, so a change that alters the emitted code shows up there even when
its time per shader stays inside the noise. A `BASE` older than the host
benchmark runs neither leg's, and the run says so. The same caches are what
`cold_start` measures in both legs. A cache only one build can read (a
format or schema change between them) is skipped for both with a note.
`cold_start` counts a cache's records with the candidate's reader, so
when the base's layer writes another schema the base leg reports its
timings without those counts, and their rows read as added. Any
other difference in what a benchmark ran, such as its own configuration
entries or the depth path it took, stops the comparison: only the build and
the run may differ between the legs.

After its timed rounds, each scene benchmark (one whose metrics declare its
frame in `shape` lines) runs once more per leg with the pass trace on
(`mtld3d::d3d9::passes=trace`, the rest of the layer at warn but for the
lines that name its build and the perf windows its measured frames start
on), untimed, into
`<leg>/shape/` of the run's directory; a benchmark without `shape` lines, such
as the shader-stutter one, gets a note instead. The runner stops that run
once its log holds 33 submissions after the line the benchmark prints where
its measured frames start, so it writes no metrics: its stamp and images are
read from the log's identity lines and held to the leg's stamp and to the
images of the leg's timed rounds. That costs a run of about six seconds per
leg per scene benchmark and some MB of trace in the directory. The report
compares the most common pass shape of the last thirty complete submissions
between the legs, with every load and store action the load/store rules
decided on those passes, and a leg in which fewer than 80 % of them agree is
an untrustworthy run, exit 2. A rule that drops a store a
later pass needs makes the frame faster, not slower, so the timings cannot
catch it and this comparison does: any difference is a shape change, which
fails the run like a changed exact metric unless `ACCEPT` names `shape` or
`shape:<bench>`.

`make bench-shape GAME_LOG=<layer log> BENCH_METRICS=<bench-<name>.metrics>`
checks a benchmark's scene against a frame a game dumped with Ctrl+Shift+P:
the pass count, and per pass the draw count, the fixed-function share and the
textures per draw, with render-target sizes shown relative to each side's
back buffer. Each pass is also compared on its state mix, which a benchmark
declares by appending these counts to its `shape` line: `blend=`, `atest=`,
`zwrite_off=`, `cull_none=` and `cmask0=` count the draws with blending on,
with alpha test on, whose depth-write field is 0, with cull mode NONE, and
whose render target 0 colour write mask is 0, and are judged as shares of
the pass's draws within 10 points. `vs_sw=`, `ps_sw=`, `tex_sw=`,
`blend_sw=`, `atest_sw=` and `cull_sw=` count the draws after the pass's
first whose vertex shader, pixel shader, stage-0 texture, whole blend tuple,
whole alpha-test tuple or cull mode differs from the draw before (fixed
function and no texture are values like any other), and `vs_n=`, `ps_n=`
and `tex_n=` count the distinct programmable vertex and pixel shaders and
the distinct textures on any stage. Those nine are judged within 15 % of the
game's count, but never closer than 5. A `shape` line without the state keys
still parses: its state mix prints as not reported and is not judged. Run it
by hand when building or reshaping a scene that stands for a game; it is no
gate.

Bench numbers come from `PROD=1 PERF=1` builds only; `make bench-ab` builds
both legs that way and refuses any other profile, since `release` carries
debug assertions that cost more than the frame does. Nothing else may run on
the machine during a benchmark: no test leg, no conformance run, no build in
another worktree, no game. Each of them takes the same cores and GPU the
numbers measure, and a verdict is only as good as the quiet it was measured
in. The benchmark targets unset `MTL_CAPTURE_ENABLED` for their Wine
processes, as they turn the Metal validation layer and HUD off: Metal's
capture layer sits under every Metal call, and
[`ARCHITECTURE.md`](docs/ARCHITECTURE.md#ctrlshiftp-three-frame-dump-and-gpu-capture)
gives its measured cost. A deliberate comparison with the capture layer
runs the benchmark binary through the runner directly; every perf window
says whether it was on (`metal_capture_layer_count`, judged exact, so legs
that differ in it fail).
`BASE=HEAD` is an A/A run of one commit against itself (or of your
uncommitted changes against the commit they sit on) and shows how far the
machine moves the numbers by itself; run it when a verdict looks surprising.

Run `make bench-ab BASE=origin/main` with the default `BENCH_SET=wow` before
merging a change to the render, encoder, submit or shader path, and put the
summary line in the pull request's verification. The `wow` set is the two
World of Warcraft frames, the EVENT-query throttle under the game's settings
and under the D3D9 defaults, the per-call API cost, and the buffer-lock and
texture-streaming benchmarks at the game's rates. A leg's round is one
process that runs all seven in libtest's order, the same in both legs, each
on a device of its own: about a second for Wine's start, then per benchmark
about 4.5 s for its device, the warm-up, untimed frames until its second perf
window opens two seconds after its first frame, and that one window
measured. So a round of one leg takes about 33 s and the default five rounds
of both legs about 5.5 minutes, the four shape runs under half a minute and
the host emitter's rounds about 20 s. Before the first run the two legs build at the same time
and their prefixes boot while the benchmark binary builds: about a minute and
a half with a new base worktree and a changed candidate, well under a minute
when neither needs a build, longer when a change rebuilds the production
layer in both legs. An A/A run of one commit against itself should take
about 8 minutes in all, down from the 20 an earlier layout of the run took
(an estimate from that run's timestamps and the new spans; the first run's
timestamps say what it is on your machine). `BENCH_SET=full` runs every
benchmark, the shader-stutter and cold-start ones included, and takes about
twice as long. `BENCH_SET` may also be a space-separated list of test-name
filters, each selecting the benchmarks whose test path contains it, mixed
with the set names (`wow cold_start`), so `BENCH_SET=dynamic_buffer_churn`
rechecks one suspicious benchmark without the rest of the set; the host
emitter's rounds run with a set name or a filter that is part of
`host::emit_corpus`.

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
  check the diff: a re-record drops flaky- or ceiling-pinned sites that happened
  to read zero in that run.
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
- Shader emission has a source-derived fingerprint (`windows/core/build.rs`).
  Changes under `dxso/` and to its shared MSL inputs invalidate MSL automatically;
  extend the source list when adding an emission dependency outside those paths.
  Bump the shader-cache schema for incompatible shader keys or pipeline-translation
  semantics, and the container version for binary-format changes. Programmable
  DXSO survives emitter changes; fixed-function entries rebuild on first use.
- A new config key ships with its dispatch arm, its unit test, and its entry in
  the `mtld3d.conf` sample with the default and a short why.
- A new built-in app profile ships with the rationale for every key it sets as
  the comment on its entry in `windows/core/src/app_profile.rs` and a test that
  resolves it from the version strings the shipped binary actually carries.
  `docs/GAMES.md` names a game's profile in its row; the settings and
  reasons stay in `app_profile.rs`.
  A profile that pins no version field is not a profile, it is a name collision
  waiting to happen.
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
- Pure logic belongs in `mtld3d-core`; `windows/d3d9` is COM wiring. Runtime
  placement follows the Unix-first policy in `docs/ARCHITECTURE.md`; native
  orchestration reuses core logic. A COM wrapper carries a vtable pointer, a
  refcount and an opaque inner pointer, and every other field lives on the
  inner struct.
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
runs formatting and audit first, then documentation, Clippy and unit tests
as steps of one job. Production bundles are built by the release job only.
Every run on `main` has its own concurrency group so pending runs survive
later pushes and every commit keeps its CI result. PR updates cancel the
superseded run. The test machines carry no toolchain: they install the stage
(`STAGE=<dir>`) and run the end-to-end and conformance suites on up to three
images: the newest macOS on arm64, the oldest macOS mtld3d supports on arm64,
and the Intel image, whose device has no unified memory and none of the
packed 16-bit formats, so it runs the Intel/AMD code paths for real. A pull
request runs the two arm64 images, and every leg it runs gates it. The Intel
image's paravirtual GPU hangs and reads multisampled results wrong by itself,
with no Wine and no mtld3d involved, often enough that a pull request could
not count on a green Intel leg, so its legs run every night on `main` and on
a dispatch with `intel_only`, which run that image alone; a push to `main`
and any other dispatch run all three images. An Intel failure in the nightly
run that is not a GPU hang comes from the commits since the last green one:
a regression, or a `@mac2` conformance pin left stale by a pull request that
fixed a site and updated only the Apple entries, which is re-recorded rather
than read as a regression (`unix/conformance/CONFORMANCE.md` has the
procedure). So a pull request that moves conformance counts, edits
`baseline.txt`, or touches an Intel code path (the `intel.*` keys, Managed
memory, the packed 16-bit formats) dispatches the workflow with `intel_only`
on its branch before it merges. A release waits for a green Intel run (see
"Cutting a release"). One more end-to-end leg, and one more conformance leg,
run their suite at `render.scale = 0.75`, the evidence that the coordinates
the tests assert on stay in the space D3D9 reports when the frame is
rasterized smaller; a test that needs single-pixel resolution asks
`render_scale_is_identity()` and pins its exact shape at the identity rather
than failing that leg, and the conformance sites that cannot (a probe on a
colour boundary) are classified under "The scaled leg" in
`unix/conformance/CONFORMANCE.md`. The Intel image reads the conformance
baseline's `@mac2` entries, which only it can record: dispatch the workflow
with `record_intel_baseline` (and `intel_only`, to run nothing else) and
commit the `@mac2` sections from the `baseline-mac2-<arch>` artifacts
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

The end-to-end legs run one test at a time in CI on purpose (`JOBS=1`, against
a local default of four), because parallel device creation aborts on a runner.
A flake there is not fixed by re-enabling parallelism.

A leg a GPU hang cut short (the runner's exit code 3, under "Reading a test
run") has no verdict, and a hosted machine whose GPU hung stays hung, so the
e2e and conformance steps mark such a leg with an error annotation titled
`GPU hang`. When a run fails and every job that did not succeed carries that
annotation, `.github/workflows/rerun-gpu-hangs.yml` re-runs the failed jobs
once, which puts them on fresh machines; it leaves a pull request's run alone
once a newer push has replaced it. A second hang, or a hang beside any other
failure, is for a person to read, and the failed jobs are re-run by hand
with `gh run rerun <run-id> --failed`.

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

## Cutting a release

A release is a `v*` tag, and pushing one is the whole trigger: CI drafts the
release, attaches the two archives `make bundle` produces, and leaves the body
empty for the notes, which are written by hand.

The version lives in the `[workspace.package]` table of both `Cargo.toml`s and
in the local-crate entries of both `Cargo.lock`s, so the bump is its own commit
touching four files. Refresh each lock with `cargo metadata --format-version 1
--manifest-path <workspace>/Cargo.toml > /dev/null`, which rewrites the version
lines without the dependency churn `cargo update` would bring, and check the bump
before tagging with `make version-check TAG=vX.Y.Z`.

Before the bump, measure the release against the previous one on a quiet
machine with `make bench-ab BASE=<previous tag> BENCH_SET=full
LOG_DIR=$PWD/.codex/evidence/bench-release/<version>`, which archives the runs
and the report there. A regression it reports is fixed before the tag or named
in the notes.

Land that commit and wait for its run on `main` to go green. The latest Intel
run on `main` has to be green too: that is the commit's own push run, which
carries the Intel legs, unless a nightly or an `intel_only` dispatch ran
after it. Pull requests do not run the Intel image, so that run is the only
Intel verdict the release has, and a run whose Intel legs show nothing but
GPU hangs is re-run until it gives one. Then push the tag. The
release job refuses a tag whose version disagrees with either workspace, and
refuses one whose commit has no green run on `main`; a tag pushed while that run
is still going is waited on rather than rejected. It builds the bundle itself,
after the tag exists, because every binary stamps `git describe` and one built
before the tag names the previous release.

What is left is the notes and the publish. Group the points into sections rather
than one flat list, give the two or three headline items a section each, and end
every point with the pull requests that did it. Then
`gh release edit vX.Y.Z --draft=false --latest`.
