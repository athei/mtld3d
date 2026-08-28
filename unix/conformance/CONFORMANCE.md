# d3d9 conformance against Wine's test suite

The de-facto conformance suite for any D3D9 reimplementation is Wine's
`dlls/d3d9/tests/` (there is no public, portable Microsoft D3D9 conformance
kit — the WHQL/HLK tests are driver-certification machinery). Those tests build
into one `d3d9_test.exe` per architecture, with four subtests selected by
source-file stem: `device`, `visual`, `stateblock`, `d3d9ex`.

Because our `d3d9.dll` is installed as a Wine *builtin* (`make install`), running
`d3d9_test.exe` exercises our implementation directly.

## Running

```
make conformance                # diff both arches vs baseline.txt
make conformance-i686           # one arch, one runner process (what CI runs)
make conformance-x86_64
make conformance-baseline       # (re)record baseline.txt, both arches in sequence
```

Set `MTLD3D_CONFORMANCE_RAW_DIR=<dir>` to also persist each subtest's full raw
output to `<dir>/<arch>-<subtest>.log`. The normal run reduces output to per-site
counts and drops the assertion text; the raw logs keep every
`<file>.c:<line>: Test failed: Got <actual>, expected <expected>` message (plus
the Metal-validation lines), which is what the per-cluster audit below was built
from — the *actual-vs-expected* values distinguish a real defect from an
acceptable `caps` difference. Off unless the variable is set.

There is no conformance-specific input to set. The test binaries ship inside the
Wine SDK bundle (`$WINE_SDK/lib/wine/tests/{i386,x86_64}-windows/d3d9_test.exe`,
published by the [wine-build](https://github.com/athei/wine-build) bundle step),
which is the same install `make install` puts our builtin `d3d9.dll` into, so a
CI job needs nothing but that tarball and no Wine build tree at all. The
binaries are **not** vendored here: they are large and drift with the Wine
version, so `baseline.txt` records the Wine version it was taken against
instead.

The runner is the Rust tool `mtld3d-conformance` (`unix/conformance/`). It takes
the loader and one test binary as explicit paths (`--wine`, `--exe`, plus
`--arch` as the label to record under) and resolves nothing itself: every Wine
location lives in the Makefile. One invocation therefore covers one
architecture, which is what lets the 32-bit and 64-bit gates be separate CI
jobs, and `--update-baseline` rewrites only its own arch's entries. It runs each
subtest as its own process, so a crash in one cannot poison another's counts,
with Metal API validation left on in `nslog` mode (it logs rather than aborting,
so it cannot mask the failure counts) and with our logs and Wine's debug
channels silenced.

This is **not** part of `make test`: many checks fail by design (see below), so
it is a tracked-score tool, not a pass/fail gate on zero. The runner exits
non-zero on a *regression* vs the baseline — a per-site failure count that went
up, a new failing site, or a subtest that started crashing — and equally on a
*stale baseline* — a count that dropped, a site that disappeared, or a crash
that cleared. An improvement fails the gate on purpose: tolerating it would let
baseline.txt overstate reality, and the surplus becomes a budget a later
regression can hide in. The fix for a stale baseline is `make
conformance-baseline` plus the matching triage edit here, not a code hunt. The
`flaky` and `ceiling` classes are the two tolerances (see below).

## What the baseline records — and where classes live

Each datum has exactly one authoritative home, split by who writes it:

- **`baseline.txt` (machine-owned)** records the current *results*: for each
  `(arch, subtest)`, the crash bit plus every failing assertion as a
  `<file>.c:<line>` site with a hit count. No classifications — the file is
  freely rewritten by `make conformance-baseline` without ever touching prose.

  ```
  [i686/device] crash=1
    device.c:125 count=28
    ...
  ```

- **This document (human-owned)** records *why* each site fails: the
  per-cluster section below declares every site's classification as a
  `<line>=<class>` token on a `Sites:` line, next to the rationale prose.
  The runner loads classes from here (for the flaky/ceiling tolerances and
  untriaged reporting); a unit test in the runner crate fails `make test` unless the
  two files cover exactly the same sites — so a new baseline site stays loud
  until someone writes its rationale, and a fixed site's prose must be
  removed rather than lingering as history.

Per-site granularity is what makes the score actionable: the 74 `device`
failures, for instance, are really three source lines hit repeatedly in a loop,
not 74 distinct defects. Recording the location (not just a total) means a fixed
bug and a new regression can no longer net out to the same number and hide each
other.

`make conformance-baseline` re-records `baseline.txt` and prints exactly which
sites are new (add them to a cluster below, with a rationale) and which were
dropped (delete their tokens and trim the prose). A run whose Wine version
differs from the baseline's recorded version warns that `file:line` sites may
have drifted (a Wine update renumbers source lines) and a re-baseline is
expected — the `Sites:` tokens here renumber with it.

## Classification tags

Each failing site carries one tag. The tag is a deterministic property of the
divergence's NATURE — never of fixability, difficulty, or in-game value (a
hard-to-fix or low-value defect is still `real`):

- **`real`** — a genuine defect we intend to fix: our output/behavior is wrong
  and no deliberate design rationale covers the divergence. A mixed line (any
  intend-to-fix component alongside by-design assertions) is `real`, with the
  remainder explained in prose.
- **`caps`** — the failure exists only because the test assumes a capability we
  deliberately don't advertise, AND our actual behavior is the conformant
  response for a device without that capability (correct pixels, or the
  spec-correct rejection). A cap-*respecting* test simply passes and never
  lands here; `caps` covers cap-*blind* assertions (Wine's tests assume caps
  that real desktop drivers always have) and escapes offered only under
  `broken()`, which the runner does not honor. If our response to the missing
  capability is itself non-conformant, the site is `real`.
- **`expected`** — we deliberately do not implement this and intend to keep it
  that way, for a positive, documented reason: a scope decision (D3D9Ex,
  device loss, desktop mode switching — see below), a kept perf tradeoff
  (the TBDR depth-store elision,
  buffer-rename over stalls), or an accepted platform limitation (Metal's
  0xffff primitive restart, GPU-defined NaN encodings). "We don't want to fix
  it" or "the fix is invasive" is not a rationale — without a positive reason
  to keep the divergence, the site is `real`.
- **`flaky`** — environmental/non-deterministic (display config, Retina scale,
  macdrv window-manager timing). Count changes in either direction never gate.
  Tag reactively — only once a flutter actually trips the gate — and pin the
  HIGHER observed count so a flutter back up is not a false regression.
- **`ceiling`** — the pinned count is a cross-environment MAXIMUM, not an exact
  value: the same baseline serves environments where the site legitimately
  reads lower (a CI runner's virtual display accepts the mode changes this
  machine's macdrv rejects, so the desktop-mode sites read zero there; the
  fetch4 counts wobble with the attached display). Reading below the pin is
  tolerated and does not demand a re-record; reading above it gates like any
  regression. The tag adds only that tolerance — the divergence's nature stays
  in the cluster prose, and like `flaky` it is assigned reactively, from a
  measured cross-environment delta, never speculatively.
- **`crash`** — a site attributed to a crash/abort path.
- **`untriaged`** — an explicit placeholder for a site a human has not yet
  triaged. Normally untriaged means *absent from this document* (the sync test
  stays red until prose exists); writing `=untriaged` is the escape hatch for
  landing a re-baseline before the triage is done — the runner still flags it
  on every run.

The counts are the signal, not a target of zero. Wine's `todo_wine`/`broken()`
annotations are tuned for a real-GPU driver, not for us, so a raw failure is
not necessarily a real defect — the classification is what turns the number into
something actionable. Note that when a subtest crashes, the counts cover only the
failures reached *before* the crash truncated the run.

## Per-cluster classification

This section is the authoritative home of every failing site's classification
and rationale, grouped by enclosing Wine test function. The classes exist only
here (`baseline.txt` holds counts); the runner loads them at gate time, and a
unit test in the runner crate fails if any baseline site has no `Sites:` entry
below, any entry names a site that no longer fails, or a site is declared
twice. When a re-baseline adds or removes sites, update the matching cluster
block (and its rationale) in the same commit.

Line numbers refer to the Wine version recorded in the baseline header. A
`Sites:` line lists every baseline site of the cluster as `<line>=<class>`;
prose explains why. One source line can fire many assertions and can mix
sub-causes — a `real` line may carry a by-design remainder (noted in prose),
per the mixed-site rule: if any intend-to-fix component remains on a line,
the line is `real`.

Audit provenance: every cluster below was re-derived on 2026-07-20 from the
Wine test source, the raw actual-vs-expected failure messages
(`MTLD3D_CONFORMANCE_RAW_DIR`), and the implementation — independently
re-checked before retagging. Headline: **4 `real` · 93 `expected` ·
3 `caps` · 22 `ceiling` · 3 `flaky` · 0 `untriaged`** unique sites; all 8
subtest-arches `crash=0`. (2026-08-27: device.c:15088 moved from `expected`
to `ceiling`, it fires only where the Wine build ships a loadable d3d12.dll;
the SRGBTEXTURE decode landing the same day changed no site counts — the
newly-running `srgbtexture_test` passes. 2026-08-28: honouring
`D3DCREATE_NOWINDOWCHANGES` dropped test_window_style 5215, and test_wndproc
4551 was re-derived from the raw capture and corrected from `real` to
`expected`. Multisampling then moved the counts in both directions. Four
`device.c/test_reset` sites now pass, because a fullscreen `Reset` with a
zeroed `D3DPRESENT_PARAMETERS` is rejected for its `D3DFMT_UNKNOWN`
back-buffer format. Seven `visual.c` tests stopped skipping (every one of
them gates on `CheckDeviceMultiSampleType`) and five of the seven pass
outright; the clusters below cover what the other two and the tests they
unblocked leave failing, including the four `real` sites they add.) Only two
tags change what the gate tolerates: `flaky` (count changes in either
direction) and `ceiling` (reads below the pin). Every other tag is
documentation, so a correction between `real`, `expected` and `caps` is never
a gate change.

#### Desktop mode switching, and how fullscreen honors the requested size

A fullscreen device takes a borderless window over the monitor and never
changes the display mode. Wine's mac driver would hand a mode change to
`CGDisplaySetDisplayMode`, reconfiguring the user's screen and rearranging
every other window, and dodging that would depend on the `EmulateModeset`
registry setting the user has to know about. It also leaves the z-order alone:
raising the window to the topmost level deadlocks winemac (see
test_window_style 5220).

When the app requests an enumerable display mode, the back buffer keeps it,
exactly as it would under a real mode-set, and present scales it to the
drawable (MetalFX when enlarging, the same resample `render.scale` rides).
That keeps the D3D9 half of the contract at the requested mode: the default
viewport and scissor, the reported present parameters, and the device's and
swap chain's `GetDisplayMode` all agree with the size the app rendered for.
(Until 2026-08 the back buffer instead followed the monitor-covering window;
apps that sized their viewport from their own request rendered into a
corner.) A request that matches no enumerable mode still follows the window:
native would reject it, so nothing can depend on it being honored, and the
apps that make such requests (WoW's windowed-to-fullscreen toggle carries its
window size) size their rendering and mouse handling from the window — the
window-sized back buffer is the assignment that keeps them consistent.

What the emulation cannot deliver is the Win32 half of a real mode-set:
the desktop mode, `GetSystemMetrics`, `GetMonitorInfo`, and window rects
derived from them keep the monitor's native resolution, and mouse input
arrives in window coordinates rather than mode coordinates. Every site
asserting that half — the desktop mode following a create, `GetSystemMetrics`
reporting it, the device window adopting the mode's rect — is `expected`
under this decision, as is rejecting a non-enumerable mode (we do not
validate against a mode list because no mode is set). The harness pins
emulated mode switching and Retina mode so window-management assertions use a
stable physical-pixel coordinate space.

Two refinements landed 2026-08 after the CI runner exposed them (its virtual
display accepts the mode changes this machine's macdrv rejects, so the tests
walk further):

- **One source of display truth.** `EnumAdapterModes` / `GetAdapterDisplayMode`
  seed from `EnumDisplaySettingsW(ENUM_CURRENT_SETTINGS)` — the same view
  win32u validates `ChangeDisplaySettingsW` against and derives
  `GetMonitorInfoW` from — instead of `NSScreen`. On this machine the two
  agree under the pinned Retina mode; on the runner's virtual display they
  disagreed by exactly 2x (Win32 2048x1536, `NSScreen` 1024x768), which split
  `GetDisplayMode` from the monitor rect (test_get_display_mode 14472/14474)
  and fed the tests modes that user32 then refused.
- **Registry-mode restore.** The one direction of the mode contract compatible
  with never modesetting: when a fullscreen device loses focus
  (`WM_ACTIVATEAPP FALSE`) or leaves fullscreen (windowed `Reset`, final
  release) and the current mode differs from the registry mode — the app
  changed it through user32, we never did — the device puts the registry mode
  back (`ChangeDisplaySettingsW(NULL, 0)`), as native does. Locally this is
  dormant (the tests' own mode changes fail first, so current always equals
  registry); on the runner it covers test_wndproc 4261/4302/4328/4329 and
  test_mode_change 5563/5593. A fullscreen `Reset` still does not modeset, so
  the sites asserting the mode follows a Reset stay `expected`.

### The `real` backlog (empty)

No site is currently classified `real`. Every failing site is a recorded
decision (`expected`), a capability we do not advertise (`caps`), a pin that
reads zero on other hardware (`ceiling`), or a known flap (`flaky`), each with
its rationale in the per-cluster section below. A site returns to `real` the
moment a divergence is found that we intend to close.

The `device` subtest used to die silently inside test_volume_get_container
(a `GetContainer` that answered E_NOINTERFACE with a null container, which
the test then released), and the baseline recorded before the runner learnt
to treat a missing end-of-run summary as a crash carried only the sites
before that point. Every cluster from test_occlusion_query on was
re-triaged when the run first reached its end again.

Vertex streams 1..15 and `SetStreamSourceFreq` instancing are implemented, so
the clusters that used to sit on "single-stream rendering" (stream_test,
fixed_function_decl_test, the stream-1 half of test_sysmem_draw, and the
state-block stream capture in resource_check_data) no longer appear in the
baseline.

### device.c clusters

### device.c/test_wndproc
Sites: 4207=expected 4212=expected 4214=expected 4219=expected
Sites: 4223=expected 4248=expected 4257=expected 4293=expected
Sites: 4298=expected 4319=expected 4340=expected 4410=expected 4420=expected
Sites: 4424=expected 4432=expected 4487=expected 4525=expected 4545=expected
Sites: 4572=expected 4161=ceiling 4231=ceiling 4551=expected 4475=flaky
Sites: 4480=flaky

4161/4231 are the test's own `ChangeDisplaySettingsW(CDS_FULLSCREEN)` call
failing before any D3D9 object is involved (`ceiling`: on a display that
accepts the CDS — a CI runner's — they read zero). With `EmulateModeset=Y`, Wine does
not deliver the requested desktop mode. No mtld3d code participates. The rest
of the fullscreen focus lifecycle we
deliberately do not drive: no focus/foreground mutation (4212/4214), no focus-
window subclass (4223/4572), no WM_* activation/mode message generation
(4207/4248/4293/4319/4340/4410/4432/4525/4545), no focus-window minimize
(4420), device-never-lost TestCooperativeLevel (4257/4298/4424/4487).
Caveat on 4219: it fails because OUR cursor wndproc subclass replaced the
device window's proc — a deliberate, load-bearing hook we keep (cursor
realization), not a missing feature.

4257/4298/4424/4487 are the kept device-loss divergence, not an unwritten
stub: no exclusive mode is ever taken, so nothing is ever lost, and
`TestCooperativeLevel` answers `D3D_OK` across a focus change. The
transition is detectable (the device window's subclass already handles
`WM_ACTIVATEAPP` for the registry-mode restore), so this is a decision
rather than a gap: reporting a loss that did not happen sends every
fullscreen game through releasing and rebuilding its whole `D3DPOOL_DEFAULT`
working set on each activation change, which costs frame time and risks the
game's own recreate path, for a device that lost nothing. The
`D3DERR_DEVICENOTRESET` half is real and implemented: a failed `Reset`
latches it until one succeeds. Also listed in the README's kept divergences.

4551 is `expected`, and follows from the same no-modeset decision as the
message sites above. It reads a `WINDOWPOS` the test's wndproc only captures
once the expected-message walk reaches the fifth entry of
`mode_change_messages_hidden`, and the walk stops one entry earlier, on the
`WM_SIZE` the device window never receives: a fullscreen mode-change `Reset`
resizes the back buffer, not the window, which already covers the monitor
and keeps covering it, so its client rect is unchanged and user32 sends no
`WM_SIZE`. 4525/4545 record that stall directly (both raw failures read
`Expected message 0x5`), which leaves the capture zeroed and the assertion
comparing against a null HWND. Reaching it needs a real mode-set, so the
line moves only with that decision.

4475/4480 are flaky macdrv window-message timing sites;
mtld3d does not call `SetWindowPos` or `MoveWindow` on those paths.

### device.c/test_reset
Sites: 2126=expected 2127=expected 2179=expected 2180=expected
Sites: 2234=ceiling 2237=ceiling 2238=ceiling 2250=ceiling
Sites: 2251=ceiling

Everything fullscreen in this cluster follows from one decision: we never
change the desktop mode. The back buffer honors the resolution it was asked
for, so the request-side assertions (the default viewport matching the
request at 2133/2134 and 2172/2173, `GetPresentParameters` reporting it at
2187/2189) pass; what remains `expected` is the Win32 half:

- 2126/2127, 2179/2180, 2250/2251 read `GetSystemMetrics(SM_CXSCREEN)` and
  expect the requested mode. The desktop keeps its own resolution.
- 2234/2237/2238 are the test's own `ChangeDisplaySettingsW` call failing,
  before any D3D9 object is involved. Not ours to implement. These and
  2250/2251 are `ceiling`: a CI runner's display accepts the CDS and they
  read zero there.

The fullscreen Resets to a non-enumerable mode (32x32, 801x600) now return
INVALIDCALL, for a reason that has nothing to do with the resolution: each
zeroes its whole `D3DPRESENT_PARAMETERS`, so `BackBufferFormat` is
`D3DFMT_UNKNOWN`, which a fullscreen Reset has to reject. The resolution
itself is still not validated against a mode list, since no mode is set.

The windowed API contract in this test passes: Reset rejects an outstanding
app reference to a DEFAULT-pool resource or an implicit surface, and a
failed Reset latches DEVICENOTRESET until one succeeds.

### device.c/test_scissor_size
Sites: 3685=expected 3700=expected

The default scissor rect must equal the back buffer the app asked for, both
after create (3685) and after a Reset (3700). Every window in this test is
created `WS_MAXIMIZE`, and a maximized window is sized by the window manager
rather than the app, so we take its client rect and ignore the requested size
— the same rule as fullscreen, for the same reason. The scissor itself is
correct: it matches the back buffer we actually created.

Note 3700 additionally expects the *full screen* size, while a maximized
window's client rect is the work area (screen minus menu bar and Dock), so
this line would differ even if the create path honoured the request.

### device.c/test_wndproc_windowed
Sites: 4681=expected 4697=expected 4701=expected 4708=expected 4751=expected
Sites: 4774=expected 4778=expected 4785=expected

4701/4778 expect the focus window subclassed in fullscreen (we don't).
The other six expect the device window's wndproc UNCHANGED and fail because
of our cursor subclass — the same deliberate hook as test_wndproc 4219,
kept on purpose (cursor realization is driven from it).

### device.c/test_reset_fullscreen
Sites: 4871=expected

WM_ACTIVATEAPP delivery on a windowed→fullscreen Reset; we do not
force-show/activate the window.

### device.c/test_fpu_setup
Sites: 5041=expected 5051=expected

i686 only. Native D3D9 rewrites the x87 control word to single precision
(0x7f) at device creation and keeps it for callbacks; we deliberately never
touch the FPU control word. On x86_64 the same checks are todo_wine (free).

### device.c/test_window_style
Sites: 5220=expected

5220 is `expected`: the fullscreen extended style must carry `WS_EX_TOPMOST`.
We deliberately leave the z-order alone, because raising a window to the
topmost level makes Wine's mac driver re-derive the Cocoa window's level and
parent while holding winemac's per-window lock and hop to the main thread to
do it; a focus event arriving meanwhile re-enters `NtUserSetWindowPos` on
another thread and the process deadlocks. Reproduced in the `visual` subtest.
A borderless window covering the monitor already presents as fullscreen, so
the z-order buys nothing. (5200, the window-rect adoption, now passes.)

5215 passes since `D3DCREATE_NOWINDOWCHANGES` is honoured: a device created
with that flag leaves the device window's style, rect and visibility to the
app, in fullscreen as in windowed mode, so a window the app kept hidden is
still hidden after the fullscreen round trip. The three todo_wine lines that
the flag also covers (5179/5197/5238) now succeed inside their todo blocks,
which the runner does not count.

### device.c/test_mode_change
Sites: 5509=ceiling 5533=ceiling 5537=ceiling 5542=expected 5584=ceiling
Sites: 5602=ceiling 5622=ceiling 5636=ceiling 5639=ceiling 5646=ceiling
Sites: 5662=expected 5671=ceiling 5674=ceiling

Desktop display-mode-change lifecycle (`ChangeDisplaySettingsW` success,
`EnumDisplaySettings` reflecting changes/restores, fullscreen window resize).
All fail under the same decision: physical mode switching is disabled. Most
are `ceiling` rather than `expected` because they read zero on a CI runner,
whose display accepts the CDS calls and whose registry-mode restore then
satisfies the later assertions; 5542 and 5662 fail there too and stay
`expected`. 5552/5554 (the back buffer must keep the size a fullscreen create
asked for across an external mode change) pass now that the back buffer
honors the request and never follows the window.

### device.c/test_device_window_reset
Sites: 5968=expected

The device window must adopt the *requested mode's* rect across a fullscreen
Reset. Ours adopts the monitor rect instead, since there is no mode. (5951 and
5971, which check that the window covers the screen at all, now pass.)

### device.c/test_occlusion_query
Sites: 6780=expected

The >2^32-sample query (65 fullscreen 8192x8192 quads under one query, depth
test off) undercounts because Apple's TBDR hidden-surface removal merges
same-encoder opaque overdraw before fragment processing: the visibility
counter reports the samples that *survive* HSR, not every sample that would
have passed the depth test on an immediate-mode GPU. Proven by
instrumentation, not inferred: the GPU-written slot value itself is short
(our BEGIN..END span is a single slot in a single frame, summed correctly),
the value is always an integer number of 8192-wide quad ROWS (0x1de98f00 =
8192 x 61260; an earlier environment read 0x077a63c0 = 8192 x 15315), and
tile-row-granular partial renders decide how many overdraw layers escape
culling, which is why the number moves between environments. The same test's
single-quad section counts bit-exactly (0x75cf00 = one 3456x2234 quad), so
the machinery is precise whenever HSR has nothing to merge. Counting all
overdraw layers would need an encoder per draw under active queries,
destroying pass batching; the kept optimization is single-encoder pass
batching, so this is `expected`. Real-game occlusion (a bounding box tested
against a populated depth buffer, read as zero/non-zero) is unaffected.

### device.c/test_lockrect_invalid
Sites: 8664=expected 8682=expected 8701=expected

We PASS the accept-invalid lock checks (the `broken()`-guarded Win7 reject
alternative is not what we take). These offset assertions then compare our
returned pointer against blind `top*pitch + left*bpp` arithmetic on the
invalid rect. `parse_rect` clamps invalid rects (negatives to 0,
inverted/zero-area to the full mip), so our offsets differ; matching XP
exactly would require handing out pointers OUTSIDE the staging allocation,
which the lock-safety model forbids (`lock_region_ptr` bounds assert).
Deliberate safety tradeoff, kept. (Cube's garbage offsets are pointer diffs
across unrelated per-lock allocations: meaningless, not out-of-bounds.)

### device.c/test_pinned_buffers
Sites: 10074=expected 10079=expected

The test expects a DISCARD re-lock to return the same pinned pointer with
prior contents intact, a driver-specific optimization probe with no cap
branch. Our rename-on-DISCARD model returns fresh backing by design, and
DISCARD contents are spec-undefined, so our behavior is legal. Intent to
keep (the rename model is core).

### device.c/test_lost_device
Sites: 12144=expected 12146=expected 12153=expected 12155=expected
Sites: 12199=expected

Focus-loss/device-lost lifecycle: TestCooperativeLevel/Present/Reset must
report DEVICELOST/DEVICENOTRESET across a fullscreen focus cycle. Our
device is never lost by design (no exclusive fullscreen, no GPU loss on
Metal). The only non-OK `TestCooperativeLevel` answer we give is the
DEVICENOTRESET latch a failed `Reset` leaves behind, which is the windowed
API contract test_reset exercises, not focus-driven loss.

### device.c/test_check_device_format
Sites: 12689=expected 12694=expected

CheckDepthStencilMatch(..., D3DFMT_D32): native returns NOTAVAILABLE; we
return D3D_OK because D32 genuinely maps to Depth32Float and works. We
advertise MORE than native here, deliberately; not an omitted-cap (`caps`)
case, and our answer is truthful for our backend.

### device.c/test_miptree_layout
Sites: 12784=expected 12823=expected

The test asserts each mip's lock pointer sits at a contiguous offset from
level 0 (single-allocation mip chain). Our staging is one PageBox per mip,
which is load-bearing for the rename-at-overlap versioning model (each
mip's Arc swaps independently); a contiguous chain is structurally
incompatible with that design, which we keep. Site 12823 is the same pointer
layout assertion across six cube faces and their mip levels. Cube staging is
also one PageBox per subresource so a face or mip can rename independently.
Per-subresource pixel data is correct.

### device.c/test_resource_access
Sites: 13838=caps

"Test 2D 6" creates a DEFAULT-pool, `D3DUSAGE_DEPTHSTENCIL` texture in the
device's depth format and derives the expected HRESULT from
`CheckDeviceFormat(usage = 0, D3DRTYPE_TEXTURE, depth format)`: on every
desktop driver a depth format that works as a depth-stencil texture also
works as a plain (usage 0) texture, so the test treats the two queries as
one capability. They are two capabilities here. We expose depth textures
only with `D3DUSAGE_DEPTHSTENCIL` (the shadow-map idiom: bind as depth, then
sample), never as plain textures (no mip chains, no lockable levels), and
each query answers for its own usage: the usage-0 query says NOTAVAILABLE
and a usage-0 create fails with it, the DEPTHSTENCIL query says OK and the
DEPTHSTENCIL create succeeds with it. The test's inference from the first
answer to the second create is the cap-blind step; our responses are each
the conformant one for the capability set we advertise.

### device.c/test_get_display_mode
Sites: 14383=expected 14384=expected 14480=expected 14482=expected
Sites: 14491=expected 14493=expected

All one decision. A fullscreen device's and swap chain's `GetDisplayMode`
report the honored mode (14378/14379, 14390/14391 pass) and a windowed
device's reports the desktop mode, so the windowed halves of
14480/14482 and 14491/14493 pass too. What remains: `GetAdapterDisplayMode`
must report the *switched* desktop mode after a fullscreen create
(14383/14384), and the fullscreen halves of 14480-14493 expect
`GetDisplayMode` to equal the monitor rect because a real mode-set would
have shrunk the monitor to the mode. We change no mode, so the monitor keeps
its native size while the device reports the mode it honored.

Raw output confirms the remaining halves: 14480-14493 fail only as
"Adapter 0 test 1" (the CREATE_DEVICE_FULLSCREEN iteration), "Expect width
3456, got 640", deterministic, not a GetMonitorInfoW environment cascade.
Their sibling sites 14451/14454 and 14472/14474 pass with the harness's
pinned Retina mode.

### device.c/init_d3d9on12_modules
Sites: 15088=ceiling

`win_skip("Direct3DCreate9On12 is not supported…")`: under Wine, win_skip
counts as a test failure. We don't provide the D3D9-on-D3D12 bridge; N/A on
Metal. Ceiling, not expected, because the site only fires where the Wine
build ships a loadable d3d12.dll: the win_skip sits after the three
LoadLibrary calls, and a failed load takes a plain `skip()` that counts
nothing. The pinned CI release is built with Vulkan and reads 1; the current
local dist is built without (no winevulkan.dll, no i386 d3d12.dll at all)
and reads 0. Not test-source drift: `dlls/d3d9/tests/` is identical between
the two builds.

### device.c/test_d3d9on12
Sites: 15160=expected

`win_skip("Failed to load d3d9on12 modules…")`: the companion to 15088,
same D3D9-on-D3D12 rationale. `expected`, not `ceiling`, because it fires
on both kinds of build: this skip is the module-load failure itself, which
under a Vulkan-less build happens one dll earlier but still lands on this
line's win_skip in `test_d3d9on12`.

### visual.c clusters

### visual.c/z_range_test
Sites: 3887=expected 3889=expected 3891=expected 3894=expected
Sites: 3963=expected 3965=expected

All six depend on a depth clear (0.75) written BEFORE a Present surviving
into later frames with ZWRITE off. Store-action Rule B flips the auto DS
store to DontCare at Present — the deliberate TBDR depth-store elision (the
preserve fix was implemented and reverted to keep the optimization). The
broken() r500 alternatives are ignored by the runner; the primary
assertions need cross-Present depth.

### visual.c/texdepth_test
Sites: 5360=expected 5398=expected 5436=expected 5454=expected

The ps_1_4 depth-gradient math is correct (the same-frame cycle passes and
is absent here). The failing cycles read the gradient across Presents —
the same Rule B depth-store elision as z_range_test.

### visual.c/pixelshader_blending_test
Sites: 12008=expected

Renders into a one- or two-channel texture (G16R16, R16F, G16R16F, R32F,
G32R32F) with blending on, then samples it and expects the channels the
format does not store to read as 1.0 (`0x001820ff`, blue forced to `ff`). We
return `0x00182000`: the stored channels are exact, the missing ones read 0.
The 1.0 rule is implemented as a sampler swizzle on the texture view, and
Metal forbids `RenderTarget` usage on a swizzled view, so a render-target
texture is bound as its base texture and loses the swizzle when sampled. The
same trade already covers X8R8G8B8 render targets (`unix/unix/src/metal/
texture.rs`). Lifting it means carrying two handles per render-target
texture (base for attachment, swizzled view for sampling); worth doing only
if a workload samples its own single/dual-channel render target and relies
on the missing lanes. The four-channel members (A16B16G16R16F,
A32B32G32R32F) and L8 pass.

### visual.c/test_fetch4
Sites: 15617=caps 15668=ceiling 15824=ceiling 15829=ceiling

Fetch4 is an AMD vendor extension enabled via a magic FOURCC through
D3DSAMP_MIPMAPLODBIAS; DF16/DF24 are vendor depth-texture FOURCCs we map to
Depth32. We deliberately advertise none of it; our output is the correct
fetch4-off/format-absent result (accepted by the test only under broken()).
15668/15824/15829 counts wobble with display environment, hence `ceiling` —
keep the higher pin (a count-down is tolerated; a low pin makes the flutter-back a false
regression).

### visual.c/fp_special_test
Sites: 16433=expected

VS special-float ops on NaN/±inf: the test accepts four distinct vendor
results (r500/r600/nv40/nv50) plus broken(warp) — special-value handling is
GPU-defined, not spec-mandated. Our Metal GPU produces a fifth valid IEEE
result matching no vendor's encoding. Matching a specific vendor is neither
feasible nor desirable. No capability involved (old `caps` tag incoherent).

### visual.c/multisampled_depth_buffer_test
Sites: 17476=expected

The second half of the test binds a 2x multisampled depth surface while
render target 0 is the single-sampled back buffer. Metal takes a render
pass's sample count from its attachments and rejects a pass where they
disagree, so mtld3d drops the mismatched depth attachment and the draw runs
untested. D3D9 leaves this case to the driver, and the test says so itself:
it accepts one result from AMD and a different one from Nvidia, and does not
require depth testing to work at all. The matched half of the test (a 2x
depth surface beside a 2x render target) passes.

### visual.c/resz_test
Sites: 17724=real 17862=real

The RESZ hack (`SetRenderState(D3DRS_POINTSIZE, 0x7fa05000)`) resolves the
bound depth buffer into an INTZ texture, and mtld3d implements it as a
depth-to-depth copy. From a *multisampled* depth surface that copy is not
what RESZ means: the samples have to be resolved, which on Metal is a render
pass with a depth resolve attachment rather than a blit. The colours the
test reads back are close to the expected gradient but wrong per pixel. The
single-sampled RESZ path is unaffected and covered by the end-to-end suite.

### visual.c/add_dirty_rect_test
Sites: 19210=expected 19217=expected 19232=expected

The surviving sites require STALE data to be shown: a NO_DIRTY_UPDATE lock
must NOT be uploaded (19210/19217), and after AddDirtyRect only the dirty
sub-rect may refresh (19232). Our design uploads whole mips eagerly with
self-tracked dirtiness and treats AddDirtyRect as a no-op — we show fresher
data than required. Deliberate; the READONLY-first-lock upload defect that
used to live here (19156/19163) is fixed.

### visual.c/test_multisample_mismatch
Sites: 20880=expected 20883=expected 20959=expected 20962=expected

The whole test draws with a multisampled render target beside a
single-sampled depth buffer and the other way round. Metal rejects a render
pass whose attachments disagree on sample count, so mtld3d drops the
mismatched depth attachment; the draws land but the depth test does not
gate them. Same rationale as multisampled_depth_buffer_test 17476, and the
same evidence that D3D9 never defined the case: every assertion here carries
a second accepted colour under `broken()`, and the comments in the test
record that AMD and Nvidia disagree about whether the draw happens at all.

### visual.c/test_flip
Sites: 22053=expected 22055=expected 22064=expected 22066=expected
Sites: 22072=expected

The device is created with D3DSWAPEFFECT_DISCARD, under which post-Present
backbuffer contents are UNDEFINED by spec; the test observes native's
incidental flip-chain content rotation. Not emulating that is
spec-compliant. Surface identity and lockable read-back pass. A title
relying on flip-chain read-back under FLIP/COPY swap effects would be a
different (real) matter.

### visual.c/test_max_index16
Sites: 24133=expected 24135=expected

Metal treats index 0xffff as the un-disableable uint16 primitive-restart
sentinel, dropping the triangle that uses it; the test additionally writes
vertex 0xffff OUTSIDE its lock (UB, may never reach the GPU). broken(warp)
shows even the MS reference rasterizer fails this; the runner ignores
broken(). Accepted platform limitation (no cap branch — old `caps` tag was
wrong).

### visual.c/test_map_synchronisation
Sites: 25148=flaky

The failing config is exactly the plain (no DISCARD/NOOVERWRITE) PARTIAL
lock of a contended Direct buffer, which native stalls for. Our buffer-
rename design deliberately removed that stall (`plan_lock` → WriteInPlace);
re-adding it is the only fix and is a rejected perf regression. Whether the
test observes the divergence is a per-run CPU-vs-GPU race (the probe's Lock
write lands before or after the GPU consumes the in-flight draw), so the
count flutters between 0 and 1 across runs of the same binary; it read 0 on
the CI runner and tripped the stale-baseline gate (PR #12), hence the flaky
tolerance. That race is in the observation only, not in the decision:
`plan_lock` is a pure function of a `coherent_seq` its caller read once
with Acquire before calling, and only the unix side raises that counter
(`fetch_max`, on GPU retirement), so a stale read can turn a legal
in-place write into a needless rename but never the reverse. The kept
divergence itself is unchanged.

### visual.c/test_alpha_to_coverage
Sites: 26538=caps

`win_skip("Alpha to coverage is not supported.")`, which counts as a failure
under Wine. Alpha to coverage is reached through a vendor pseudo-format
(NVIDIA's `ATOC` through `D3DRS_ADAPTIVETESS_Y`, AMD's `A2M1` through
`D3DRS_POINTSIZE`); mtld3d advertises neither, and answering NOTAVAILABLE for
the probe is the conformant response for a device without the extension. The
test only reaches the probe on a multisample-capable device, which is why the
site is new.

### visual.c/test_mipmap_upload
Sites: 27550=expected

The app writes the whole mip chain through a single level-0 lock pointer;
with per-mip PageBox staging the upper mips never receive the data. Same
architecture-we-keep rationale as test_miptree_layout — but this is the
weakest `expected` in the file: it produces wrong rendered pixels for a
real-app pattern (Wine cites shipped titles). If the per-mip staging
commitment is ever softened, retag `real` first.

### visual.c/test_default_attribute_components
Sites: 27902=expected

FLOAT→unorm rounding at exactly .5: Metal rounds 76.5 up (77), refrast
truncates (76). A ±1 GPU rounding-convention difference with no cap branch;
mimicking refrast exactly is not feasible or desirable.

### stateblock.c clusters

### d3d9ex.c clusters

### d3d9ex.c/START_TEST
Sites: 5184=expected

`win_skip("Failed to get address of Direct3DCreate9Ex")` — win_skip counts
as a failure under Wine, and START_TEST returns immediately, so no d3d9ex
test ever runs. We deliberately don't export Direct3DCreate9Ex (D3D9Ex out
of scope; target titles use plain D3D9). (Previously mis-attributed to
test_scene, which never executes.)
