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
make conformance           WINE_BUILD=<wine-build-tree>   # diff vs baseline.txt
make conformance-baseline  WINE_BUILD=<wine-build-tree>   # (re)record baseline.txt
```

Set `MTLD3D_CONFORMANCE_RAW_DIR=<dir>` to also persist each subtest's full raw
output to `<dir>/<arch>-<subtest>.log`. The normal run reduces output to per-site
counts and drops the assertion text; the raw logs keep every
`<file>.c:<line>: Test failed: Got <actual>, expected <expected>` message (plus
the Metal-validation lines), which is what the per-cluster audit below was built
from — the *actual-vs-expected* values distinguish a real defect from an
acceptable `caps` difference. Off unless the variable is set.

`WINE_BUILD` is the only conformance-specific input: it is the Wine *build tree*
that has compiled `dlls/d3d9/tests/{i386,x86_64}-windows/d3d9_test.exe`. The
binaries are **not** vendored — they are large and drift with the Wine version,
so `baseline.txt` records the Wine version it was taken against instead. (The
runner finds the wine loader itself via the global `WINE_SDK` install that holds
our builtin; that variable is already mandatory for the whole Makefile, so it is
not a conformance-specific knob.)

The runner is the Rust tool `mtld3d-conformance` (`unix/conformance/`). It runs
each subtest as its own process — so a crash in one cannot poison another's
counts — with the Metal debug layer disabled (`MTL_DEBUG_LAYER=0`) so a
validation abort cannot mask the failure counts, and with our logs and Wine's
debug channels silenced.

This is **not** part of `make test`: many checks fail by design (see below), so
it is a tracked-score tool, not a pass/fail gate. The runner exits non-zero only
on a *regression* vs the baseline — a per-site failure count that went up, a new
failing site, or a subtest that started crashing. Improvements (a count that
dropped, a site that disappeared, a crash that cleared) are reported but do not
fail the gate; they are a cue to re-record the baseline.

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
  The runner loads classes from here (for the flaky tolerance and untriaged
  reporting); a unit test in the runner crate fails `make test` unless the
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
  desktop mode switching), a kept perf tradeoff (the TBDR depth-store elision,
  buffer-rename over stalls), or an accepted platform limitation (Metal's
  0xffff primitive restart, GPU-defined NaN encodings). "We don't want to fix
  it" or "the fix is invasive" is not a rationale — without a positive reason
  to keep the divergence, the site is `real`.
- **`flaky`** — environmental/non-deterministic (display config, Retina scale,
  macdrv window-manager timing). Count changes in either direction never gate.
  Tag reactively — only once a flutter actually trips the gate — and pin the
  HIGHER observed count so a flutter back up is not a false regression.
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
re-checked before retagging. Headline: **46 `real` · 134 `expected` ·
6 `caps` · 2 `flaky` · 0 `untriaged`** unique sites; all 8 subtest-arches
`crash=0`. Tags do not affect the gate (which rejects count/site/crash
regressions), so tag corrections are documentation, not gate changes.

### The `real` backlog (13 distinct defects behind the 46 sites)

| defect | cluster(s) | sites |
|---|---|---:|
| SetStreamSourceFreq/GetStreamSourceFreq are INVALIDCALL stubs (state round-trip, independent of instancing) | stream_test | 24 |
| Reset: no outstanding-DEFAULT-pool / implicit-surface-ref rejection | test_reset | 4 |
| TestCooperativeLevel: no DEVICENOTRESET latch after a failed Reset | test_reset | 1 |
| StateSnapshot (D3DSBT_ALL) never captures the stream-0 vertex buffer | resource_check_data | 3 |
| Reset does not re-show a device window whose WS_VISIBLE was cleared | test_wndproc, test_window_style | 3 |
| Clears ignore D3DRS_SRGBWRITEENABLE (draw path honors it) | clear_test | 2 |
| FF lighting renders black for default-light/world-matrix cases | lighting_test | 1 |
| ProcessVertices is an INVALIDCALL stub | test_sysmem_draw | 2 |
| Depth→depth StretchRect is an S_OK no-op | depth_blit_test | 1 |
| CheckDeviceFormatConversion reuses the present predicate; wrong for R5G6B5→X8R8G8B8 | test_format_conversion | 1 |
| Cube-cap-off rejection shape too permissive (MANAGED/SYSTEMMEM accepted) | test_cube_textures | 2 |
| CreateTexture(depth) succeeds while our own CheckDeviceFormat denies it | test_resource_access | 1 |
| Occlusion query undercounts a >2^32-sample span (cause unproven) | test_occlusion_query | 1 |

Coupling note for the stream-frequency fix: implementing the Set/Get round-trip
un-gates the instanced draws behind it, so the by-design instancing pixel sites
(12423/12464/12466/12468/12470) will rise in count — re-baseline in the same
commit.

### device.c clusters

### device.c/test_wndproc
Sites: 4161=expected 4207=expected 4212=expected 4214=expected 4219=expected
Sites: 4223=expected 4231=expected 4248=expected 4257=expected 4293=expected
Sites: 4298=expected 4319=expected 4340=expected 4410=expected 4420=expected
Sites: 4424=expected 4432=expected 4487=expected 4525=expected 4545=expected
Sites: 4572=expected 4551=real 4568=real 4475=flaky 4480=flaky

Fullscreen focus/mode lifecycle we deliberately do not drive: no desktop
mode switch (4161/4231), no focus/foreground mutation (4212/4214), no focus-
window subclass (4223/4572), no WM_* activation/mode message generation
(4207/4248/4293/4319/4340/4410/4432/4525/4545), no focus-window minimize
(4420), device-never-lost TestCooperativeLevel (4257/4298/4424/4487).
Caveat on 4219: it fails because OUR cursor wndproc subclass replaced the
device window's proc — a deliberate, load-bearing hook we keep (cursor
realization), not a missing feature. 4551/4568 are `real`: after Reset a
device window whose WS_VISIBLE was cleared must be re-shown
(SWP_SHOWWINDOW); the test cites a real title relying on it. 4475/4480 are
the only flaky pins: macdrv WM timing, no SetWindowPos/MoveWindow anywhere
in our code.

### device.c/test_reset
Sites: 2126=expected 2127=expected 2179=expected 2180=expected 2234=expected
Sites: 2237=expected 2238=expected 2250=expected 2251=expected 2519=expected
Sites: 2521=expected 2529=expected 2531=expected
Sites: 2370=real 2372=real 2496=real 2498=real 2541=real

The expected half is the fullscreen mode environment: screen-resolution
asserts after fullscreen create/Reset (2126–2251) and fullscreen Reset to
non-enumerable modes 32x32/801x600 (2519–2531) — with no exclusive display
modes, any backbuffer size is valid for us, so accepting is internally
consistent. The real half is windowed API contract, not environment:
Reset must return INVALIDCALL with an outstanding DEFAULT-pool surface
(2370) or a held implicit-backbuffer reference (2496), with
TestCooperativeLevel reporting DEVICENOTRESET afterwards (2372/2498); and
a failed Reset (0x0 — which we do reject) must latch DEVICENOTRESET until
a successful Reset (2541). `device_test_cooperative_level` hardcodes S_OK.

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
Sites: 5200=expected 5220=expected 5215=real

5200: fullscreen window-rect adoption we don't perform. 5220: fullscreen
extended-style (TOPMOST) management. 5215 is `real`: the windowed-Reset
re-show contract (WS_VISIBLE) — same defect as test_wndproc 4551/4568.

### device.c/test_mode_change
Sites: 5509=expected 5533=expected 5537=expected 5542=expected 5584=expected
Sites: 5602=expected 5622=expected 5636=expected 5639=expected 5646=expected
Sites: 5662=expected 5671=expected 5674=expected

Desktop display-mode-change lifecycle (ChangeDisplaySettingsW success,
EnumDisplaySettings reflecting changes/restores, fullscreen window resize).
We never switch the desktop mode by design (CAMetalLayer).

### device.c/test_device_window_reset
Sites: 5951=expected 5968=expected 5971=expected

Fullscreen device-window resize to the full screen rect across Reset; not
performed by design.

### device.c/test_occlusion_query
Sites: 6780=real

A query spanning ~2^32+ samples returns a genuine undercount (this run:
0x1de98f00 — a non-integer multiple of one fullscreen quad). The test
accepts the exact 64-bit count or 32-bit saturation; our own fallback paths
(slot exhaustion → u32::MAX) would have passed the saturation clause, so
the undercount is unexplained. No capability branch is involved, so the old
`caps` tag was wrong. Tagged `real` pending investigation of the visibility
span/slot summation; if the undercount is proven intrinsic to Metal
visibility counting, retag `expected` with that evidence.

### device.c/test_cube_textures
Sites: 7866=real 7868=real

We do not advertise D3DPTEXTURECAPS_CUBEMAP, and the test's cap-off branch
asserts a cube-less device rejects ALL cube creates with INVALIDCALL. We
correctly reject DEFAULT (7864 passes) but accept MANAGED/SYSTEMMEM as
CPU-only shells — a too-permissive rejection shape vs a native cube-less
device. Faithful fix: reject non-SCRATCH pools while the cap is off
(SCRATCH is asserted creatable, 7871, and passes).

### device.c/test_lockrect_invalid
Sites: 8664=expected 8682=expected 8701=expected

We PASS the accept-invalid lock checks (the `broken()`-guarded Win7 reject
alternative is not what we take). These offset assertions then compare our
returned pointer against blind `top*pitch + left*bpp` arithmetic on the
invalid rect. `parse_rect` clamps invalid rects (negatives→0,
inverted/zero-area→full mip), so our offsets differ; matching XP exactly
would require handing out pointers OUTSIDE the staging allocation, which
the lock-safety model forbids (`lock_region_ptr` bounds assert). Deliberate
safety tradeoff, kept. (Cube's garbage offsets are pointer diffs across
unrelated per-lock allocations — meaningless, not out-of-bounds.)

### device.c/test_pinned_buffers
Sites: 10074=expected 10079=expected

The test expects a DISCARD re-lock to return the same pinned pointer with
prior contents intact — a driver-specific optimization probe with no cap
branch. Our rename-on-DISCARD model returns fresh backing by design, and
DISCARD contents are spec-undefined, so our behavior is legal. Intent-to-
keep (the rename model is core); previously mis-tagged `caps`.

### device.c/test_npot_textures
Sites: 10178=caps

The no-POW2 branch asserts CreateCubeTexture(EdgeLength=3) succeeds without
checking D3DPTEXTURECAPS_CUBEMAP — cap-blind. Our INVALIDCALL for the
DEFAULT pool is the correct cube-less answer (the cap-off branch of
test_cube_textures asserts exactly that at 7864). Note: fixing the
test_cube_textures rejection shape will make the MANAGED/SYSTEMMEM
iterations here fail too (count 1→3, still `caps`) — re-baseline together.

### device.c/test_lost_device
Sites: 12144=expected 12146=expected 12153=expected 12155=expected
Sites: 12199=expected

Focus-loss/device-lost lifecycle: TestCooperativeLevel/Present/Reset must
report DEVICELOST/DEVICENOTRESET across a fullscreen focus cycle. Our
device is never lost by design (no exclusive fullscreen, no GPU loss on
Metal). Unlike the test_reset real subset, these are all genuinely
focus-driven.

### device.c/test_check_device_format
Sites: 12689=expected 12694=expected

CheckDepthStencilMatch(..., D3DFMT_D32) — native returns NOTAVAILABLE; we
return D3D_OK because D32 genuinely maps to Depth32Float and works. We
advertise MORE than native here, deliberately; not an omitted-cap (`caps`)
case, and our answer is truthful for our backend.

### device.c/test_miptree_layout
Sites: 12784=expected

The test asserts each mip's lock pointer sits at a contiguous offset from
level 0 (single-allocation mip chain). Our staging is one PageBox per mip,
which is load-bearing for the rename-at-overlap versioning model (each
mip's Arc swaps independently); a contiguous chain is structurally
incompatible with that design, which we keep. Per-mip pixel data is
correct.

### device.c/test_resource_access
Sites: 13838=real 13853=caps

13838 ("Test 2D 6": DEFAULT pool, depth format, USAGE_DEPTHSTENCIL
texture): the test derives its expectation from OUR OWN CheckDeviceFormat,
which denies depth textures — yet the create succeeds. Internal
inconsistency between the capability report and the create path = defect
(either advertise or reject; decide with the usual never-advertise-what-we-
fail rule in mind). 13853 (CUBE 0/3/7, valid DEFAULT colour cubes): the
formula is cap-blind on CUBEMAP; our INVALIDCALL is the correct cube-less
response — `caps`.

### device.c/test_get_display_mode
Sites: 14383=expected 14384=expected 14451=expected 14454=expected
Sites: 14472=expected 14474=expected 14480=expected 14482=expected
Sites: 14491=expected 14493=expected

14383/14384: GetAdapterDisplayMode after a fullscreen 640x480 create must
reflect the switched desktop mode; we never switch (device/swapchain
GetDisplayMode correctly return 640x480 and pass). The rest are a
monitor-environment cascade: GetAdapterMonitor/GetMonitorInfoW can fail on
the conformance desktop (display-config dependent — absent in some runs),
poisoning the width/height comparisons downstream. GetAdapterMonitor itself
is implemented (MonitorFromPoint → primary).

### device.c/test_window_position
Sites: 14967=expected 14970=expected 14987=expected 15035=expected
Sites: 15053=expected

14987/15035/15053: fullscreen device window must fill the monitor rect
(create / Reset / activation); we don't reposition windows. 14967/14970:
the same GetAdapterMonitor/GetMonitorInfoW environment cascade as
test_get_display_mode (absent in some display configs).

### device.c/init_d3d9on12_modules
Sites: 15088=expected

`win_skip("Direct3DCreate9On12 is not supported…")` — under Wine, win_skip
counts as a test failure. We don't provide the D3D9-on-D3D12 bridge; N/A on
Metal. (This site was previously mis-clustered under test_window_position.)

### device.c/test_d3d9on12
Sites: 15160=expected

The `win_skip("Failed to load d3d9on12 modules…")` companion to 15088,
same rationale.

### visual.c clusters

### visual.c/lighting_test
Sites: 713=real

The world-matrix loop: a lit quad with a default light must render blue
(0x000000ff) under identity/singular/rotation matrices; we render BLACK for
all three (the non-affine black case passes trivially). No broken()/todo
escapes — the result is well-defined across drivers. This is a genuine FF
lighting defect (default-light parameters and/or normal transform), not the
"minor fidelity difference" it was previously filed as.

### visual.c/clear_test
Sites: 1473=real 1525=real

With D3DRS_SRGBWRITEENABLE on, Clear(0x7f7f7f7f) must produce the
sRGB-encoded 0xbbbbbb (asserted unconditionally; the CheckDeviceFormat
probe above feeds only a trace). Our draw pipelines honor sRGB write, but
the clear paths (loadAction fold and clear-quad) never consume it — we
output raw 0x7f. Same root for both: 1473 backbuffer, 1525 offscreen RT.
(Previously and inconsistently tagged caps/expected.)

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

### visual.c/fixed_function_decl_test
Sites: 9632=expected 9638=expected 9641=expected 9645=expected
Sites: 9651=expected 9654=expected

The failing draws source the color attribute from STREAM 1 (position from
stream 0) via the D3DCOLOR/UBYTE4N declarations; we render stream 0 only
(single-stream architecture, kept). The previous "two-sided-stencil /
color-ubyte switching" rationale was wrong about the mechanism.

### visual.c/stream_test
Sites: 12258=real 12261=real 12265=real 12275=real 12276=real 12278=real
Sites: 12280=real 12281=real 12283=real 12285=real 12286=real 12288=real
Sites: 12290=real 12291=real 12295=real 12296=real 12300=real 12393=real
Sites: 12398=real 12404=real 12410=real 12448=real 12452=real 12456=real
Sites: 12423=expected 12464=expected 12466=expected 12468=expected
Sites: 12470=expected

The real block is the Set/GetStreamSourceFreq STATE ROUND-TRIP: both entry
points are INVALIDCALL stubs, so every set/get/value assertion fails (the
sibling INVALIDCALL-expecting checks pass by accident of the stub). Storing
the frequency state is plain D3D9 API surface, independent of rendering
instancing — previously mis-filed under "single-stream architecture". The
expected block is actual instanced RENDERING output (INSTANCEDATA freq
dividers), which single-stream rendering deliberately does not implement.
See the coupling note in the backlog table: fixing the round-trip un-gates
the instanced draws and their counts will rise.

### visual.c/depth_blit_test
Sites: 14835=real

Depth→depth StretchRect returns S_OK but emits no GPU copy, so the
destination keeps its cleared depth and 12 of 16 probe pixels mismatch —
all within one frame (readback precedes the Present), so this is NOT the
Rule B family. The no-op exists because a naive copyFromTexture didn't
survive the bound-DS pass reload, i.e. "the naive fix was wrong" — not a
kept tradeoff (it buys no perf). Real: emit the copy and order it against
the deferred depth clear.

### visual.c/test_fetch4
Sites: 15617=caps 15668=caps 15824=caps 15829=caps

Fetch4 is an AMD vendor extension enabled via a magic FOURCC through
D3DSAMP_MIPMAPLODBIAS; DF16/DF24 are vendor depth-texture FOURCCs we map to
Depth32. We deliberately advertise none of it; our output is the correct
fetch4-off/format-absent result (accepted by the test only under broken()).
15668/15824/15829 counts wobble with display environment — keep the higher
pin (a count-down is tolerated; a low pin makes the flutter-back a false
regression).

### visual.c/clip_planes
Sites: 16129=expected 16131=expected

The test applies FF clip planes without branching on MaxUserClipPlanes (we
report 0). SetClipPlane/GetClipPlane are a CPU round-trip store with no GPU
application, consistent with the zero cap: a conformant app would not use
clip planes on this device. Deliberate scope decision. Becomes `real` the
moment a target title needs user clip planes.

### visual.c/fp_special_test
Sites: 16433=expected

VS special-float ops on NaN/±inf: the test accepts four distinct vendor
results (r500/r600/nv40/nv50) plus broken(warp) — special-value handling is
GPU-defined, not spec-mandated. Our Metal GPU produces a fifth valid IEEE
result matching no vendor's encoding. Matching a specific vendor is neither
feasible nor desirable. No capability involved (old `caps` tag incoherent).

### visual.c/add_dirty_rect_test
Sites: 19210=expected 19217=expected 19232=expected

The surviving sites require STALE data to be shown: a NO_DIRTY_UPDATE lock
must NOT be uploaded (19210/19217), and after AddDirtyRect only the dirty
sub-rect may refresh (19232). Our design uploads whole mips eagerly with
self-tracked dirtiness and treats AddDirtyRect as a no-op — we show fresher
data than required. Deliberate; the READONLY-first-lock upload defect that
used to live here (19156/19163) is fixed.

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
Sites: 25148=expected

The failing config is exactly the plain (no DISCARD/NOOVERWRITE) PARTIAL
lock of a contended Direct buffer, which native stalls for. Our buffer-
rename design deliberately removed that stall (`plan_lock` → WriteInPlace);
re-adding it is the only fix and is a rejected perf regression.

### visual.c/test_sysmem_draw
Sites: 25431=real 25436=real 25505=expected 25518=expected 25565=expected

25431/25436: ProcessVertices is an INVALIDCALL stub — unimplemented SW
vertex processing, no design rationale (real). 25505/25518/25565: the
colour attribute comes from stream 1 of a two-stream SYSTEMMEM declaration;
single-stream rendering by design. Note single-stream SYSTEMMEM draws
themselves work (those checks pass) — the old "SYSTEMMEM draw" rationale
was wrong.

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

### visual.c/test_format_conversion
Sites: 28024=real

Three rows fail, all expecting S_OK from CheckDeviceFormatConversion:
R5G6B5→X8R8G8B8 (no escape — every real driver converts this, and our own
StretchRect render-quad path CAN, so our NOTAVAILABLE is a false report =
the real component) plus YUY2→X8R8G8B8/R5G6B5 (broken_warp rows — our
NOTAVAILABLE is the honest answer for a device without YUV conversion; the
runner ignores broken()). Mixed line ⇒ real. The fix is the dedicated
conversion predicate (decoupled from `is_present_compatible`); note the
known coupling: test_display_formats asserts CheckDeviceType(windowed)
agrees, and the YUV blit tests gate on this predicate — change all three
consistently.

### stateblock.c clusters

### stateblock.c/resource_check_data
Sites: 1810=real 1812=real 1813=real

The 16-stream verification loop (VB pointer / offset / stride). Two causes
share these lines: (a) ~210 of 215 assertions are streams 1–15, which state
blocks deliberately don't capture (single-stream architecture — the
recorder path stores stream 0 only); (b) the intend-to-fix core: the
snapshot path (`StateSnapshot`, D3DSBT_ALL) captures the index buffer but
has NO stream-0 vertex-buffer/offset/stride field at all, so the two
CreateStateBlock(ALL)→Apply chains lose the stream-0 binding (5 assertions:
2 on 1810, 1 on 1812, 2 on 1813 — stream-0 failures confirmed in raw logs).
Mixed-site rule ⇒ real. Fix: capture/apply stream 0 in StateSnapshot like
the recorder already does. (The 77/61/77 counts: the offset quirk
SB_QUIRK_STREAM_OFFSET_NOT_UPDATED lets 16 offset asserts pass in one
chain.)

### d3d9ex.c clusters

### d3d9ex.c/START_TEST
Sites: 5184=expected

`win_skip("Failed to get address of Direct3DCreate9Ex")` — win_skip counts
as a failure under Wine, and START_TEST returns immediately, so no d3d9ex
test ever runs. We deliberately don't export Direct3DCreate9Ex (D3D9Ex out
of scope; target titles use plain D3D9). (Previously mis-attributed to
test_scene, which never executes.)
