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

## What the baseline records

`baseline.txt` is the single source of truth for the current results. For each
`(arch, subtest)` it records the crash bit plus every failing assertion as a
`<file>.c:<line>` site with a hit count and a classification tag:

```
[i686/device] crash=1
  device.c:125 count=28 class=real
  ...
```

Per-site granularity is what makes the score actionable: the 74 `device`
failures, for instance, are really three source lines hit repeatedly in a loop,
not 74 distinct defects. Recording the location (not just a total) means a fixed
bug and a new regression can no longer net out to the same number and hide each
other.

`make conformance-baseline` re-records the file. The merge **preserves** the
classification a human assigned to any site that still fails, marks
newly-appeared sites `untriaged`, and drops sites that no longer fail — so triage
survives across runs while genuinely new failures stay loud. A run whose Wine
version differs from the baseline's recorded version warns that `file:line`
sites may have drifted (a Wine update renumbers source lines) and a re-baseline
is expected.

## Classification tags

Each failing site carries one tag, explaining *why* it fails:

- **`real`** — a genuine defect we intend to fix.
- **`caps`** — the test takes a different expected-value branch because our
  capability bits omit something; our actual pixels/values are correct, so this
  is not a defect.
- **`expected`** — we deliberately do not implement this (e.g. D3D9Ex); the
  failure is the documented, by-design outcome.
- **`crash`** — a site attributed to a crash/abort path.
- **`untriaged`** — newly appeared and not yet classified by a human. The runner
  flags these on every run; they should be triaged and re-tagged in
  `baseline.txt`.

The counts are the signal, not a target of zero. Wine's `todo_wine`/`broken()`
annotations are tuned for a real-GPU driver, not for us, so a raw failure is
not necessarily a real defect — the classification is what turns the number into
something actionable. Note that when a subtest crashes, the counts cover only the
failures reached *before* the crash truncated the run.

## Per-cluster audit — current authoritative picture

This section is the authoritative current state of every failing site. Every
failing assertion was captured with its real message
(`MTLD3D_CONFORMANCE_RAW_DIR=<dir> make conformance` dumps each subtest's raw
output to `<dir>/<arch>-<subtest>.log`), grouped into 66 clusters by enclosing
Wine test function, and each cluster audited against the Wine test source, the
*actual-vs-expected* runtime values, and our implementation — then the
classification was adversarially re-verified.

Headline (per-(arch,subtest) site-lines, both PE arches): **87 `real` · 28
`caps` · 315 `expected` · 2 `flaky` · 0 `untriaged`**; all 8 subtest-arches
`crash=0`, gate green. The audit retagged 24 clusters: the prior tags had filed
a number of *fixable* defects as `caps`/`expected`. Tags do not affect the gate
(which rejects count/site/crash regressions), so the retag is documentation-only.

**Already cleared, each gate-verified (no NET regression, `crash=0`):**
`test_get_display_mode` (swapchain GetDisplayMode stub,
14356/14389/14390/14391/14489), `test_swapchain_parameters` (CreateDevice/Reset
present-param validation + caller writeback,
12439/12449/12477/12527/12558/12560/12562/12564), `test_getdc` backbuffer
(lockable-backbuffer GetDC via read-back, 9329/9330/9332), `test_stretch_rect`
cross-format RT blit (13357/13386/13427 — 13412 offscreen→offscreen deferred),
`stretchrect_test` (4247, offscreen staging refresh), `test_draw_mapped_buffer`
(26251/26276), `add_dirty_rect_test` (19156/19163, managed READONLY-first upload),
`srgbtexture_test` (8532, stop advertising SRGBREAD without a sampling decode),
and `fog_test` linear-fog math (2410/2412/2450/2454/2458/2462,
reversed/equal-range). Also pinned `device.c:4475` (wndproc) `flaky` — it flutters
cross-environment and now trips the gate. **`test_format_conversion` (28024) was
attempted then REVERTED: broadening `CheckDeviceFormatConversion` past
`is_present_compatible` regresses `test_display_formats` (which asserts
`CheckDeviceType(windowed)` agrees with it) and `yuv_color/layout_test` (which
gate their unsupported YUV→RGB blit on it) — it must stay narrow; deferred.**

**Cleared since the audit** (these rows are removed from the actionable table
below): `test_clip_planes_limits` (clip-plane CPU round-trip), `test_viewport`
(z-range fixup), `test_device_caps` (PERSPECTIVE bit), `test_multi_adapter`
(GetAdapterMonitor → primary), `test_occlusion_query` 6590 (never-issued 0xdd
poison), and the `test_check_device_format` real sub-set (UNKNOWN adapter format →
INVALIDCALL; SRGBWRITE-without-RENDERTARGET surface rejected). Implementing
GetAdapterMonitor let `test_multi_adapter` run into its fullscreen
window-rect-vs-monitor section (device.c:14987/15035/15053), now tagged `expected`
(a display/window environment we don't implement).

**`DrawIndexedPrimitiveUP` is now implemented** (inline `IndexSource::Up` +
a transient per-draw Metal index buffer). That un-gated the draw clusters that
routed every draw through the former stub: `test_drawindexedprimitiveup` (fully
cleared), `fog_test`, `lighting_test`, and `test_specular_lighting` (the
DIPUP-driven sites cleared; residual sites remain — see the per-line notes), and
the `test_draw_primitive` DIPUP sites (3265/3283/3290/3292). All un-gated draws
rendered correctly with no new failing sites.

A `caps`/`expected` line means "not a defect / by design"; a `real` line is a
defect we could fix. One source line can fire many assertions and can mix
sub-cases, so a line tagged `real` may still include some by-design assertions —
see the per-line notes.

### Key findings (highest value)

- **stateblock `D3DSBT_ALL` stream-0 vertex-buffer snapshot gap** — a real,
  **in-game-relevant** bug (portrait/state save-restore): `StateSnapshot` captures
  the index buffer but not the stream-0 VB. It shares baseline lines with the
  by-design multi-stream cluster, so it is documented here rather than separately
  tagged.
- **`test_filling_convention` (336, caps→real, HIGH in-game risk)** — missing the
  D3D9 half-pixel rasterization fixup (the pixel-center clip-space position
  offset a native D3D9 driver applies); our own `vPos` handling already
  contradicts the "pixel-center parity" premise the `caps` tag assumed. Likely
  the systematic sub-pixel offset behind past jitter hunts.
  **Touches all geometry — verify in-game before any fix.**

### Per-line / mixed-cluster notes (line tag = its primary cause)

- `visual.c:28481` (`test_fog`, real): 64/160 assertions are the real
  RHW/programmable-VS fog-fallback; 96/160 are by-design table fog. `28475` stays
  `expected` (pure table fog).
- `visual.c:15727` (`test_fetch4`, real): the 3D-volume box-upload gap. The other
  lines (`15617`/`15668` fetch4 gather, `15824`/`15829` DF16/DF24→Depth32) stay
  `caps`.
- stateblock `1810`/`1812`/`1813` (`expected`): dominantly by-design multi-stream
  capture/apply; the real `D3DSBT_ALL` stream-0 VB bug (~12 lines) is embedded in
  these counts and cannot be isolated to its own line.
- device `test_get_display_mode`: `14356`/`14389`/`14390`/`14391`/`14489` real
  (`swapchain_get_display_mode` stub); the rest `expected` (fullscreen
  mode/monitor environment we don't implement).
- device `test_check_device_format`: the former real sub-set
  (`12619`/`12626`/`12629`/`12632`/`12635` — SRGBWRITE-without-RT surface +
  `UNKNOWN`→`INVALIDCALL`) is **fixed**; `12689`/`12694` `caps` (D32 is genuinely
  creatable for us).
- device `test_occlusion_query`: `6590` (NeverIssued `0xdd` poison) is **fixed**;
  `6780` `caps` (Apple TBDR visibility undercount).
- device `test_draw_primitive`: the DIPUP sites `3265`/`3283`/`3290`/`3292` are
  **cleared**; `3269`/`3295` real (a residual DIPUP draw sub-case) + `3330` real
  (refcount).
- `add_dirty_rect_test`: `19156`/`19163` real (managed READONLY-first-lock never
  uploaded); `19210`/`19217`/`19232` `expected` (NO_DIRTY_UPDATE / explicit
  `AddDirtyRect` by design).
- device `test_stretch_rect`: `13357`/`13386`/`13427` real (cross-format RT blit,
  clearable via the render-quad path); `13412` real but harder (offscreen→
  offscreen, no clean Metal path).

### Reclassifications applied (24 clusters, 196 site-lines)

- **caps→real**: `test_clip_planes_limits`, `test_device_caps`,
  `test_multi_adapter`, `test_check_device_format` (A/B sites), `test_viewport`,
  `test_filling_convention`, `test_fetch4` (volume site), `test_shademode`,
  `z_range_test`, `srgbwrite_format_test`, `test_get_display_mode` (swapchain
  sites).
- **expected→real**: `test_swapchain_parameters`, `test_stretch_rect`,
  `test_occlusion_query` (6590), `test_vidmem_accounting`, `test_fog` (28481),
  `fog_test`, `zenable_test`, `test_drawindexedprimitiveup`, `srgbtexture_test`,
  `test_draw_primitive` (6 DIPUP sites).
- **caps→expected**: `test_get_display_mode` (mode/monitor sites),
  `test_window_position`.
- **expected→caps**: `test_occlusion_query` (6780, TBDR undercount).
- **real→caps**: `test_max_index16` (Metal `0xffff` primitive-restart +
  write-outside-lock UB; `broken(warp)` confirms).

### Actionable clusters (a `real` defect present)

Sorted by assertion count. The `caps`/`expected` remainder of mixed clusters is
in the per-line notes above. Lines column may be elided (`+N`).

> **Note:** the cleared batch listed under the headline already covers
> `test_get_display_mode`, `test_swapchain_parameters`, `test_getdc` (backbuffer),
> `test_stretch_rect` (13357/13386/13427), `stretchrect_test`,
> `test_draw_mapped_buffer`, `add_dirty_rect_test`, `srgbtexture_test`, and
> `fog_test` (linear math). Those rows below are kept for history; the live
> backlog is the remainder (`test_fog` 28481, `test_fetch4`, `test_filling_
> convention`, `test_miptree_layout`, `test_shademode`, `zenable_test`,
> `depth_blit_test`, `clear_test`, the passes.rs depth/store family, …).

| cluster | lines | cnt | root cause | fix (size) |
|---|---|---:|---|---|
| visual/`test_fog` | — | 0 | **FIXED (schema 54)**: RHW-with-bound-VS bypass + programmable-VS no-`oFog` specular-alpha fallback + per-pixel table fog (Z source = the `fog_z [[center_no_perspective]]` varying + raw `D3DRS_DEPTHBIAS` — Metal folds `setDepthBias` into `[[position]].z` scaled to float-buffer ulps, so the fragcoord depth is unusable as the fog source; W source = `1/in.position.w`). Surviving todo_wine configs pass-inside-todo, which the runner doesn't count. | done |
| visual/`test_fetch4` | 15617,15668,15727,15824,15829 | 438 | fetch4 gather = AMD vendor ext (caps); real gap is 3D-volume box-upload (15727) — `lock_box` contents never uploaded (slice0/z0/depth1 hardwired). | volume box upload (~150-250 lines + 1 wire field); fetch4 itself not worth it |
| stateblock/`resource_check_data` | 1810,1812,1813 | 430 | (a) multi-stream capture by-design (expected); (b) hidden real: `StateSnapshot` (D3DSBT_ALL) never captures the stream-0 VB (only index buffer). | hidden bug ~12 lines (snapshot+restore stream-0 VB); multi-stream stays deferred |
| visual/`test_filling_convention` | 27409 | 336 | missing the D3D9 half-pixel rasterization fixup (the pixel-center clip-space position offset); our `vPos` path already assumes the opposite convention. **HIGH in-game risk.** | medium (~40-80 lines, sign/Y-flip care + in-game smoke) |
| device/`test_miptree_layout` | 12784 | 144 | per-mip non-contiguous staging (`Vec<Arc<PageBox>>`); test checks the contiguous single-lock mip layout. | invasive refactor to one contiguous per-texture PageBox (deferred) |
| device/`test_swapchain_parameters` | 12439,12449,12477,12527,+4 | 126 | `d3d9_create_device` does no present-param validation and no resolved-param writeback (the Reset path does both). | shared `validate_present_params` + writeback (~80-120 lines, d3d9 only) |
| device/`test_stretch_rect` | 13357,13386,13412,13427 | 72 | `check_stretch_rect_formats` rejects cross-Metal-format pairs before the scaling decision; the converting render-quad path is never reached. | relax + route cross-format to render-quad (~40-80 lines); offscreen→offscreen harder |
| device/`test_get_display_mode` | 14356,14383,14384,14389,+11 | 48 | mostly by-design (CAMetalLayer no desktop mode-switch / monitor enum); real sub-set = `swapchain_get_display_mode` stub. | swapchain stub ~8-10 lines; the mode/monitor sites are expected |
| visual/`fog_test` | — | 0 | **FIXED** with the `test_fog` fallback work (RHW / programmable-VS fog fallback family). | done |
| visual/`zenable_test` | — | 0 | **FIXED**: `SetDepthClipMode` command driven by the D3D9 depth-clamp rule — clamp ⇔ depth test inactive (`ZENABLE` off OR no depth attachment) AND pre-transformed (RHW). Both conjuncts load-bearing: the S13 unconditional-RHW clamp broke `depth_clamp_test` (test live ⇒ clip, any `D3DRS_CLIPPING`), and a depth-test-only predicate broke zenable's second half (18152: regular-VS quad still clips with ZENABLE off). The RHW-with-bound-VS bypass makes the FF key's `has_rhw` cover every pre-transformed draw. | done |
| visual/`test_shademode` | 8852,8854 | 28 | `D3DRS_SHADEMODE` stored but never consumed; varyings always smooth (no `[[flat]]`). | FLAT variant + `[[flat]]` on diffuse/specular, cache-key bump (~100-200 lines) |
| visual/`depth_blit_test` | 14835 | 24 | depth-stencil `StretchRect` branch is a deliberate no-op (returns S_OK, emits no GPU copy). | emit a real depth→depth blit (non-trivial) |
| visual/`clear_test` | 1292,1294,1296,1303,+6 | 20 | NULL-rect clear fold path uses a full-attachment `loadAction=Clear` that ignores the viewport (only has-work/cross-pass paths emit a viewport-clipped clear-quad). | viewport-vs-attachment check in the fold path (~small) |
| device/`test_draw_primitive` | 3269,3295,3330 | 3 | DIPUP now implemented (3265/3283/3290/3292 cleared); residual = a DIPUP draw sub-case (3269/3295) + a refcount tail (3330). | investigate the residual sub-case + refcount |
| device/`test_getdc` | 9124,9329,9330,9332 | 14 | (A) `CreateDIBSection` `biSizeImage=0` DIB mismatch under Wine; (B) backbuffer GetDC readback. | two separable fixes in surface.rs |
| visual/`test_draw_mapped_buffer` | 26251,26276 | 12 | a draw from a mapped/relocked staged buffer reads stale interior; staged-buffer draw snapshot doesn't flush the pending map. | flush pending stage on draw-while-mapped (~30-60 lines) |
| visual/`z_range_test` | 3887,3889,3891,3894,3963,3965 | 12 | store-action Rule B flips depth-store to DontCare; a Clear(z) before Present is computed but never committed, next frame reads stale depth. | optional, **NOT recommended** (re-adds per-frame depth bandwidth for nil game gain) |
| visual/`add_dirty_rect_test` | 19156,19163,19210,19217,19232 | 10 | real: a managed mip whose first lock is READONLY is never GPU-uploaded (black). other sites NO_DIRTY_UPDATE/explicit-AddDirtyRect = expected. | real part ~6-10 lines (gate the READONLY early-return for managed-not-yet-uploaded) |
| visual/`test_mipmap_upload` | 27550 | 10 | per-mip non-contiguous staging can't satisfy a single full-chain `LockRect(0)` contiguous pointer walk. | contiguous per-texture PageBox refactor (sizable) |
| device/`test_resource_access` | 13838,13853 | 8 | (1) DEFAULT cube rejected (cube cap off); (2) a volume/usage access check. | both fixable but in-game risk for nil gain (LARGE for cube) |
| visual/`texdepth_test` | 5360,5398,5436,5454 | 8 | NOT a shader bug — the ps_1_4 `saturate(src.x/min(src.y,1.0))` clamp is present and matches the D3D9 ps_1_4 reference behavior. The 4 sites fail because Rule B (`finalize_store_actions`) flips the auto depth-stencil store to `DontCare` at `Present`, so the gradient depth buffer doesn't survive to data2-7. | real, in the passes.rs depth/store family — HIGH WoW risk, deferred (see test_fog/z_range Rule B group) |
| visual/`srgbwrite_format_test` | 16575 | 6 | sRGB-write not implemented + a loose cap advertises SRGBWRITE for formats without a Metal sRGB twin + a dropped TFACTOR fill. | tighten cap (~3 lines) + sRGB-write views (~60-120) + repro the drop |
| visual/`test_format_conversion` | 28024 | 6 | `CheckDeviceFormatConversion` gates on `is_present_compatible` (a present-time predicate) and rejects pairs the D3D9 spec permits. | dedicated conversion predicate in direct3d9.rs |
| device/`test_vidmem_accounting` | 10248,10250 | 4 | `GetAvailableTextureMem` returns a constant 512MB; no allocation accounting. | `vram_bytes_used` accounting per create/release (~80-130 lines) |
| visual/`vface_register_test` | 10124,10126 | 4 | render-to-texture quad never reaches the later blit's sampler — we sample the clear colour (front/back both wrong). | reproduce via e2e, then fix the RT→sample ordering |
| visual/`offscreen_test` | 2944 | 2 | load/store Rule E drops a depth clear that crosses a mid-frame RT switch. | treat an intervening LOAD as blocking the clear merge (passes.rs) |
| visual/`srgbtexture_test` | 8532 | 2 | cap advertises SRGBREAD for A8R8G8B8 but sRGB decode is never applied (sampler has no decode property). | minimal: stop advertising SRGBREAD (→caps-skip, ~3-5 lines); or full sRGB views (perf cost) |
| visual/`stretchrect_test` | 4247 | 2 | non-scaling offscreen→offscreen StretchRect copies into the Metal texture but the CPU staging mirror isn't refreshed, so a later Lock reads stale. | ~20-40 lines: refresh staging after the GPU copy |
| visual/`test_blend` | 9029 | 2 | `X8R8G8B8` maps to a real-alpha `Bgra8Unorm` RT; DSTALPHA blend reads the written alpha instead of the implied 1.0. | format_has_alpha + force-opaque for X8 RTs (~40-60 lines) |
| visual/`test_map_synchronisation` | 25148 | 2 | a plain (no NOOVERWRITE/DISCARD) PARTIAL lock of a contended Direct buffer isn't synchronised. | deferred (rare contended-direct case) |
| visual/`test_max_index16` | 24135 | 2 | **caps**: Metal treats `0xffff` as primitive-restart for strips (no disable) + test writes outside its lock (UB); `broken(warp)` confirms. | n/a (caps) |

### Confirmed non-`real` (do-not-chase / acceptable) — audit upheld the tag

| cluster | cnt | tag | why |
|---|---:|---|---|
| visual/`fog_special_test` | 0 | — | **FIXED** by the fog overhaul (18342/18345 → 0 both arches). |
| device/`test_pinned_buffers` | 4 | caps | a D3D9-driver "pinned buffer" optimization probe; not a correctness requirement |
| visual/`fp_special_test` | 4 | caps | NaN/±inf special-value categories; GPU-specific encodings |
| visual/`test_default_attribute_components` | 4 | caps | FLOAT3 unorm rounding (76.5→77 vs refrast 76) |
| visual/`fog_with_shader_test` | 0 | — | **FIXED** by the fog overhaul (3350: 198/arch → 0 both arches). |
| visual/`stream_test` | 368 | expected | hardware geometry instancing (single-stream architecture) |
| device/`test_lockrect_invalid` | 198 | expected | `broken()`-only offset checks our runner doesn't honor |
| device/`test_wndproc` | 84 | expected | fullscreen Win32 wndproc/mode environment |
| device/`test_reset` | 36 | expected | fullscreen Win32 Reset contract (mode change / focus) |
| visual/`lighting_test` | 3 | expected | DIPUP-driven sites cleared; residual `713` is a minor FF-lighting fidelity difference |
| device/`test_mode_change` | 26 | expected | desktop display-mode-change lifecycle |
| device/`test_wndproc_windowed` | 16 | expected | windowed wndproc-hook environment |
| visual/`clip_planes` | 16 | expected | `SetClipPlane` GPU application is a no-op (state round-trip is separate) |
| visual/`test_sysmem_draw` | 16 | expected | ProcessVertices SW transform / SYSTEMMEM draw |
| visual/`fixed_function_decl_test` | 12 | expected | two-sided-stencil / color-ubyte switching loop |
| device/`test_lost_device` | 10 | expected | fullscreen lost-device focus lifecycle |
| visual/`test_flip` | 10 | expected | windowed flip/present-flag behaviour |
| device/`test_window_style` | 8 | expected | fullscreen window-style adoption |
| device/`test_device_window_reset` | 6 | expected | fullscreen device-window management |
| device/`test_cube_textures` | 4 | expected | cube create rejection when CUBEMAP cap off (by design) |
| device/`test_window_position` | 4 | expected | fullscreen window repositioning / monitor enum (env we don't implement) |
| d3d9ex/`test_scene` | 2 | expected | whole d3d9ex file `win_skip`s (no `Direct3DCreate9Ex`) |
| device/`init_d3d9on12_modules` | 2 | expected | no `Direct3DCreate9On12` export |
| device/`test_d3d9on12` | 2 | expected | D3D9-on-12 interop N/A |
| device/`test_fpu_setup` | 2 | expected | i686 x87 control-word rewrite we don't (and shouldn't) do |
| device/`test_npot_textures` | 2 | expected | NPOT cube create when POW2 cap unset |
| device/`test_reset_fullscreen` | 2 | expected | windowed→fullscreen Reset activation (WM_ACTIVATEAPP) |

*(The `test_cursor_pos` `device.c:5368` site is pinned-flaky: it fluttered to 0 on
both arches during this audit's capture, confirming non-determinism; it is left at
its existing tag and handled by the flaky-tolerant gate.)*
