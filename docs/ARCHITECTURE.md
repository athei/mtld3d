# Architecture

mtld3d is a Wine-side translation layer that ships a D3D9 implementation backed by Metal. The runtime is split across three linkage units that meet at the Wine PE/Unix boundary.

```
test.exe → d3d9.dll → mtld3d.dll → mtld3d.so
(i386 PE)  (i386 PE)  (i386 PE)  (x86_64 Mach-O)

test.exe → d3d9.dll → mtld3d.dll → mtld3d.so
(x64 PE)   (x64 PE)   (x64 PE)   (x86_64 Mach-O)
```

- `d3d9.dll` — D3D9 API implementation. COM vtables, caps, state management. Calls Metal-level thunks via its internal `unix_call` caller stub (`windows/d3d9/src/unix_call.rs`).
- `mtld3d.dll` — PE shim. Links winecrt0, owns Wine unix-call globals, exports `mtld3d_unix_call()`. Forwards every cross-boundary call from `d3d9.dll` into `mtld3d.so`.
- `mtld3d.so` — native macOS side. Pure Metal abstraction layer: thunks expose Metal operations only, no D3D9 knowledge.
- `mtld3d-core` — pure-Rust rlib linked into `d3d9.dll`. Host-testable.
- `shared` — PE↔Unix wire-format definitions plus cross-linkage-unit helpers.
- `types` — D3D9 type definitions (vtables, caps structs) shared between d3d9 and tests.

## Threading model

The API thread (the game's calling thread) is the bottleneck and must be unblocked fast. Every D3D9 call snapshots the relevant state (cheap u32 copies, vertex memcpy) into a closure and pushes it onto the current frame's op list — no translation, no Metal lookup, no encoding.

A dedicated **encoder thread** (one per device, `sync_channel(1)` backpressure) does the real work. On `Present()`, the API thread sends the accumulated frame and immediately starts collecting the next. The encoder thread runs each closure with mutable access to a `FrameEncoder` that owns persistent caches (pipeline states, depth/stencil states), translates D3D9 → Metal commands into a fixed-size array, and submits across the PE/Unix boundary.

```
API thread                          Encoder thread
──────────                          ──────────────
D3D9 call → snapshot state          (blocked on channel)
         → push closure
         ...
Present() → send frame ──────────→ run closures
         → start next frame         translate D3D9 → Metal
                                    submit command buffer
```

## Thunk vs Command

Two paths cross the PE/Unix boundary:

- **Command** = `MTLRenderCommandEncoder` method. Closures accumulate commands into a fixed-size array; `submit()` sends the whole array in one `unix_call(SubmitCommandBuffer)` and the unix side replays them inside a render pass. Examples: `setRenderPipelineState`, `setViewport`, `drawPrimitives`, `setFragmentTexture`.
- **Thunk** = everything else (object creation/destruction, texture upload, blit). Individual `unix_call()`.

API-thread thunks are restricted to device lifecycle only (`CreateCommandQueue`, `AttachMetalLayer`, `CreateBackbuffer`, `DestroyCommandQueue`). Everything else — texture/sampler/pipeline creation, texture upload, resource destruction — runs on the encoder thread. Metal textures are created lazily on first draw via `FrameEncoder.texture_cache`. Resource cleanup uses closures pushed to the current frame.

Thunks are Metal-level operations, not D3D9 calls. Name thunks after what they do in Metal (`GetDeviceInfo`, `CreateCommandQueue`), not after D3D9 methods. D3D9 logic stays in `d3d9.dll`. Objects with no Metal state (like `IDirect3D9`) are PE-only.

## Raw pointers across the boundary need stable backing

Commands carry `u64` param fields the unix side dereferences (`setVertexBytes` ptrs, `commands_ptr` inside `PassDescriptor`, the `PassDescriptor` array itself). The backing must not move between hand-out and `unix_call` return.

A growing `Vec<u8>` silently reallocates on capacity growth, invalidating every previously-returned pointer; the unix side then dereferences freed memory, Wine's SEH shim translates SIGSEGV to `STATUS_ACCESS_VIOLATION` (`0xc0000005`), and the PE side sees `unix_call` return non-zero. Prefer `Vec<Box<[u8]>>` (one heap block per allocation) or a chunked bump allocator where chunks never move. `FrameEncoder.scratch` is the canonical example.

## `status=0xc0000005` from `unix_call` is a unix-side SIGSEGV

Wine's unix-call dispatcher wraps each handler in a SEH-translation shim. Any `0xc0000005` means the unix side crashed mid-call — the PE side's own early-return error logs (`queue retain failed`, `renderCommandEncoderWithDescriptor returned nil`, …) will **not** fire. Diagnose by instrumenting each step of the unix-side path with a log line keyed to what it's doing; the last line printed before the PE-side error is the crash site.

## Shared wire values are typed in `unix/shared/src/mtl.rs`

Any integer crossing the boundary with *symbolic* meaning — Metal enum codes (storage mode, pixel format, compare func, blend factor, primitive type, sampler filter/address, …), bitflag masks (texture usage, color write mask), stage selectors — is declared **once** in `mtl.rs` as a `#[repr(u32)]` enum (single-choice) or `bitflags!` struct (multi-bit). Param fields in `params.rs` use that type directly, *not* `u32`.

**Never** restate the encoding as a local `const`. **Never** write an integer literal at a call site or decode arm.

The three cdylibs are separate linkage units. An untyped integer is a silent-drift risk: PE adds a variant, Unix decode forgets, a `log_once_warn!` covers it. Typed fields move the contract into the shared crate; exhaustive `match` becomes a compile error the instant a variant is added. Sound because `make` / `make install` rebuild and copy all three together — the "unknown variant" UB bit pattern never appears on the wire.

How to apply:
- New thunk field with symbolic meaning → `mtl::` type. Sizes/offsets/counts/`!= 0` booleans → `u32`.
- Adding a value: extend `mtl.rs`, extend `d3d_to_metal_*` if one exists, the compiler points at every Unix-side match site.
- Bit-flag fields use `bitflags!` (`TextureUsage`, `ColorWriteMask`).
- `Command::param_a/b/c/d` carry polymorphic `u32`s whose meaning depends on `Command::cmd`. Stay `u32` on the struct; encode via `Enum::Variant as u32` in the `Command::foo` constructor and decode via `Enum::from_repr(raw)` (strum `FromRepr`) in the dispatcher — never a bare `match raw { 0 => …, 1 => …, … }`.

## Adding new thunks

1. Add variant to `Thunks` enum in `mtld3d-shared` `lib.rs` (count and iteration via strum).
2. Add param struct in `mtld3d-shared` `params.rs` (`#[repr(C, align(8))]`). Field types: `u64`, `u32`, `#[repr(u32)]` enums from `mtl::`, or `bitflags!` structs from `mtl::`. Symbolic-meaning integers must use `mtl::`. `impl Thunk` with the matching code.
3. Add handler in `mtld3d-unix`, add arm to `dispatch()` (exhaustive match = compile error if forgotten).
4. Call via `unix_call(&mut params)` from `d3d9`.

## Label every Metal object created

Every `MTLDevice.new*…`, `MTLCommandQueue.commandBuffer()`, `MTLCommandBuffer.{render,blit}CommandEncoder*`, and per-stage descriptor that produces a Metal-side state object must get a `setLabel:` call before it's handed back across the boundary or used. Strings start with `mtld3d-` followed by the role and an identifying suffix — `mtld3d-tex-{tex_id:#x}`, `mtld3d-vbib-{buffer_id:#x}`, `mtld3d-frame-{submit_seq:#x}`, `mtld3d-pass-{idx}`, `mtld3d-samp-{key:#x}`, `mtld3d-backbuffer`, `mtld3d-depth`, `mtld3d-readback`, `mtld3d-mipgen`, …

Xcode GPU frame captures, Metal validation logs, and the Metal HUD display these labels everywhere they show a Metal object. Without them every handle shows up as `Buffer (8KB)` / `RenderCommandEncoder` / `Texture (BGRA8 1024×1024)`, which makes any handle-recycle / cross-device-alias / contention investigation start with "and which one is this?". With them the mapping back to a mtld3d-side identity (`TextureId` / `BufferId` / `submit_seq` / pass index / packed-bits state key) is one column in the resource browser.

How to apply:
- For *create-style* thunks (texture, buffer, sampler, DSS), the param struct on the PE side carries an `id: u64` (and a `kind` enum where one struct serves multiple roles, e.g. `BufferKind` on `CreateBufferParams`). PE side fills it from the appropriate strong-typed identifier (`tex_id.raw()`, `buffer_id.raw()`, `SamplerKey::raw()`, `DepthStencilKey::raw()`); unix side composes the label string and calls `setLabel`.
- For *per-frame* objects created entirely on the unix side (the per-frame `MTLCommandBuffer`, per-pass `MTLRenderCommandEncoder`, blit encoders, mipgen + readback transients), label inline at the create site using whatever in-scope identity disambiguates instances (`SubmitFrameParams::submit_seq`, the `pass_idx` loop variable, a static role string).
- For *descriptor-then-state* paths (`MTLRenderPipelineDescriptor`, `MTLSamplerDescriptor`, `MTLDepthStencilDescriptor`), call `setLabel` on the **descriptor** before the `newXxxStateWithDescriptor:` call — the label propagates onto the resulting state object.

Trait-import caveat: `setLabel` lives on different traits depending on the object. `MTLBuffer` / `MTLTexture` / `MTLSamplerState` / `MTLDepthStencilState` need `use objc2_metal::MTLResource;`. `MTLRenderCommandEncoder` / `MTLBlitCommandEncoder` need `use objc2_metal::MTLCommandEncoder;`. `MTLCommandBuffer` and `MTLCommandQueue` provide it on their own protocol traits, no extra import.

Cost: one `format!` + one `NSString::from_str` + one objc dispatch per create call. Negligible — paid only at object-create time (cache miss / per-frame at most). Ship unconditionally; never gate on `cfg(debug_assertions)`.

## Perf infrastructure

The `mtld3d::perf` summary in `windows/core/src/perf.rs` emits a multi-line report every 5 s under `RUST_LOG=mtld3d::perf=debug`. Counters group by which thread owns them (API, encoder, GPU wait); subtimers indent under their parent. Banner shows `bottleneck=…` based on `present_block` share + `gpu_wait` vs `enc_cpu`; the four terminal buckets are echoed on a `buckets:` line for auditability. The same Debug gate also enables the per-call cycle accounting — single switch. Pass / workload shape (per-pass dump, `present_texture=…` audit line, per-RT pair stats) lives on the separate `mtld3d::d3d9::passes=trace` switch — those are diagnostics, not perf metrics.

Counter aggregation — mixing these up misreads the log:

- **Time counters** (anything ending in `ms`): per-frame averages.
- **Event counters** (passes, commands, draws, fresh, discards, wraps, …): raw window totals — divide by `frames=N` for a rate. Never average an event counter — silently rounds rare signals to zero.
- **Depth counters** (retention depth, retention KB): f64 averages, formatted `.1`.
- **Cache-size snapshots**: point-in-time at window emit, neither averaged nor summed.
- **Peak counters** (`peak …` cells): max value on any single frame in the window.

ANSI colour by default. Override with `NO_COLOR=1` (no escapes when redirecting to a file) or `CLICOLOR_FORCE=1`. Auto-detection via `is_terminal()` is not used: under Wine the Windows console check returns false even when macOS fd 2 is a real TTY.

### Don't hand-roll `rdtsc()` brackets — use `perf::ApiTimer` / `CycleSetTimer` / `CycleAddTimer`

Time measurements that flow into the perf summary go through one of:

- `ApiTimer` — D3D9 vtable entry brackets, accumulates into `api_cycles_by_category[Category]`.
- `CycleAddTimer` — sub-scope inside an outer `ApiTimer`, accumulates into a `*mut u64` field (e.g. `query_wait_cycles`).
- `CycleSetTimer` — once-per-frame measurement that overwrites a `*mut u64` field (e.g. `present_block_cycles`, `op_cycles`, `submit_cycles`, `drawable_wait_tsc`).

All three gate on a static `PERF_TRACKING_ENABLED: AtomicBool` latched once at logger init from `log_enabled!(target: "mtld3d::perf", Level::Debug)`. When perf isn't being reported the helpers cost ~1 ns per call (one `Relaxed` atomic load + branch); when it is, the cached load avoids the per-call env_logger filter walk — the level chosen for the latch (Debug) is paid once at init, not per call.

The summary itself emits at `debug!` on the same target, so the user-facing switch is a single `RUST_LOG=mtld3d::perf=debug` for both the cycle accounting and the rendered grid.

A second cached gate, `PAIR_STATS_ENABLED`, latched from `log_enabled!(target: "mtld3d::d3d9::passes", Level::Trace)`, fronts `bump_pair_stats` and the per-pass / `present_texture=…` / per-RT pair lines that `log_frame_summary` appends after the grid. Those are pass-shape and workload-shape diagnostics — they ride the same `mtld3d::d3d9::passes` target as the per-event pass-break / pass-open probes in `windows/core/src/passes.rs`, not the perf target.

The unix `.so` carries its own `PERF_TRACKING_ENABLED` + `CycleSetTimer` (in `metal/command.rs`) — each cdylib has its own `log` statics so the cache is per-runtime. Each cdylib calls its own `init_tracking_enabled` from logger init.

The two legitimate raw-`rdtsc()` use cases are (a) inside `mtld3d-core::perf` — the helpers themselves, frame/window boundary timestamps, calibration in `tsc.rs` — and (b) rdtsc as a *clock argument*, not a bracket — e.g. `BurstTracker::poll(now, …)` in shader-compile debounce.

## Debugging rendering bugs — shader/pass toolkit

Four off-by-default knobs answer "which shader on which RT produced the bad pixels":

1. **Pass × shader correlation log** — `RUST_LOG=mtld3d::d3d9=debug`. One `debug!` line per unique `(RT size, VS, PS)` triple; shaders tagged `prog 0x…` (content-hash, stable across runs) or `ff 0x…`. Implementation: `FrameEncoder::maybe_log_pass_shader` from `draw::emit_draw`.
2. **MSL dumps** — `RUST_LOG=mtld3d::dxso=trace`. Bracketed by `── VS MSL prog 0x… ──` / `── /VS MSL prog 0x… ──` (and PS).
3. **Raw DXSO bytecode dump** — `MTLD3D_BYTECODE_DUMP=/tmp/mtld3d_shaders`. Writes raw LE `u32` token streams to `{vs|ps}_{id:x}.dxso` on `Create*Shader`. Idempotent per id.
4. **Offline disassembler** — `cd windows && cargo run --example disasm --target aarch64-apple-darwin -- /tmp/mtld3d_shaders/ps_<id>.dxso`. Prints raw tokens, parsed IR (`{:#?}` on `DxsoProgram`), and emitted MSL. Host-only, no Wine.

Typical workflow:

```sh
RUST_LOG=mtld3d=warn,mtld3d::d3d9=debug,mtld3d::dxso=trace MTLD3D_BYTECODE_DUMP=/tmp/mtld3d_shaders ./<game>.exe > /tmp/trace.log 2>&1
```

Reproduce → grep `pass RT` for suspect ids → grep `── PS MSL <id>` for emitted MSL → `cargo run --example disasm` for deeper analysis → seed a regression test in `core/src/dxso/emit_tests.rs`.

## Debugging heap corruption — `MTLD3D_CRUMB=1` mmap breadcrumb

`unix/shared/src/crumb.rs` is a zero-I/O crash breadcrumb mapped at `Z:\tmp\mtld3d-crumb.bin` (= `/tmp/mtld3d-crumb.bin` under Wine). Probe calls compile to a single `mov [ptr], rax` when enabled and to nothing when disabled.

```sh
MTLD3D_CRUMB=1 make install   # cfg routed through build.rs
./<game>.exe                  # reproduce
xxd /tmp/mtld3d-crumb.bin     # read last-recorded state
```

`MTLD3D_CRUMB=1` is used instead of `RUSTFLAGS="--cfg mtld3d_crumb"` because cargo prefers env `RUSTFLAGS` over `[target.*.rustflags]` (does not merge), so the env approach silently drops xwin `-Lnative=…` paths. `windows/d3d9/build.rs` reads `MTLD3D_CRUMB` and emits the cfg through `cargo:rustc-cfg`, which composes correctly.

Slot layout (8 bytes each, 128-byte map):

| Off | Writer | Meaning |
|-----|--------|---------|
| `0x00` | encoder | `(frame << 32) \| op_idx` |
| `0x08` | encoder | `Phase` tag (see `Phase` enum in `crumb.rs`) |
| `0x10` | API | `(ApiMethod << 56) \| (level << 48) \| (flags << 16)` for the last Lock/Unlock |
| `0x18` | API | pointer returned to the game from that Lock |
| `0x20` | any | `(thunk_code << 32) \| (status << 8) \| marker` — `0xEE` mid-call, `0xDD` returned |
| `0x28` | any | `unix_call` `params` pointer at entry |
| `0x30` | API | `(seq << 32) \| (tcc_max << 16) \| tcc_last` — `FfVsKey::tex_coord_count` at draw-snapshot capture |
| `0x38` | encoder | same shape — same field at `emit_vs_ff` dispatch |
| `0x40` | API | address of the captured `&FfVsKey` (PE-side closure storage) |
| `0x48` | encoder | address of the dispatched `&FfVsKey` |
| `0x50` | API | wrapping byte-sum + rotate fingerprint of all FfVsKey bytes at capture |
| `0x58` | encoder | same fingerprint at dispatch — mismatch ⇒ at least one byte changed in transit |

Adding a new probe: define under both `enabled` and `disabled` modules with matching signatures, document the new slot in this table, call directly from the suspect site (no `#[cfg]` at the call site). Single `write_volatile` so the disabled-build optimizer fully elides them. No formatting or syscalls — preserve the zero-cost-when-off contract.
