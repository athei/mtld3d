# Conventions

Code-organisation and style rules enforced across the codebase. New code follows these without exception; existing code is brought along on any touch.

## Mechanical audit

Most of this document is enforced by `make check`: `cargo +nightly fmt --check`, then clippy at `-D warnings` with `nursery` + `pedantic` denied, then `make audit`, then `make doc`. But a lint can only express a lint-shaped rule, and the rules a lint *can't* express are exactly the ones that drift — nothing runs them, so they decay into prose nobody rereads.

`make audit` (`scripts/audit.sh`) closes that gap, and `make check` runs it — so a violation fails the same gate as a clippy warning, and the tree stays at zero rather than drifting until someone notices. It covers:

| Check | Section |
| --- | --- |
| Doc-block shape (title / blank / body) and the 100-column doc line cap | §Doc comments |
| The `Clone` / `Copy` derive inventory, diffed against `scripts/derive_inventory.txt` | §No default `Copy` / `Clone` |
| `#[allow(...)]` confined to the three accepted files | §Warning suppressions |
| `#[inline(always)]` confined to the one measured site | §Inline attributes |
| `static … : OnceLock` confined to the three runtime-argument sites | §`LazyLock` over `OnceLock` |
| `pub(crate)` = 0 | §No `pub(crate)` |
| `extern "stdcall"` = 0 | §`extern "system"` everywhere |
| `msg_send!` / `class!` / `sel!` = 0 | §No raw `msg_send!` |
| `mod.rs` files = 0 | §Module style |
| Release hygiene (see below) | §Release hygiene |

Every finding names the section it came from. The confined-pattern checks compare **sets of files**, not counts, so moving an exception to a new file fails even though the count is unchanged — which is the point: each of those files earned its exception with an argument recorded here, and a new one needs a new argument.

`scripts/audit.sh --file <path>` runs the per-file subset, for an editor hook that wants feedback at edit time.

### `make doc`

`make check` also builds the docs with `RUSTDOCFLAGS="-D warnings"`, because rustdoc holds the last set of rules nothing else sees: whether a doc link actually **resolves**. Audit gates the *shape* of a doc block and clippy gates its *prose*; only rustdoc knows that `[`foo`]` points at something real, that a public item isn't linking to a private one the reader can't open, and that `"D3DFMT_<code>"` in running text is a malformed HTML tag rather than words.

Two traps worth knowing, since both were live in the tree:

- **Square brackets in prose are link syntax.** `palette[0]`, `saturate to [0,1]` — rustdoc reads each as an intra-doc link and warns. Backtick them.
- **A bare method name doesn't resolve from inside an `impl`.** Intra-doc links resolve against the *module*, so a doc on one method linking to a sibling with ``[`restore_into`]`` is broken; it needs ``[`Self::restore_into`]``.

The windows workspace is documented for the **i686 PE target**, not the host: `d3d9` and `mtld3d` are `cdylib`s with raw-dylib imports and don't build for the host, so a host-only run would silently skip them.

## Release hygiene

The repository is public. Source should read as the engineering itself — not as a diary of how the engineering got written, and not as a document that assumes the reader has access to things they don't. Every rule below was written down long before anything ran it, and every one of them had drifted by the time it was checked. `make audit` now enforces them.

Do not write, in any comment:

- **A reference to a competing D3D9 implementation** (`DXVK`, `dxmt`, `wined3d`, `d9vk`). State the contract in spec terms instead — "per the D3D9 spec", not "matches what the other one does".
- **A citation of Wine's d3d9 test suite** (`device.c`, `visual.c`, `test_refcount`, `test_fog`) outside `unix/conformance/`. Somebody who clones this repo cannot open `device.c`, and the behavioural statement never needed it: keep "GetStreamSource reports the last non-null stride", drop "(Wine device.c test_refcount)". Inside `unix/conformance/` the citations *are* the data — that crate exists to run Wine's suite and key failures by `file:line` — so they stay.
- **A reference to private or non-public work**, e.g. a reverse-engineered third-party binary, or shorthand for a machine that came up while debugging. Keep the technical claim; drop the provenance.
- **Incident provenance**: "the bug class that bit us", "until commit `abc1234`", a date. State the invariant and why it holds. If a comment only makes sense to someone who knows what changed, restate the change as prose.
- **A tooling signal or a reference to a private note** (`project_*`, `feedback_*`), or an absolute `/Users/...` path.

**Wine itself is not on this list, and never will be.** mtld3d *is* a d3d9 layer *for* Wine: SEH/NTSTATUS, the unix-call boundary, `macdrv`, `wineserver`, `winebuild`, `server/thread.c`, and Wine's own d3d9 source all document the real host, and they belong in the source. The ban is on the *competing implementation* and on the *test suite cited as provenance* — not on the runtime this thing is built to live inside.

## Factor pure functionality into `mtld3d-core`; `d3d9` is wiring

`windows/d3d9` is the COM-wrapper layer — `#[repr(C)]` vtables, refcount accounting, `extern "system"` dispatchers, the raw-dylib `unix_call` stub, `DllMain`, and `DeviceInner`. **Everything else belongs in `windows/core`**: format mapping, state packing, bytecode parsing and MSL emission, allocator bookkeeping, geometry math, key hashing, render-pass sequencing, FF state, newtype identifiers.

`windows/d3d9` is a `cdylib` with `raw-dylib` imports, so it only builds for `*-pc-windows-msvc`. Any `#[test]` inside `d3d9` is unreachable without Wine. `mtld3d-core` is a pure-Rust rlib that builds on the macOS host target, so `cargo test -p mtld3d-core --target aarch64-apple-darwin` runs natively in ~0 s. `make test` already invokes it, auto-detecting the host triple so tests run on the native arch (no Rosetta) rather than the shipped `x86_64-apple-darwin` target.

How to apply:

- New `IDirect3DXxx9::Method` body: "unpack args → call helper → route return". Helper lives in `mtld3d-core` with a unit test.
- Helpers needing `DeviceInner` data take the leaf field/accessor slice they actually use — never `DeviceInner` itself. E.g. `FfStateSnapshot::restore_into(&mut FfState)`.
- Helpers needing `unix_call` inject it via `fn` pointer or trait, not a direct `mtld3d_shared` import. E.g. `slab::SeqWaiter = fn(&Arc<AtomicU64>, u64)`.
- "Done" = green host tests, not only a green Windows build.

Refactor signals: a module in `windows/d3d9/src/` has `#[cfg(test)] mod tests` · a function references nothing outside its own args + well-known types · a comment says "wrapper around …" · the same mapping table appears in two `d3d9` modules.

## End every edit cycle with `make fmt` + `make install`

After any `.rs` change, run both before declaring the task done. `make` only rebuilds — it does **not** copy DLLs/.so into `$WINE_INSTALL_DIR`, so launching the game without `make install` tests stale bits. `make test` implicitly runs both.

## winecrt0 TLS conflict

Linking the full `libwinecrt0.a` causes duplicate TLS symbols (`__tls_index`, `__tls_start`, …) with the CRT. Fix: extract only `unix_lib.o` via `ar p` in `build.rs`.

## `extern "system"` everywhere, not `extern "stdcall"`

`extern "system"` maps to stdcall on i386 and the x64 ABI on x86_64, so the same code compiles correctly for both targets.

## DLL exports: `.def` file + `import_name_type = "undecorated"`

On MSVC i386, `raw-dylib` imports add stdcall decoration (`_Name@N`), but DLL exports use undecorated names (matching real Windows DLLs).

- **Export side**: `#[unsafe(no_mangle)]` + list exports in `d3d9.def` (linked via `/DEF:` in `build.rs`).
- **Import side**: `import_name_type = "undecorated"` on i386:

```rust
#[cfg_attr(target_arch = "x86", link(name = "d3d9", kind = "raw-dylib", import_name_type = "undecorated"))]
#[cfg_attr(not(target_arch = "x86"), link(name = "d3d9", kind = "raw-dylib"))]
```

## Module style: `foo.rs` + `foo/`, not `foo/mod.rs`

Rust 2018+ module layout. Keeps meaningful filenames in editor tabs.

## No `pub(crate)` — use module hierarchy

Visibility via module hierarchy: private modules already restrict `pub` items to the crate.

## File layout: pub first, private last

1. Inner attributes (`#![...]`)
2. `use` imports
3. `mod foo;` / `pub mod foo;` declarations (grouped, not split by visibility)
4. All `const` / `static` / `type` aliases (pub then private) at the top regardless of visibility
5. `unsafe extern "ABI" { ... }` blocks with `#[link(...)]` attributes (top, not in the private section)
6. `pub use` re-exports + all `pub` items including `pub mod foo { ... }` inline modules
7. All private items including private inline `mod foo { ... }` blocks
8. `#[cfg(test)] mod tests { ... }` last

`impl` blocks stay adjacent to their type. Inline `mod foo { ... }` follows the same pub/private rule as other items.

## Encapsulation: minimize public fields

Prefer private fields with accessor methods. `pub` is for FFI data structs in `types/` (`D3DCAPS9`, vtable structs) and `*CreateInfo` parameter bags. COM wrappers' `vtbl` field is first but **private** — `#[repr(C)]` fixes layout independent of visibility; expose via a `vtbl(&self)` method.

## Submodule state: struct-in-submodule + one accessor pair

When a submodule needs state on a parent struct, define `FooState` *inside the submodule* with private fields, embed it as one field, and expose a single `foo()` / `foo_mut()` accessor pair. Never per-field getters on the parent. The parent's surface grows by 2 methods, not 2N. See `FfState`, `CursorState`.

## Data structure discipline

State carried on the hot path — every per-draw snapshot, every cache key, every encoder op — is shaped for the move/borrow semantics that keep memcpys out of the inner loop. Wide derives and wide field types are accidents waiting to happen: the compiler won't flag a 200 B `Copy` deref or a `Vec` `.clone()` until profiling does, by which point the cost is baked into every frame. The rules below get applied to every new struct and to existing structs on any touch.

### No default `Copy` / `Clone` on aggregate structs

The default for a struct >16 B is **no derive**. `Copy` and `Clone` are added only when a concrete callsite needs them and the structural alternative (passing references, moving once, building in place into the scratch arena) doesn't fit.

- **`Copy`** turns every accidental whole-struct read into a silent memcpy. Aggregate snapshot types (`CurrentSnapshot`, `RenderStateSnapshot`, `StageBinding`, `TextureInfo`, `FfVsKey`, `FfPsKey`, `PipelineSnapshot`, etc.) do not derive `Copy`. Removing `Copy` is how silent per-draw memcpys become compile errors — it has exposed ~200 B/draw copies hiding on the snapshot path.
- **`Clone`** is the opt-in form of the same hazard: a `.clone()` call on a wide aggregate is a per-call heap-bandwidth tax, just one the compiler doesn't flag. The default replacement for "I need a copy of this big struct in scratch" is `ScratchArena::alloc_from(&value)` (bytewise copy, no `Clone` bound) or `scratch.alloc_uninit::<T>()` + per-field `addr_of_mut!().write()` for in-place construction. The `scratch.alloc_from(&cache)` form — replacing per-field writes — is the canonical construction pattern.
- **Small types keep `Copy`.** Single-word newtypes (`MetalHandle<K>`, `TextureId`, `ProgramId`, `ScratchSlice`, `CurrentSnapshotPtr`, the various `*Flags` bitflag bytes) keep `Copy` because they exist to be cheap pointer-sized handles and the trait is structurally needed (Op enum payload, `Option<...>` field load). The rough threshold is 16 B — wider than that, justify each derive at the use site or don't derive it.
- **Tests stay unaffected.** Removing `Copy`/`Clone` doesn't break `#[derive(PartialEq, Eq, Hash, Debug)]`; tests compare via reference.

When a callsite *seems* to require `Clone`, the structural fix is almost always "build it in scratch once and pass the pointer" — the same lesson as the snapshot work. The `[[T; N]; M]` initialization pattern (which requires `Copy` to use the `[expr; N]` form) is the common exception; use array-of-`const`-default or `core::array::from_fn` instead.

**Derives are never speculative.** A `Clone` or `Copy` derive needs a concrete callsite *today* — not one it might plausibly need later. "It's a small struct, might as well" is how the tree accumulated a dozen `Clone`s that nothing called. Every type deriving either trait is recorded in `scripts/derive_inventory.txt`, which `make audit` diffs against the tree: adding a derive is a deliberate act that shows up in the diff, and the reviewer gets to ask for the callsite. Regenerate with `scripts/audit.sh --update-derives` once the callsite exists.

Note that a derive can be *structurally* required without any visible `.clone()`: `vec![elem; n]` fills by cloning, and a `Clone` type containing a field by value requires that field to be `Clone` too. Those are real callsites. Let the compiler arbitrate — remove the derive, build, and put back only what fails.

### Booleans pack

Two or more `bool` fields on the same struct collapse into `bitflags!`. A single bool field is fine as `bool`. The `bitflags` crate is already a workspace dep and the macro is used throughout: `DepthScissorFlags`, `PipelineRsFlags`, `PipelineAttachFlags`, `FfVsFlags`, `SnapshotDirty`. The bitflag value lives in the smallest int that holds the bit count (u8 ≤8 bits, u16 ≤16, u64 ≤64). Derive `Clone, Copy, Debug, Default, PartialEq, Eq` on the bitflag struct — the value is a single integer, so `Copy` is fine and structurally needed for the bitwise ops.

Per-`[stage][type]` once-warn latches (`tss_warn_fired`, `samp_warn_fired`) and per-slot enable arrays (`light_enabled`) follow the same rule: `[[bool; N]; M]` becomes `[uN; M]` packed at the bit level, with accessor methods (`mark_warn(stage, ty)`, `warn_fired(stage, ty)`) replacing direct array indexing. The savings are sometimes large (`[[bool; 33]; 8]` is 264 B; `[u64; 8]` is 64 B).

### Narrowest type for the range

Field types match the value range, not the on-the-wire format. D3D9 ships enum render states as `u32`, but the enum spaces (`D3DCMP_*`, `D3DCULL_*`, `D3DBLEND_*`, `D3DBLENDOP_*`) all fit in `u8`; snapshot/cache-key structs store them as `u8`. Scissor coordinates fit in `u16` (D3D9 max RT dim is 16384). Indices into bounded arrays (stages ≤16, samplers ≤16, RTs ≤4, RS slots ≤210) are `u8`. The wide types stay in the source RS array (FFI wire format) and in the `*CreateInfo` parameter bags — narrowing happens at the snapshot/cache-key boundary, where each byte matters per-draw.

`u32`-storing-a-bit-pattern fields (raw `D3DCOLOR`, raw `f32` bit pattern for depth bias) keep `u32` — the storage type is the right one, just with a non-numeric interpretation. Decode at the consumer site (`u32::to_le_bytes` for D3DCOLOR, `f32::from_bits` for the bias floats).

### Composition over flat fields

Fields that appear on ≥2 sibling structs with the same name + same type belong in a named substruct, embedded by value. See `PipelineRsBits` (10-field rasterizer/blend cluster shared by `PipelineSnapshot` and `RenderStateSnapshot`) and `PipelineAttachFlags` (3-bool collapse of `has_depth` / `has_stencil` / `has_color_output`). The signal to extract:

- The same field name + type appears on two sibling structs.
- ≥5 fields encode one conceptual thing ("blend state", "rasterizer state", "viewport", "presentation handles", "creation params").
- Two fields are always read/written as a pair (e.g. `depth_bias` + `slope_scale_depth_bias` always travel together into `Command::set_depth_bias`).

When extracting, derive `Copy`/`Clone` on the substruct only if the substruct is ≤16 B and gets embedded by value (the bitflag substructs are the canonical pattern). A wide substruct stays move-only.

### Extract substructs only when they pay for themselves

Composition is for reducing duplication or enforcing an invariant, not for cosmetic grouping. The four extant submodule structs on `DeviceInner` each earn their keep:

- **`FfState`** — shared by the device, state-block save/restore, and tests; the state-block API can't be written without a typed snapshot.
- **`CursorState`** — encapsulates WM_* subclass invariants the rest of `d3d9` shouldn't reach into.
- **`StageBindings`** — owns the `[CachedComPtr; STAGE_COUNT]` array's refcount accounting plus per-(sampler, type_) latches; the slot-count-bounded loops live there.
- **`ShaderBindings`** — pairs each stage's shader pointer with its constants array and `populated_rows` watermark.

Before introducing a new substruct, name the rule it enforces. "These fields are accessed together in `f()`" is not a rule — it's a coincidence of one function's shape. "Same field name + type appears on ≥2 sibling structs" *is* a rule (the `PipelineRsBits` extraction collapsed 10 such fields shared between `PipelineSnapshot` and `RenderStateSnapshot`). When in doubt, leave the fields flat; the next reader can extract once the duplication is real.

The split-borrow constraint matters when a substruct does land: keep clusters as *sibling fields* of `DeviceInner`, not nested under one wrapper. `dev.current_frame.scratch + dev.snapshot_cache` exercises field-level disjoint `&mut` access on the per-draw hot path; any new substruct must preserve that.

## Strong types for cross-module identifiers

Wrap distinct `u64`s (D3D9 object id, Metal handle, content hash, packed-bits state key) in newtypes with **private inner field**. Construction owned by the defining module via domain-specific factories — never raw-`u64` constructors. Examples: `TextureId::new_unique()`, `ProgramId::from_tokens(&[u32])`, `DepthStencilKey::from_state(...)`. For logging implement `fmt::LowerHex`. Add `.raw()` only when a concrete site (typically a thunk param at FFI) needs the `u64`. See `ids.rs`.

## ABI constants have one home — never restate them locally

Every constant that names a value in an external ABI is declared **once**, in a shared crate, and referenced everywhere else. There are two homes, mirroring the two ABIs the codebase straddles:

- **D3D9 ABI constants** → `mtld3d-types` (`windows/types/src/`). `D3DFMT_*`, `D3DRS_*`, `D3DTSS_*`, `D3DSAMP_*`, `D3DUSAGE_*`, `D3DPOOL_*`, `D3DRTYPE_*`, `D3DSBT_*`, `D3DGETDATA_*`, `D3DDECLUSAGE_*`, the FF op/arg/compare spaces, HRESULT codes (`hresult.rs`), primitive/FVF flags (`primitive.rs`), caps bits — all live here. `device.rs` / `direct3d9.rs` are glob-exported, so a new `pub const` there needs no `lib.rs` edit; a new module does.
- **Metal wire values** → `unix/shared/src/mtl.rs` (see ARCHITECTURE.md §"Shared wire values are typed in `unix/shared/src/mtl.rs`"). Storage modes, pixel formats, compare funcs, blend factors, primitive types, usage/write-mask bitflags — declared as `#[repr(u32)]` enums or `bitflags!`, never as bare `u32`.

**One principle, two homes: D3D ABI → `mtld3d-types`; Metal wire → `mtl.rs`; never restate either locally.** No function-local or module-local `const D3D*`, no bare ABI integer literal at a call site or `match` arm (`101 =>` becomes `D3DFMT_INDEX16 =>`). A local restatement is a silent-drift hazard: a local `D3DRTYPE_*` copy can drift to the wrong SDK value while the canonical constant stays fixed in its one home.

Width nuance: the types-crate constants are `u32` (matching the D3D9 wire width) except where a narrower field is the natural home (`D3DDECLUSAGE_*` are `u8`). When a `u8`-keyed consumer must `match` against a `u32` constant (the FF emitter stores op/arg codes in `u8` cache-key fields), widen the scrutinee with `u32::from(x)` at the match rather than mirroring the constant — a `match` pattern can't carry an `as` cast, and a parallel `u8` mirror would reintroduce a second definition. The integration-test harness (`windows/tests`) depends only on `mtld3d-types`, so anything it needs must resolve from there, not from `mtld3d-core`.

## Prefer snake_case Rust names

Use snake_case for internal function names, even D3D9/COM impls. When an exported symbol must be PascalCase (e.g. `DllMain`, `Direct3DCreate9`), pin via `#[unsafe(export_name = "DllMain")]` — never `#[allow(non_snake_case)]`.

## Unsafe is a last resort

`unsafe` exists for FFI, COM vtable dispatch, and a small set of documented zero-cost abstractions (`PageBox`, the crumb buffer, the `ffi_boundary` newtypes, `MetalHandle`, `CachedComPtr`). Anywhere else, find a safe alternative first.

### The boundary principle

The canonical pattern in this codebase is **typed boundary newtypes**, not safe-fn helpers. A newtype with an `unsafe fn` constructor and safe methods type-encodes the ABI contract once — the caller writes one `unsafe { Type::new(p) }` per FFI entry, the SAFETY: comment lives at that single contract-assertion site, and downstream methods are safe because the invariant rides on the type.

This mirrors the stdlib's approach. `slice::from_raw_parts`, `Box::from_raw`, `Pin::new_unchecked`, `NonNull::as_ref` are all `unsafe fn`: there's no sound way to give them a safe signature because the caller-upheld preconditions can't be checked at runtime. A "safe" wrapper that internally `unsafe`-deref's a caller-supplied pointer with a documented contract is unsound — the safe signature accepts any input, but only some inputs satisfy the documented contract. The contract assertion belongs at the call site, marked with `unsafe`, not buried behind a safe signature.

### Escalation ladder

Before writing `unsafe`, in order:

1. Is there a safe std primitive? (`Vec`, `slice::from_ref`, `Cell`, `Box::leak`, `core::ptr::NonNull` safe methods in 2024 edition.)
2. Is there a vetted crate that wraps the operation? (`libloading` for `dlsym`, `objc2-*` for selectors, `bytemuck` for POD reinterpret.)
3. Can I model this as a boundary newtype — an `unsafe fn` constructor with safe methods, where the invariant is encoded in the type? (`PageBox`, `Crumb`, `InPtr<T>`, `MetalHandle<K>`, `CachedComPtr<T>`.)
4. Only if 1–3 don't apply: `unsafe { … }` with a SAFETY: comment at the assertion site.

### Mandatory SAFETY: comment

Every `unsafe {}` block carries a `// SAFETY:` comment directly above it. Format:

    // SAFETY: <invariant the unsafe op relies on>; <why it holds here>.
    let q = unsafe { MetalHandle::<MTLCommandQueueKind>::new(raw) };

State what invariant is being asserted ("ptr is non-null and aligned", "the wire u64 carries a previously-retained `id<MTLDevice>`") and why it's true in this context ("checked above by `is_null()`", "IDirect3DDevice9 ABI guarantees `this` is *mut Direct3DDevice9"). Bare "FFI call" or "manual layout" is not acceptable.

### One operation per unsafe block

Don't wrap a 16-line block in one `unsafe {}`. Each pointer deref, each transmute, each FFI call gets its own block with its own SAFETY: comment. Two reasons: (a) `rg 'SAFETY:'` returns a focused answer per operation, (b) refactoring half the block to be safe doesn't strand the comment on the other half.

Acceptable exception: a tight loop where every iteration is the same unsafe op (e.g. `for slot in slots { unsafe { write_volatile(slot, …) } }`).

### Canonical boundary newtypes

Use these in preference to raw pointer juggling. Each lives in `mtld3d_shared` (or `mtld3d-d3d9` for PE-only types) so the same code is shared across the workspaces:

**FFI pointer boundaries** (`mtld3d_shared::ffi_boundary`):

- **`InPtr<'a, T>`** — borrowed input pointer. `unsafe fn opt(p)` filters null; safe `Deref` to `T`. Use at COM vtable entries (`this: *mut c_void` → `&Direct3DDevice9`) and read-only typed in-params.
- **`InPtrMut<'a, T>`** — exclusive borrow. Same shape; `DerefMut` to `T`. Use at unix-call handler params written back to the caller.
- **`ValueIn<'a, T: Copy>`** — by-value typed in-param read (`D3DRECT`, `D3DMATRIX`, …). `unsafe fn opt(p)`; safe `.read()` consumes self and returns `T`. One-shot convenience: `ValueIn::<T>::read_opt(p)`.
- **`OutPtr<'a, T>`** — null-guarded typed out-param write. `unsafe fn opt(out)`; safe `.write(val)` consumes self. One-shot convenience: `OutPtr::write_opt(out, val)`.
- **`VtableThis<'a, T>`** — `IUnknown` vtable `this` cast. Crashes on null per D3D9 spec (preserves crash-on-refcount-miscount). `unsafe fn new(this)`; safe `Deref`/`DerefMut` to `T`. Used at `AddRef`/`Release`/`QueryInterface` thunks where silently filtering null would mask a refcount underflow.

**Metal protocol handles** (`mtld3d_shared::mtl_handle` + `unix/unix/src/metal/handle.rs`):

- **`MetalHandle<K>`** — `#[repr(transparent)]` newtype over `u64` tagged with a marker kind (`MTLDeviceKind`, `MTLTextureKind`, `NSViewKind`, …). Used end-to-end: every typed field in `unix/shared/src/params.rs`, every PE-side slot that stores a Metal protocol-object handle (`DeviceInner`, `FrameData`, `Direct3DSurface9`, `Direct3DTexture9`, `Pass`/`PassState`, `FrameSummaryContext`, `VisibilityQueryCore`, …), and every unix-side helper that crosses the boundary. The `unsafe fn new(raw)` constructor fires at four legitimate boundary shapes: (1) unix-side OUT writes wrapping `Retained::into_raw(...) as u64` once per Create*; (2) PE-side wraps that recover typed handles from u64-bound wire fields by design — `Command::param_b` / `BlitCommand::{src,dst}_handle` (heterogeneous-kind by variant), `get_or_create_texture` returns (still u64-bound for Command construction), and the visibility-buffer batch-create return; (3) tests with opaque test values; (4) nowhere else. Safe `.into_retained()` (defined unix-side, where `objc2-metal` is visible) calls `Retained::retain` on the canonical retain. `unsafe trait ReleaseRetain` at destroy sites consumes the canonical retain via `Retained::from_raw + drop`.

**External COM pointers** (`windows/d3d9/src/com_ref.rs`):

- **`CachedComPtr<T: ComUnknown>`** — owning cached pointer with RAII refcount. `unsafe fn adopt(p)` calls `AddRef`; `Drop` calls `Release`. Use to cache external D3D9 COM pointers we don't own (bound RT/DS/VB/IB). Replaces inline `(*p).vtbl().add_ref(...)` / `release(...)` pairs.

Other concentration examples: `PageBox`, `Crumb`, `ScratchArena`, the per-Win32-function wrappers in `cursor.rs` (`set_cursor`, `delete_object`, `create_bitmap_packed`, …).

### Don't sprinkle — concentrate at three sites

If you find yourself writing the same `unsafe` pattern at three call sites, that's a missing boundary newtype. Add one to the appropriate module rather than letting the unsafe spread.

### `unsafe fn` only when callers must uphold a precondition

`extern "system" fn` and `extern "C" fn` are safe-to-call from Rust; they coerce to `unsafe` fn pointers automatically when stored in vtable arrays. A vtable entry function does **not** need to be `unsafe fn`. Canonical pattern:

    // YES: vtable thunk is `extern "system" fn`; one `unsafe` at the InPtr
    // constructor where the ABI contract is asserted; body is safe.
    extern "system" fn device_set_render_state(this: *mut c_void, state: u32, value: u32) -> i32 {
        // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
        let Some(dev) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
            return D3DERR_INVALIDCALL;
        };
        dev.set_render_state(state, value)
    }

    // NO: don't reach for raw `unsafe { &*(this as *const T) }` in new code —
    // use `InPtr::<T>::opt` (or `VtableThis::<T>::new` for IUnknown thunks).

For `AddRef`/`Release`/`QueryInterface` thunks where D3D9 spec leaves null-`this` UB, use `VtableThis::<T>::new(this)` — it crashes on null by design so a refcount miscount surfaces as a null-deref panic rather than a silent recovery.

### No `unsafe_code` rustc lint

This codebase deliberately does not enable the rustc `unsafe_code` lint. Every crate structurally bears unsafe (FFI extern blocks, COM dispatch, Metal handle conversion, allocator), so blanket per-crate `#![allow(unsafe_code)]` would just be noise. The enforcement is layered: clippy lints (`missing_safety_doc`, `transmute_ptr_to_ref`, `not_unsafe_ptr_arg_deref`) plus `undocumented_unsafe_blocks` and `multiple_unsafe_ops_per_block` — all four are `deny` workspace-wide in `windows/Cargo.toml` and `unix/Cargo.toml`. Together they require every `unsafe {}` block to carry a `// SAFETY:` comment naming the invariant and to perform exactly one unsafe operation; new code that drifts fails `make clippy`.

## Warning suppressions

Default rule: **never** `#[allow(...)]`, `#[expect(...)]`, or `#[cfg_attr(..., allow(...))]`. The lint is followed. `make clippy` runs at `-D warnings` with `clippy::nursery` and `clippy::pedantic` denied workspace-wide and **stays clean by fixing the code, not by silencing the lint**. Every lint is treated as serious — refactor by default; reaching for `#[allow]` is the absolute last resort.

The tree carries exactly **three** per-site `#[allow(clippy::...)]` attributes, plus one workspace-level `too_many_lines = "allow"`. Each is listed under "Accepted per-site exceptions" below. Adding another per-site allow needs a rationale comment that names which of the accepted exception classes applies — and the structural fix (a non-panicking `cast_unsigned`/`to_le_bytes` reinterpret, `try_from(..).expect(..)`, or a literal cross-checked by a `const` assert) clears the great majority of numeric-cast lints without one.

### Standard structural fixes

These handle ~95% of every clippy complaint raised in the tree. Apply the fix; do not allow.

| Lint                              | Fix                                                                                                              |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `cast_possible_truncation`        | `T::try_from(v).expect("<contractual bound>")` — turns silent truncation into a loud panic at the violated bound |
| `cast_possible_wrap`              | same; or fix the storage type at definition                                                                      |
| `cast_sign_loss`                  | `u32::try_from(signed).expect(...)` or `signed.unsigned_abs()`                                                   |
| `cast_precision_loss`             | use a wider float (`f64`) at the source; for D3DCOLOR-style u32→u8 unpack use `to_le_bytes` destructure; for u64→f64 use the `u64_to_f64_exact` / `usize_to_f64_exact` helpers in `tsc.rs` (hi/lo split + `mul_add`) |
| `cast_ptr_alignment`              | `core::ptr::read_unaligned` instead of a direct pointer cast                                                     |
| `too_many_arguments`              | introduce a `FooParams` / `FooCreateInfo` struct                                                                 |
| `items_after_statements`          | hoist `const` / `type` / `use` to function top, or to file scope                                                 |
| `similar_names`                   | rename one of the bindings                                                                                       |
| `many_single_char_names`          | rename to descriptive names — or delete the dead code carrying them                                              |
| `needless_pass_by_value`          | take `&T`; ctors that previously took owned `Info` structs take `&Info`                                          |
| `needless_pass_by_ref_mut`        | take `&self` / `&T`                                                                                              |
| `missing_const_for_fn`            | add `const fn` (including `const extern "system" fn`)                                                            |
| `unnecessary_wraps`               | drop the `Option` / `Result` wrap; let callers wrap at the boundary                                              |
| `unnecessary_box_returns`         | return `Self`; let callers `Box::new(value)` before `Box::into_raw`                                              |
| `struct_excessive_bools`          | convert to `bitflags!` or a packed `u8` / `u16` / `u64` per §"Booleans pack"                                     |
| `pub_underscore_fields`           | rename the field (drop the `_` prefix)                                                                           |
| `match_same_arms`                 | restructure: `let-else` for the catch-all arm, exhaustive `match` for the rest                                   |
| `float_cmp` (in tests)            | `.to_bits()` comparisons — `assert_eq!(f.to_bits(), v.to_bits())`                                                |
| `doc_markdown`, `assigning_clones`, `too_long_first_doc_paragraph` | rewrite the doc / code                                                          |
| `missing_errors_doc`, `missing_panics_doc`, `missing_safety_doc` | add the doc                                                                       |

### Accepted per-site exceptions

1. **`unix/unix/src/metal/macdrv.rs` `mod bounded_cast`** — `cast_possible_truncation` + `cast_sign_loss` + `cast_precision_loss`, mod-level. Three numeric-cast helpers (`f64_to_u32_saturating`, `f64_to_f32`, `i32_to_f32`) whose bounds are established by callers. The structural alternative (IEEE-754 bit-manipulation reimplementing `as`) is ~25 lines of mantissa/exponent extraction producing identical bits with zero precision benefit. Each fn doc names the bound; callers document why their input satisfies it.
2. **`unix/unix/src/metal/command.rs` `PendingCmdBuf`** — `non_send_fields_in_send_ty`. Wraps `Retained<dyn MTLCommandBuffer>`; the three ops we use (refcount inc/dec, `waitUntilCompleted`) are Apple-documented thread-safe. Alternatives are `SendWrapper` (panics on cross-thread access) or raw `usize` + `Retained::retain` everywhere — both worse than a well-documented `unsafe impl Send` at one narrow site.
3. **`windows/core/src/config.rs` `Mtld3dConfig`** — `struct_excessive_bools`. The config is a flat one-`bool`-per-`mtld3d.conf`-key bag; each field name mirrors a file key (`caps_all` ↔ `debug.capsAll`). Packing into `bitflags!` (the §"Booleans pack" default) would decouple the field names from the conf keys for no hot-path benefit — `Mtld3dConfig` is read once at startup, never per-draw. The clarity of the 1:1 mapping outweighs the byte savings, so the bools stay flat.

**Note on `as`:** even when an `as` cast is identity at compile time (e.g. `u64 as usize` on a 64-bit-only build), prefer `usize::try_from(v).expect(...)` with a one-line rationale. `as` is the smell; `try_from` keeps the bound check explicit and survives a future port to a 32-bit unix host without silent truncation.

### Adding a new per-site allow

Acceptable only when **all** of:

1. The lint is in `clippy::nursery` or `clippy::pedantic` (never `complexity`, `correctness`, `perf`, `style`, `suspicious`, or any rustc warning).
2. The structural fix from the table above has been attempted and produces verbose code with identical machine behaviour, or defeats a safety mechanism / breaks a binary contract / forces every caller through the same hack.
3. A `// allow: <reason>` comment (or doc comment on the surrounding fn / mod) names which of (2) applies and why the table entry doesn't.

A single `// allow:` comment may cover a contiguous run of `#[allow(...)]` attributes on the same item when the same rationale applies. Prefer the multi-lint form `#[allow(a, b, c)]` or a mod-level `#[allow(...)]` covering the helpers (see `bounded_cast`).

### Workspace-level

Both workspace `[lints.clippy]` blocks — `windows/Cargo.toml` and `unix/Cargo.toml`, inherited by every member via `[lints] workspace = true` — deny `nursery` + `pedantic`. The **only** workspace-level allow in the tree is `too_many_lines = "allow"`. Rationale: line count is a poor proxy for complexity. The lints that catch real complexity (`cognitive_complexity`, `cyclomatic_complexity`) stay denied; long but linear procedural functions (frame encoder, MSL codegen templates, per-opcode dispatchers, test narratives) are left intact rather than fragmented into helpers that thread the same state.

Adding any other `[workspace.lints] foo = "allow"` requires the same justification as a per-site allow plus an argument that it applies uniformly across every member crate.

## No silent failures — use `log_once_warn!`

Every stub, unimplemented branch, default `_ =>` arm, silent early-return, and silent fallback must log. Silent fallthroughs hide real bugs behind a clean `RUST_LOG=info`.

Macros live in `unix/shared/src/log_helpers.rs`, exported via `#[macro_export]`. Every crate calls them as `mtld3d_shared::log_once_warn!(target: LOG_TARGET, "…")` (or `log_once_warn_by!`, `log_once_trace_by!`, `log_once_info!`, `log_once_info_by!`). Backed by per-call-site `static AtomicBool` — fires once per process per site.

When to `log_once_warn!`: COM vtable stubs returning a constant; `match` arms falling through because the input isn't handled yet; `else { return }` short-circuits dropping a draw / skipping a binding / swallowing state; FFI/Metal helpers with safe-fallback returns for unmapped values.

When to `log_once_info!` instead: the D3D9 no-op IS the complete correct Metal-world behaviour — VRAM-residency hints (`PreLoad`, `Set/GetPriority`, `EvictManagedResources`), obsolete features modern drivers also ignore (`GetNPatchMode`). Info only when (a) Metal has no equivalent or the feature is obsolete on all modern drivers, (b) the returned value matches a real driver, and (c) returning it causes no visible misbehaviour.

When NOT to use either: legitimate by-design no-ops (cache hits, `None`-then-`continue` in a search) — comment instead. Real error paths that should fire every time — keep `error!`.

Format: `"stub IDirect3DDevice9::Foo → ReturnCode"`, `"IDirect3D<Type>9::<Method>: no Metal analog, no-op"`, `"<module>: <what happened> → <fallback>"`. One line; once-per-site means it can be descriptive.

## No raw `msg_send!` — use typed `objc2-*` bindings

Every Obj-C selector goes through a typed method on an `objc2-*` framework crate (`objc2-metal`, `objc2-quartz-core`, `objc2-foundation`, `objc2-core-foundation`, `objc2-core-graphics`, `objc2-app-kit`). Selector name and return type become compile errors instead of runtime `unrecognized selector` crashes, and the per-call `unsafe { … }` wrapping disappears for safe selectors.

Untyped `msg_send!` is a silent-drift risk. A selector typo (`setDisplaySyncEnabld:`) compiles, ships, and fires the first time the user resizes a window. Return-type mismatches are equally undetectable until a corrupted value shows up downstream. The objc2 framework crates are already wired in `unix/Cargo.toml`, so reaching for `msg_send!` is leaving free type-checking on the table.

This is one instance of the general rule in §"Unsafe is a last resort": prefer typed safe wrappers over raw unsafe. `msg_send!` is an untyped unsafe surface that already has typed alternatives.

How to apply:

- New selector → use the typed method from the right `objc2-*` crate. If the feature isn't enabled yet, add it to the right entry under `[workspace.dependencies]` in `unix/Cargo.toml`.
- Need a class from an `objc2-*` framework crate not yet wired? Add it with `default-features = false` + an explicit feature list — these crates default to "everything on" (`objc2-app-kit` alone has hundreds of default features), which inflates compile time.
- Feature dependencies are not transitive: enabling `NSView` requires `NSResponder`; methods returning `CGFloat`/`NSRect` need the `objc2-core-foundation` feature on `objc2-app-kit`. The compile error names the missing item; the crate's `Cargo.toml` `[features]` section names the gate.
- `*mut c_void` from FFI → `Retained::retain(ptr.cast::<TypedClass>())`, then call typed methods. Pattern used throughout `macdrv.rs`.
- Protocol inheritance (e.g. handing a `CAMetalDrawable` to `MTLCommandBuffer::presentDrawable`) uses `ProtocolObject::from_ref(&*sub_obj)` — sound because the sub-protocol trait extends the super-protocol trait. Type inference at the call site picks the target protocol.
- `MainThreadOnly` classes (`NSScreen` / `NSView` / `NSWindow` / most of AppKit): mtld3d runs the API thread off the AppKit main thread, so class methods that require `MainThreadMarker` need `unsafe { MainThreadMarker::new_unchecked() }` with a SAFETY comment naming the read-only property being queried.

Hard rule: no new `msg_send!`, `class!`, or `sel!` callsites — they're a grep away from being banned mechanically. If a binding is genuinely missing from objc2 framework crates, declare it locally via `objc2::extern_class!` / `extern_methods!` (the same macros the framework crates use) so the surface stays typed.

This is one instance of the general rule in §"Unsafe is a last resort": prefer typed safe wrappers over raw unsafe. `msg_send!` is unsafe surface that already has typed alternatives.

## Doc comments

- Use `///` for items, `//` for fn-body and inline notes.
- Doc first paragraph stays brief (avoids `clippy::too_long_first_doc_paragraph`).
- Identifiers in doc comments use backticks; clones into existing buffers use `clone_into` (avoids `clippy::doc_markdown` and `clippy::assigning_clones`).
- Default to writing no comment at all. Add one when the *why* is non-obvious: a hidden constraint, a subtle invariant, a workaround for a specific behaviour, a surprise. Don't explain *what* the code does — well-named identifiers already do that.

### Shape: title, blank line, body

Every doc block of **two or more lines** — `///` or `//!`, on any item, including struct fields and enum variants — is shaped:

1. Line 1 is a **single-line title**: one sentence, no wrap, within the 100 columns rustfmt gives code.
2. Line 2 is an **empty doc line**.
3. The body follows.

A one-line doc comment is already a title and needs nothing. Blocks whose content is only a `# Safety` / `# Errors` / `# Panics` section still get a title line above it.

The title is what rustdoc puts in the summary column of every index, and what a reader sees before deciding to keep reading. A wrapped opening sentence gives them a fragment. `make audit` enforces the shape; body prose stays hard-wrapped at the file's existing width.

```rust
/// A standalone `CreateRenderTarget` colour surface.
///
/// `parent_texture` is null (so it is not texture-backed) but it carries its
/// own persistent `metal_color_handle` distinct from the backbuffer, plus its
/// own format and dimensions.
```

## Dependencies

- All third-party versions live in `[workspace.dependencies]` at the workspace root. Member crates use `{ workspace = true }`.
- Both `Cargo.lock` files are committed for reproducible releases. Bump via `make upgrade` (semver-compatible) or `make upgrade-incompat` (with `cargo-edit`).

## Imports

No glob imports. Explicit named imports only — never `use foo::*`. Two exceptions:

- A `pub use submodule::*` re-export of a crate's **own** constant-definition modules (`mtld3d-types`' `lib.rs` re-exports `device::*` / `direct3d9::*`): the glob is the crate's public API surface, and it lets a new `pub const` land in those modules without a parallel `lib.rs` edit. That is a re-export of in-crate items, not a glob *import* of another module's names into local scope.
- `use super::*` at the top of a `#[cfg(test)] mod tests` block: the standard Rust idiom for pulling the module-under-test's items into its unit tests. The test module is private, so nothing leaks past the crate, and the glob tracks the parent's surface without churn as items come and go.

## Inline attributes

Default `#[inline]`, not `#[inline(always)]`. Thin-LTO inlines small functions on its own; `#[inline(always)]` is reserved for cases where measurement proves it pays.

## `LazyLock` over `OnceLock`

`LazyLock` is the default when the initializer is a static `fn` (stable since 1.80). Reach for `OnceLock` only when the initializer needs runtime arguments.
