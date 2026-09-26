//! Cold start with a populated shader cache: fresh processes from launch to their first frame.
//!
//! What a game pays between being launched and showing its first frame once
//! the cache beside it is full: loading `d3d9.dll`, `Direct3DCreate9`,
//! `CreateDevice`, the prewarm thread recreating every recorded shader and
//! pipeline, and the first `Present`. Each measurement needs a process of
//! its own, so the benchmark runs its workload in private copies of this
//! test executable in a directory of their own. The layer keeps
//! `mtld3d_shaders.bin` beside the executable, so the cache is private to
//! the benchmark and its path is known, and the name of each copy says
//! which part it plays.
//!
//! A priming process draws a mix shaped like World of Warcraft 1.12's
//! startup for a few frames under `shaderCache.enable=true`, which fills
//! the cache: [`SM2_PAIRS`] `vs_2_0`/`ps_2_0` pairs and [`FF_COMBOS`]
//! fixed-function combinations, about 100 shaders, each drawn under the
//! four [`BLENDS`], about 200 pipelines. One untimed process then starts on
//! that cache, which compacts it as a game's second launch does, and the
//! compacted file is put back before each of [`RUNS`] measured processes,
//! so every one starts from the same bytes. A measured process creates the
//! device, draws the mix once, presents, and reads the back buffer back,
//! which waits for the encoder and so for the prewarm the encoder waits
//! for. It times each step with the benchmarks' `rdtsc` clock (`TscClock`),
//! calibrated after the last step so its sleep is in none of them, and
//! prints one line the parent parses. The time from launch is the child's
//! wall clock at a step less the parent's just before the spawn, both read
//! from the one host clock, since `rdtsc` counts are not compared across
//! processes. The parent
//! takes the prewarm counts and time from the child's layer log and checks
//! the cache after each run: a record a run appended is a shader or a
//! pipeline the prewarm did not provide.
//!
//! Real caches are measured the same way. `make bench BENCH_CORPUS=...`
//! copies each named cache to `corpus/<name>/mtld3d_shaders.bin` under the
//! report directory, `<name>` the one the Makefile's corpus rule gives it,
//! and each is measured beside a private executable that only clears and
//! presents, so the prewarm of that cache is the work, and reported as
//! `cold_start_<name>`. A corpus whose header names another
//! format or schema than this build's is one the layer would wipe: it is
//! skipped, and the `cold_start` report says so.

use core::fmt::Write as _;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use mtld3d_core::shader_cache::{self, CacheHeader, CacheLoad};
use mtld3d_tests::{
    Harness, HarnessConfig, IndexBuffer, MemorySample, PixelShader, Texture, VertexBuffer,
    VertexDeclaration, VertexShader, run_child,
};
use mtld3d_types::{
    D3DBLEND_DESTCOLOR, D3DBLEND_INVSRCALPHA, D3DBLEND_ONE, D3DBLEND_SRCALPHA, D3DBLEND_ZERO,
    D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DFMT_D24S8, D3DFMT_INDEX16, D3DFOG_EXP, D3DFOG_LINEAR,
    D3DFOG_NONE, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DPOOL_MANAGED,
    D3DPRESENT_INTERVAL_IMMEDIATE, D3DPT_TRIANGLELIST, D3DRS_ALPHABLENDENABLE, D3DRS_DESTBLEND,
    D3DRS_FOGENABLE, D3DRS_FOGVERTEXMODE, D3DRS_LIGHTING, D3DRS_SPECULARENABLE, D3DRS_SRCBLEND,
    D3DTA_DIFFUSE, D3DTA_TEXTURE, D3DTOP_ADD, D3DTOP_ADDSIGNED, D3DTOP_MODULATE, D3DTOP_MODULATE2X,
    D3DTOP_MODULATE4X, D3DTOP_SELECTARG1, D3DTOP_SELECTARG2, D3DTOP_SUBTRACT, D3DTSS_ALPHAARG1,
    D3DTSS_ALPHAARG2, D3DTSS_ALPHAOP, D3DTSS_COLORARG1, D3DTSS_COLORARG2, D3DTSS_COLOROP,
    D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_COUNT2, D3DTTFF_DISABLE, D3DUSAGE_WRITEONLY,
};

use crate::bench::{
    Class, Direction, IDENTITY_ROWS, LayerLog, Metrics, Model, STRIDE, TEXTURED_DECL, TscClock,
    Value, grid, log_dir, material_ps, material_vs, ms, nearest_rank, ok, pattern_texture, ratio,
    rs, world_rows, write_report,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Measured processes per cache, after the untimed one that compacts it.
const RUNS: usize = 7;
/// Frames the priming process draws, so the cache holds everything the mix builds.
const PRIME_FRAMES: u32 = 3;
/// The programmable pairs of the mix, one `vs_2_0` and one `ps_2_0` each.
const SM2_PAIRS: u32 = 25;
/// The fixed-function combinations of the mix: lighting, specular, texture transform and fog.
const FF_COMBOS: u32 = 24;
/// The stage-0 colour operations the fixed-function combinations cycle through.
const COLOR_OPS: [u32; 8] = [
    D3DTOP_MODULATE,
    D3DTOP_SELECTARG1,
    D3DTOP_SELECTARG2,
    D3DTOP_ADD,
    D3DTOP_MODULATE2X,
    D3DTOP_MODULATE4X,
    D3DTOP_SUBTRACT,
    D3DTOP_ADDSIGNED,
];
/// The stage-0 alpha operations, one per eight combinations.
const ALPHA_OPS: [u32; 3] = [D3DTOP_MODULATE, D3DTOP_SELECTARG1, D3DTOP_SELECTARG2];
/// The vertex fog modes, one per eight combinations.
const FOG_MODES: [u32; 3] = [D3DFOG_NONE, D3DFOG_LINEAR, D3DFOG_EXP];
/// Blend enable, source and destination factor: every program is drawn under each.
///
/// The factors are set even with blending off, so the state a draw sees
/// does not depend on the draw before it.
const BLENDS: [(u32, u32, u32); 4] = [
    (0, D3DBLEND_ONE, D3DBLEND_ZERO),
    (1, D3DBLEND_SRCALPHA, D3DBLEND_INVSRCALPHA),
    (1, D3DBLEND_ONE, D3DBLEND_ONE),
    (1, D3DBLEND_DESTCOLOR, D3DBLEND_ZERO),
];
const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1;
/// The priming copy of the executable: draws the mix for [`PRIME_FRAMES`] frames.
const PRIME_EXE: &str = "cold-start-prime.exe";
/// The measured copy for the synthetic cache: draws the mix once.
const DRAW_EXE: &str = "cold-start.exe";
/// The measured copy for a corpus: clears and presents, drawing nothing.
const CLEAR_EXE: &str = "cold-start-clear.exe";
/// What names the temporary directories of this benchmark's runs.
const TEMP_PREFIX: &str = "bench-cold-start-";
/// The file name of each corpus cache the Makefile stages under `corpus/<name>/`.
const CORPUS_FILE: &str = "mtld3d_shaders.bin";
/// This test, which each copy runs.
const TEST_PATH: &str = "bench_cold_start::cold_start";
/// The children's own entries: the cache on, and the layer log beside the copy.
const CHILD_ENTRIES: &str = "shaderCache.enable=true;log.dir=";
/// The children's log filter, which keeps the prewarm's info lines whatever the caller's says.
const CHILD_LOG_FILTER: &str = "mtld3d::d3d9=info";
/// What opens the line a child prints its timings on.
const CHILD_LINE: &str = "cold_start_child ";
/// What sits between the shader count and its time on the prewarm's shader line.
const PREWARM_SHADERS: &str = " pre-warmed in ";
/// What precedes the pipeline count on the prewarm's closing line.
const PREWARM_PIPELINES: &str = "shader_cache: pre-warmed ";
/// The line the prewarm logs for each pipeline recipe whose build failed.
const PREWARM_FAILED: &str = "shader_cache: pipeline prewarm failed";
/// What opens the line the prewarm logs once for each recipe it skips for a shader that failed.
const PREWARM_SKIPPED: &str = "shader_cache: pipeline recipe skipped after";
/// The metric and the report label of each count a [`Prewarm`] carries, in its order.
const PREWARM_COUNTS: [(&str, &str); 3] = [
    ("prewarm.shaders", "shaders"),
    ("prewarm.pipelines", "pipelines"),
    ("prewarm.no_color", "no-color mappings"),
];
/// The `mem.ready.*` rows of a child's memory numbers, in the order of [`ChildLine::memory`].
const MEMORY_ROWS: [(&str, Direction); 4] = [
    ("committed_mib", Direction::Lower),
    ("reserved_mib", Direction::Lower),
    ("largest_free_mib", Direction::Higher),
    ("peak_ws_mib", Direction::Lower),
];
/// The timed steps of a measured process, each a report row and a `<metric>.p50`/`.max` pair.
const STEPS: [Step; 6] = [
    Step {
        metric: "start_to_test",
        label: "launch to the test body",
        pick: |run| run.child.to_test,
    },
    Step {
        metric: "create_device",
        label: "Direct3DCreate9, window and CreateDevice",
        pick: |run| run.child.line.create,
    },
    Step {
        metric: "first_present",
        label: "CreateDevice to the first Present",
        pick: |run| run.child.line.first_present,
    },
    Step {
        metric: "start_to_first_present",
        label: "launch to the first Present",
        pick: |run| run.child.to_first_present,
    },
    Step {
        metric: "ready",
        label: "CreateDevice to the first frame read back",
        pick: |run| run.child.line.ready,
    },
    Step {
        metric: "start_to_ready",
        label: "launch to the first frame read back",
        pick: |run| run.child.to_ready,
    },
];

/// Launch to first frame over fresh processes, on the synthetic cache and on each corpus.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn cold_start() {
    let exe = std::env::current_exe().expect("resolve test executable");
    if let Some(role) = Role::of(&exe) {
        child(&role);
        return;
    }
    // Creating an interface names this test as in flight, as every test
    // that reaches the layer does.
    let _factory = Harness::factory_only();
    let root = TempRoot::new();
    let root = root.path();

    let synthetic = Sandbox::new(&root.join("synthetic"), &exe, DRAW_EXE);
    fs::copy(&exe, synthetic.dir.join(PRIME_EXE)).expect("copy the priming executable");
    let prime = spawn(&synthetic.dir.join(PRIME_EXE), &synthetic.dir);
    let primed = synthetic
        .settle()
        .unwrap_or_else(|reason| panic!("the priming process left no usable cache: {reason}"));
    let measured = synthetic.measure(primed);

    let mut notes = String::new();
    let (mut corpora, mut skipped) = (0_u64, 0_u64);
    for (name, source) in corpus_files() {
        let sandbox = Sandbox::new(&root.join(format!("corpus-{name}")), &exe, CLEAR_EXE);
        fs::copy(&source, &sandbox.cache).expect("copy a corpus cache");
        let settled = compatible(&sandbox.cache).and_then(|()| sandbox.settle());
        match settled {
            Ok(settled) => {
                corpora += 1;
                let measured = sandbox.measure(settled);
                let _ = writeln!(
                    notes,
                    "corpus {name}: measured as cold_start_{name} ({})",
                    source.display()
                );
                report(
                    &format!("cold_start_{name}"),
                    &format!(
                        "shape: corpus {source}, each measured process clears, presents and reads \
                         back, drawing nothing\n",
                        source = source.display()
                    ),
                    &measured,
                    &Kind::Corpus,
                );
            }
            Err(reason) => {
                skipped += 1;
                let _ = writeln!(
                    notes,
                    "corpus {name}: skipped, {reason} ({})",
                    source.display()
                );
            }
        }
    }
    if corpora + skipped == 0 {
        notes.push_str("corpus: none under corpus/ (make bench BENCH_CORPUS=...)\n");
    }

    let shape = format!(
        "shape: back buffer {WIDTH}x{HEIGHT}, shaderCache.enable=true; the mix is {SM2_PAIRS} \
         vs_2_0/ps_2_0 pairs and {FF_COMBOS} fixed-function combinations, each under {blends} \
         blend states ({draws} draws); the priming process drew it for {PRIME_FRAMES} frames \
         (first Present {prime_present:.3} ms after CreateDevice), each measured process draws \
         it once before its first Present\n{notes}",
        blends = BLENDS.len(),
        draws = (SM2_PAIRS + FF_COMBOS) * 4,
        prime_present = ms(prime.line.first_present),
    );
    report(
        "cold_start",
        &shape,
        &measured,
        &Kind::Synthetic { corpora, skipped },
    );
}

/// What a report covers, and so which extra metrics it carries.
enum Kind {
    /// The synthetic cache, with the counts of corpora measured and skipped.
    Synthetic { corpora: u64, skipped: u64 },
    /// A real cache from `corpus/`.
    Corpus,
}

/// The temporary directory the private copies run in, removed when dropped.
///
/// A failed assertion in a test process that has installed the harness's
/// failure hook ends the process without unwinding, so the drop does not run
/// then; [`TempRoot::new`] removes what such a run left before it creates
/// its own.
struct TempRoot {
    path: PathBuf,
}

impl TempRoot {
    /// Remove earlier runs' directories and create this run's, named after the process and time.
    fn new() -> Self {
        let temp = std::env::temp_dir();
        for entry in fs::read_dir(&temp).into_iter().flatten().flatten() {
            if entry.file_name().to_string_lossy().starts_with(TEMP_PREFIX) {
                let _ = fs::remove_dir_all(entry.path());
            }
        }
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        let path = temp.join(format!("{TEMP_PREFIX}{}-{stamp}", std::process::id()));
        fs::create_dir_all(&path).expect("create the private cold-start directory");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// The part a private copy of the test executable plays, by its file name.
enum Role {
    /// Draw the mix for [`PRIME_FRAMES`] frames, filling the cache.
    Prime,
    /// Draw the mix once before the first `Present`.
    Draw,
    /// Clear and present, drawing nothing.
    Clear,
}

impl Role {
    fn of(exe: &Path) -> Option<Self> {
        let name = exe.file_name()?.to_str()?;
        [
            (PRIME_EXE, Self::Prime),
            (DRAW_EXE, Self::Draw),
            (CLEAR_EXE, Self::Clear),
        ]
        .into_iter()
        .find_map(|(exe, role)| name.eq_ignore_ascii_case(exe).then_some(role))
    }
}

/// Run the child's part and print its timings on one [`CHILD_LINE`].
///
/// The durations are nanoseconds from the start of the test body; the
/// `*_at` stamps are the wall clock as nanoseconds since the Unix epoch,
/// for the parent to set against its own stamp from before the spawn.
fn child(role: &Role) {
    let entered_at = unix_nanos(SystemTime::now());
    let started = TscClock::now();
    let h = Harness::create(&HarnessConfig {
        width: WIDTH,
        height: HEIGHT,
        depth_format: Some(D3DFMT_D24S8),
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        ..HarnessConfig::default()
    });
    let created = TscClock::now();
    let mix = match role {
        Role::Prime | Role::Draw => Some(Mix::new(&h)),
        Role::Clear => None,
    };
    let frames = match role {
        Role::Prime => PRIME_FRAMES,
        Role::Draw | Role::Clear => 1,
    };
    let mut first = None;
    for _ in 0..frames {
        assert!(h.pump(), "WM_QUIT before the first frame");
        ok(h.begin_scene(), "BeginScene");
        ok(
            h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0xFF20_3040, 1.0, 0),
            "clear",
        );
        if let Some(mix) = &mix {
            mix.draw();
        }
        ok(h.end_scene(), "EndScene");
        ok(h.present(), "Present");
        first.get_or_insert_with(|| (TscClock::now(), unix_nanos(SystemTime::now())));
    }
    let (presented, presented_at) = first.expect("a child draws at least one frame");
    let _pixel = h.read_pixel(0, 0);
    let ready = TscClock::now();
    let ready_at = unix_nanos(SystemTime::now());
    let memory = MemorySample::now();
    // Calibrated only now, so its sleep falls after every stamp the parent
    // sets against its own clock.
    let _clock = TscClock::calibrated();
    let nanos =
        |ticks: u64| u64::try_from(TscClock::duration(ticks).as_nanos()).unwrap_or(u64::MAX);
    println!(
        "{CHILD_LINE}entered_at={entered_at} create={create} first_present={present} \
         ready={ready} first_present_at={presented_at} ready_at={ready_at} \
         committed={committed} reserved={reserved} largest_free={free} peak_ws={peak}",
        create = nanos(created.saturating_sub(started)),
        present = nanos(presented.saturating_sub(created)),
        ready = nanos(ready.saturating_sub(created)),
        committed = memory.committed(),
        reserved = memory.reserved(),
        free = memory.largest_free(),
        peak = memory.peak_working_set(),
    );
}

/// The mix a child draws: the programmable pairs, then the fixed-function combinations.
struct Mix<'h> {
    h: &'h Harness,
    texture: Texture<'h>,
    vb: VertexBuffer<'h>,
    ib: IndexBuffer<'h>,
    decl: VertexDeclaration<'h>,
    programs: Vec<(VertexShader<'h>, PixelShader<'h>)>,
    vertices: u32,
    triangles: u32,
}

impl<'h> Mix<'h> {
    fn new(h: &'h Harness) -> Self {
        let (vertices, indices) = grid(2);
        let vertex_count = u32::try_from(vertices.len()).expect("mesh fits u32");
        let triangles = u32::try_from(indices.len() / 3).expect("mesh fits u32");
        let vb = h.create_vertex_buffer(
            vertex_count * STRIDE,
            D3DUSAGE_WRITEONLY,
            0,
            D3DPOOL_MANAGED,
        );
        vb.lock(0, 0, 0).write(&vertices);
        let ib = h.create_index_buffer(
            triangles * 6,
            D3DUSAGE_WRITEONLY,
            D3DFMT_INDEX16,
            D3DPOOL_MANAGED,
        );
        ib.lock(0, 0, 0).write(&indices);
        // Every third pixel program also samples `s1`, as a shadow receiver does.
        let programs = (0..SM2_PAIRS)
            .map(|at| {
                (
                    h.create_vertex_shader(&material_vs(&Model::Sm2, ratio(at + 1, 1000))),
                    h.create_pixel_shader(&material_ps(
                        &Model::Sm2,
                        [ratio(at, SM2_PAIRS), 0.5, 0.25, 0.0],
                        at.is_multiple_of(3),
                    )),
                )
            })
            .collect();
        let mix = Self {
            h,
            texture: pattern_texture(h, 0xFF40_8020),
            vb,
            ib,
            decl: h.create_vertex_declaration(&TEXTURED_DECL),
            programs,
            vertices: vertex_count,
            triangles,
        };
        for (state, value) in [
            (D3DTSS_COLORARG1, D3DTA_TEXTURE),
            (D3DTSS_COLORARG2, D3DTA_DIFFUSE),
            (D3DTSS_ALPHAARG1, D3DTA_TEXTURE),
            (D3DTSS_ALPHAARG2, D3DTA_DIFFUSE),
        ] {
            ok(h.set_texture_stage_state(0, state, value), "stage argument");
        }
        mix
    }

    /// Draw every program under every blend state, in the same order and from the same state.
    fn draw(&self) {
        let h = self.h;
        rs(h, D3DRS_FOGENABLE, 0);
        rs(h, D3DRS_LIGHTING, 0);
        rs(h, D3DRS_SPECULARENABLE, 0);
        ok(
            h.set_texture_stage_state(0, D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_DISABLE),
            "texture transform off",
        );
        ok(h.set_texture(0, &self.texture), "texture");
        ok(h.set_texture(1, &self.texture), "shadow texture");
        ok(h.set_stream_source(0, &self.vb, 0, STRIDE), "stream");
        ok(h.set_indices(&self.ib), "indices");
        ok(h.set_vertex_declaration(&self.decl), "declaration");
        ok(
            h.set_vertex_shader_constant_f(0, &IDENTITY_ROWS),
            "view-projection",
        );
        ok(h.set_pixel_shader_constant_f(0, &[1.0; 4]), "tint");
        let mut spot = 0;
        for (vs, ps) in &self.programs {
            ok(h.set_vertex_shader(vs), "VS");
            ok(h.set_pixel_shader(ps), "PS");
            for blend in &BLENDS {
                self.blend(blend);
                let (x, y) = place(spot);
                ok(
                    h.set_vertex_shader_constant_f(4, &world_rows(0.1, x, y, 0.5)),
                    "world",
                );
                self.draw_mesh();
                spot += 1;
            }
        }
        ok(h.clear_vertex_shader(), "fixed-function VS");
        ok(h.clear_pixel_shader(), "fixed-function PS");
        ok(h.set_fvf(FVF), "SetFVF");
        for combo in 0..FF_COMBOS {
            let index = usize::try_from(combo).expect("combination index fits usize");
            rs(h, D3DRS_LIGHTING, combo & 1);
            rs(h, D3DRS_SPECULARENABLE, (combo >> 1) & 1);
            let transform = if combo & 4 == 0 {
                D3DTTFF_DISABLE
            } else {
                D3DTTFF_COUNT2
            };
            ok(
                h.set_texture_stage_state(0, D3DTSS_TEXTURETRANSFORMFLAGS, transform),
                "texture transform",
            );
            let fog = FOG_MODES[index / 8];
            rs(h, D3DRS_FOGENABLE, u32::from(fog != D3DFOG_NONE));
            rs(h, D3DRS_FOGVERTEXMODE, fog);
            ok(
                h.set_texture_stage_state(0, D3DTSS_COLOROP, COLOR_OPS[index % 8]),
                "colour op",
            );
            ok(
                h.set_texture_stage_state(0, D3DTSS_ALPHAOP, ALPHA_OPS[index / 8]),
                "alpha op",
            );
            for blend in &BLENDS {
                self.blend(blend);
                self.draw_mesh();
            }
        }
    }

    fn blend(&self, &(enable, source, destination): &(u32, u32, u32)) {
        rs(self.h, D3DRS_ALPHABLENDENABLE, enable);
        rs(self.h, D3DRS_SRCBLEND, source);
        rs(self.h, D3DRS_DESTBLEND, destination);
    }

    fn draw_mesh(&self) {
        ok(
            self.h.draw_indexed_primitive(
                D3DPT_TRIANGLELIST,
                0,
                0,
                self.vertices,
                0,
                self.triangles,
            ),
            "draw",
        );
    }
}

/// A private directory holding one copy of the test executable and the cache beside it.
struct Sandbox {
    dir: PathBuf,
    /// The copy each measured process runs.
    exe: PathBuf,
    /// `mtld3d_shaders.bin` beside the copies, the cache every one of them reads.
    cache: PathBuf,
    /// The settled cache, put back before each measured process.
    snapshot: PathBuf,
}

impl Sandbox {
    /// Create `dir` and copy `exe` into it as `name`.
    fn new(dir: &Path, exe: &Path, name: &str) -> Self {
        fs::create_dir_all(dir).expect("create a private cold-start directory");
        let copy = dir.join(name);
        fs::copy(exe, &copy).expect("copy the measured executable");
        Self {
            dir: dir.to_path_buf(),
            exe: copy,
            cache: dir.join("mtld3d_shaders.bin"),
            snapshot: dir.join("settled.bin"),
        }
    }

    /// One untimed process on the cache, then keep what it left as the snapshot.
    ///
    /// # Errors
    /// Names why the cache is not usable when no current cache is left.
    fn settle(&self) -> Result<Settled, String> {
        let warm_up = spawn(&self.exe, &self.dir);
        let mut notes = String::new();
        if warm_up.log.contains("shader_cache: compacted ") {
            notes.push_str("; the untimed process compacted the cache");
        }
        let regenerated = warm_up.log.contains("shader_cache: regenerated ");
        if regenerated {
            notes.push_str("; the untimed process regenerated stale MSL");
        }
        fs::copy(&self.cache, &self.snapshot).map_err(|error| format!("no cache left: {error}"))?;
        let counts = Counts::of(&self.snapshot)?;
        if counts.appended {
            notes.push_str(
                "; the kept cache is not one bundle, so each measured process compacts it",
            );
        }
        let bytes = fs::metadata(&self.snapshot).map_or(0, |meta| meta.len());
        Ok(Settled {
            counts,
            bytes,
            regenerated,
            notes,
        })
    }

    /// The measured processes, each started on the snapshot.
    fn measure(&self, settled: Settled) -> Measured {
        let mut runs = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            fs::copy(&self.snapshot, &self.cache).expect("put the settled cache back");
            let child = spawn(&self.exe, &self.dir);
            let after = Counts::of(&self.cache).unwrap_or_else(|reason| {
                panic!("a measured process left no current cache: {reason}")
            });
            runs.push(Run {
                appended: after.records().saturating_sub(settled.counts.records()),
                grew: after.appended,
                prewarm: Prewarm::of(&child.log),
                child,
            });
        }
        let log = runs
            .last()
            .map_or_else(String::new, |run| run.child.log.clone());
        Measured { settled, runs, log }
    }
}

/// The settled cache every measured process starts from.
struct Settled {
    counts: Counts,
    bytes: u64,
    /// Whether the untimed process regenerated MSL an older emitter wrote.
    regenerated: bool,
    /// What the untimed process did to the cache, for the report.
    notes: String,
}

/// The records a cache holds.
struct Counts {
    shaders: usize,
    ff_shaders: usize,
    pipelines: usize,
    /// Whether anything was appended since the file was last compacted.
    appended: bool,
}

impl Counts {
    /// Load the cache at `path`.
    ///
    /// # Errors
    /// Names what the file is when it is not a current cache.
    fn of(path: &Path) -> Result<Self, String> {
        match shader_cache::load(path) {
            Ok(CacheLoad::Current(records)) => Ok(Self {
                shaders: records.shaders.len(),
                ff_shaders: records
                    .shaders
                    .iter()
                    .filter(|entry| !entry.kind.is_programmable())
                    .count(),
                pipelines: records.pipelines.len(),
                appended: records.needs_compaction,
            }),
            Ok(CacheLoad::Missing) => Err("no cache file".to_owned()),
            Ok(CacheLoad::InvalidatedVersion(header)) => Err(version_note(&header)),
            Ok(CacheLoad::InvalidatedWrongMagic) => Err("not a shader cache".to_owned()),
            Err(error) => Err(format!("unreadable: {error}")),
        }
    }

    const fn records(&self) -> usize {
        self.shaders + self.pipelines
    }
}

/// One child process, as it reported itself.
struct Child {
    line: ChildLine,
    /// The child's layer log.
    log: String,
    /// From the spawn to the child's test body.
    to_test: Duration,
    /// From the spawn to the first `Present` returning.
    to_first_present: Duration,
    /// From the spawn to the first frame read back.
    to_ready: Duration,
}

/// The numbers on a child's [`CHILD_LINE`].
struct ChildLine {
    entered_at: u64,
    /// `Direct3DCreate9`, the window and `CreateDevice`.
    create: Duration,
    /// From `CreateDevice` returning to the first `Present` returning.
    first_present: Duration,
    /// From `CreateDevice` returning to the first frame read back.
    ready: Duration,
    first_present_at: u64,
    ready_at: u64,
    /// Committed, reserved, largest free and peak working set bytes after the readback.
    memory: [u64; 4],
}

impl ChildLine {
    fn parse(line: &str) -> Option<Self> {
        let field = |key: &str| {
            line.split_whitespace()
                .find_map(|word| word.strip_prefix(key)?.strip_prefix('='))
                .and_then(|value| value.parse::<u64>().ok())
        };
        Some(Self {
            entered_at: field("entered_at")?,
            create: Duration::from_nanos(field("create")?),
            first_present: Duration::from_nanos(field("first_present")?),
            ready: Duration::from_nanos(field("ready")?),
            first_present_at: field("first_present_at")?,
            ready_at: field("ready_at")?,
            memory: [
                field("committed")?,
                field("reserved")?,
                field("largest_free")?,
                field("peak_ws")?,
            ],
        })
    }
}

/// What a child's layer log says its prewarm did.
struct Prewarm {
    /// Shaders, pipelines and no-color mappings, in the order of [`PREWARM_COUNTS`].
    counts: [Option<u64>; 3],
    /// The prewarm thread's whole run, as it logs it in milliseconds.
    startup: Option<Duration>,
    /// Pipeline recipes whose build failed.
    failed: u64,
    /// Pipeline recipes left out because a shader they name did not build.
    skipped: u64,
}

impl Prewarm {
    fn of(log: &str) -> Self {
        let shaders = log.lines().find_map(|line| {
            let (before, _) = line.split_once(PREWARM_SHADERS)?;
            let (_, count) = before.rsplit_once("shaders:")?;
            count.trim().parse().ok()
        });
        let closing = log
            .lines()
            .find_map(|line| Some(line.split_once(PREWARM_PIPELINES)?.1));
        let pipelines = closing.and_then(|rest| rest.split_once(" render pipelines, "));
        let no_color =
            pipelines.and_then(|(_, rest)| rest.split_once(" no-color mappings; startup "));
        let startup = no_color
            .and_then(|(_, rest)| rest.split_once("s,"))
            .and_then(|(seconds, _)| seconds.parse::<f64>().ok())
            .map(Duration::from_secs_f64);
        Self {
            counts: [
                shaders,
                pipelines.and_then(|(count, _)| count.parse().ok()),
                no_color.and_then(|(count, _)| count.parse().ok()),
            ],
            startup,
            failed: lines_with(log, PREWARM_FAILED),
            skipped: lines_with(log, PREWARM_SKIPPED),
        }
    }
}

/// One timed step of a measured process, as its report row and metrics name it.
struct Step {
    metric: &'static str,
    label: &'static str,
    pick: fn(&Run) -> Duration,
}

/// One measured process.
struct Run {
    child: Child,
    prewarm: Prewarm,
    /// Records the run added to the settled cache.
    appended: usize,
    /// Whether the run appended anything at all, a duplicate record included.
    grew: bool,
}

/// The measured processes on one cache.
struct Measured {
    settled: Settled,
    runs: Vec<Run>,
    /// The last measured process's layer log, whose load line names the build.
    log: String,
}

/// Run the copy at `exe` once and collect what it printed and logged.
///
/// The copy's layer log is the one file in `mtld3d-logs` beside it; it is
/// read and removed, so the next process's is the only one again.
///
/// # Panics
/// Panics if the child fails or prints no [`CHILD_LINE`].
fn spawn(exe: &Path, dir: &Path) -> Child {
    let mut command = Command::new(exe);
    command
        .args(["--exact", TEST_PATH, "--ignored", "--nocapture"])
        .env("RUST_LOG", CHILD_LOG_FILTER);
    let spawned = SystemTime::now();
    let output = run_child(&mut command, CHILD_ENTRIES).expect("run a cold-start child");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "cold-start child {} failed: {stdout}\n{}",
        exe.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let line = stdout
        .lines()
        .find_map(|line| ChildLine::parse(line.strip_prefix(CHILD_LINE)?))
        .unwrap_or_else(|| {
            panic!(
                "cold-start child {} printed no timings: {stdout}",
                exe.display()
            )
        });
    let logs = dir.join("mtld3d-logs");
    let entries: Vec<PathBuf> = fs::read_dir(&logs)
        .expect("list the child's logs")
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("log"))
        })
        .collect();
    assert_eq!(
        entries.len(),
        1,
        "one layer log per child in {}",
        logs.display()
    );
    let log =
        String::from_utf8_lossy(&fs::read(&entries[0]).expect("read the child's log")).into_owned();
    fs::remove_file(&entries[0]).expect("remove the child's log");
    let since = |at: u64| {
        let spawned = unix_nanos(spawned);
        Duration::from_nanos(at.saturating_sub(spawned))
    };
    Child {
        to_test: since(line.entered_at),
        to_first_present: since(line.first_present_at),
        to_ready: since(line.ready_at),
        line,
        log,
    }
}

/// The corpus caches `make bench BENCH_CORPUS=...` staged under `corpus/`, by corpus name.
///
/// Each is `corpus/<name>/mtld3d_shaders.bin`, the name the Makefile gave
/// it (the directory the cache came from, every character other than an
/// ASCII letter or digit turned into `_`, as the host emitter names a
/// corpus). The name is mapped the same way again here, so a directory
/// staged by hand cannot put a space into a report's name.
fn corpus_files() -> Vec<(String, PathBuf)> {
    let mut files: Vec<(String, PathBuf)> = fs::read_dir(log_dir().join("corpus"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path().join(CORPUS_FILE))
        .filter(|path| path.is_file())
        .filter_map(|path| {
            let dir = path.parent()?.file_name()?.to_string_lossy();
            let name: String = dir
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect();
            Some((name, path))
        })
        .collect();
    files.sort();
    files
}

/// Whether the cache at `path` is of this build's format and schema.
///
/// # Errors
/// Names the versions the file carries when the layer would wipe it.
fn compatible(path: &Path) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|error| format!("unreadable: {error}"))?;
    let header = shader_cache::read_header(&bytes).map_err(|_| "not a shader cache".to_owned())?;
    if header == CacheHeader::CURRENT {
        Ok(())
    } else {
        Err(version_note(&header))
    }
}

fn version_note(header: &CacheHeader) -> String {
    format!(
        "cache format {} schema {}, this build reads format {} schema {}",
        header.format_version,
        header.shader_schema_version,
        CacheHeader::CURRENT.format_version,
        CacheHeader::CURRENT.shader_schema_version,
    )
}

/// Write the report `name` for the measured processes on one cache.
///
/// `kind` says which extra metrics the report carries.
fn report(name: &str, shape: &str, measured: &Measured, kind: &Kind) {
    let runs = &measured.runs;
    let settled = &measured.settled;
    let dir = log_dir();
    fs::create_dir_all(&dir).expect("the report directory can be created");
    let child_log = dir.join(format!("bench-{name}-child.log"));
    fs::write(&child_log, &measured.log).expect("the child's layer log can be kept");
    let log = LayerLog::at(child_log);

    let mut metrics = Metrics::for_entries(name, CHILD_ENTRIES, &TscClock::calibrated());
    let count = |n: usize| Value::Count(u64::try_from(n).expect("a count fits u64"));
    metrics.metric("runs", count(runs.len()), Direction::Higher, Class::Info);
    let mut body = String::from(shape);
    let _ = writeln!(
        body,
        "cache: {shaders} shaders ({ff} fixed-function), {pipelines} pipeline recipes, {kib} KiB, \
         put back before each of {count} measured processes{notes}",
        shaders = settled.counts.shaders,
        ff = settled.counts.ff_shaders,
        pipelines = settled.counts.pipelines,
        kib = settled.bytes >> 10,
        count = runs.len(),
        notes = settled.notes,
    );
    for step in &STEPS {
        let times: Vec<Duration> = runs.iter().map(step.pick).collect();
        let p50 = median(&times);
        let max = times.iter().max().copied().unwrap_or_default();
        let each: Vec<String> = times
            .iter()
            .map(|time| format!("{:.1}", ms(*time)))
            .collect();
        let _ = writeln!(
            body,
            "{label}: p50 {p50:.3} ms  max {max:.3} ms  (each run: {each})",
            label = step.label,
            p50 = ms(p50),
            max = ms(max),
            each = each.join(" "),
        );
        for (row, value, class) in [("p50", p50, Class::Time), ("max", max, Class::Info)] {
            metrics.metric(
                &format!("{}.{row}", step.metric),
                Value::Ms(value),
                Direction::Lower,
                class,
            );
        }
    }

    let startups: Vec<Duration> = runs.iter().filter_map(|run| run.prewarm.startup).collect();
    if startups.len() == runs.len() {
        let p50 = median(&startups);
        let _ = writeln!(body, "prewarm (layer log): p50 {:.3} ms", ms(p50));
        // The log gives the time to the millisecond, too coarse to compare
        // builds by; `ready` and `start_to_ready` carry the prewarm's cost.
        metrics.metric("prewarm.p50", Value::Ms(p50), Direction::Lower, Class::Info);
    } else {
        let _ = writeln!(
            body,
            "prewarm (layer log): logged by {} of {} runs, no time reported",
            startups.len(),
            runs.len()
        );
    }
    for (at, (metric, label)) in PREWARM_COUNTS.into_iter().enumerate() {
        let last = runs.last().and_then(|run| run.prewarm.counts[at]);
        let agree = runs.iter().all(|run| run.prewarm.counts[at] == last);
        let _ = writeln!(
            body,
            "prewarmed {label}: {}{}",
            last.map_or_else(|| "not logged".to_owned(), |value| value.to_string()),
            if agree {
                ""
            } else {
                " (the runs disagree; the last run's)"
            }
        );
        if let Some(value) = last {
            metrics.metric(metric, Value::Count(value), Direction::Higher, Class::Exact);
        }
    }
    if let Some(last) = runs.last() {
        // Why fewer pipelines than recipes were built: a failed build, a
        // recipe naming a shader that failed, or a recipe resolving to a
        // pipeline another recipe had already built this startup.
        let prewarm = &last.prewarm;
        let recipes = u64::try_from(settled.counts.pipelines).expect("a count fits u64");
        let built = prewarm.counts[1].unwrap_or(0);
        let duplicates = recipes
            .saturating_sub(built)
            .saturating_sub(prewarm.failed)
            .saturating_sub(prewarm.skipped);
        let shaders = u64::try_from(settled.counts.shaders).expect("a count fits u64");
        let _ = writeln!(
            body,
            "prewarm of the last run: {built} of {recipes} recipes built, {failed} failed, \
             {skipped} skipped for a shader that failed, {duplicates} resolved to a pipeline \
             already built; {missing} of {shaders} shaders not prewarmed",
            failed = prewarm.failed,
            skipped = prewarm.skipped,
            missing = shaders.saturating_sub(prewarm.counts[0].unwrap_or(0)),
        );
        for (metric, value) in [
            ("prewarm.failed", prewarm.failed),
            ("prewarm.skipped", prewarm.skipped),
            ("prewarm.duplicates", duplicates),
        ] {
            metrics.metric(metric, Value::Count(value), Direction::Lower, Class::Info);
        }
    }
    for (metric, value) in [
        ("cache.shaders", settled.counts.shaders),
        ("cache.ff_shaders", settled.counts.ff_shaders),
        ("cache.pipelines", settled.counts.pipelines),
    ] {
        metrics.metric(metric, count(value), Direction::Higher, Class::Exact);
    }
    metrics.metric(
        "cache.mib",
        Value::Mib(settled.bytes),
        Direction::Lower,
        Class::Info,
    );
    let appended: usize = runs.iter().map(|run| run.appended).sum();
    let grew = runs.iter().filter(|run| run.grew).count();
    let _ = writeln!(
        body,
        "live records (appended to the settled cache, so not prewarmed): {appended}, by {grew} \
         of {} runs",
        runs.len()
    );
    metrics.metric(
        "live.records",
        count(appended),
        Direction::Lower,
        Class::Exact,
    );
    if settled.counts.appended {
        let _ = writeln!(
            body,
            "live runs: not counted, the kept cache is not one bundle, so every run appends"
        );
    } else {
        metrics.metric("live.runs", count(grew), Direction::Lower, Class::Exact);
    }

    let memory: Vec<u64> = (0..MEMORY_ROWS.len())
        .map(|at| {
            let values: Vec<u64> = runs.iter().map(|run| run.child.line.memory[at]).collect();
            median(&values)
        })
        .collect();
    let _ = writeln!(
        body,
        "address space after the first frame read back (p50): committed {} MiB, reserved {} \
         MiB, largest free region {} MiB\npeak working set (under Wine the host process's peak \
         RSS, p50): {} MiB",
        memory[0] >> 20,
        memory[1] >> 20,
        memory[2] >> 20,
        memory[3] >> 20,
    );
    for ((row, direction), bytes) in MEMORY_ROWS.into_iter().zip(memory) {
        metrics.metric(
            &format!("mem.ready.{row}"),
            Value::Mib(bytes),
            direction,
            Class::Bytes,
        );
    }
    match *kind {
        Kind::Synthetic { corpora, skipped } => {
            metrics.metric(
                "corpus.measured",
                Value::Count(corpora),
                Direction::Higher,
                Class::Info,
            );
            metrics.metric(
                "corpus.skipped",
                Value::Count(skipped),
                Direction::Lower,
                Class::Info,
            );
        }
        Kind::Corpus => metrics.metric(
            "corpus.regenerated",
            Value::Count(u64::from(settled.regenerated)),
            Direction::Lower,
            Class::Info,
        ),
    }
    write_report(&metrics, &log, &body);
}

/// Where the `at`-th programmable draw goes, in clip space.
fn place(at: u32) -> (f32, f32) {
    (
        ratio(at % 14, 14).mul_add(1.8, -0.95),
        ratio(at / 14 % 14, 14).mul_add(1.8, -0.95),
    )
}

/// The nearest-rank median of `values`.
///
/// # Panics
/// Panics if `values` is empty.
fn median<T: Ord + Copy>(values: &[T]) -> T {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[nearest_rank(sorted.len(), 50)]
}

/// `time` as `SystemTime` nanoseconds since the Unix epoch, zero before it.
fn unix_nanos(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH).map_or(0, |since| {
        u64::try_from(since.as_nanos()).unwrap_or(u64::MAX)
    })
}

/// How many lines of `log` contain `marker`.
fn lines_with(log: &str, marker: &str) -> u64 {
    let lines = log.lines().filter(|line| line.contains(marker)).count();
    u64::try_from(lines).expect("a line count fits u64")
}
