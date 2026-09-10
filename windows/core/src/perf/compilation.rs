//! Cold shader and pipeline work attributed to its owning encoder submission.
//!
//! Native durations are nanoseconds, never raw ticks from another runtime.
//! Frame totals and individual slow operations are kept distinct.

#[cfg(perf_tracking)]
use std::fmt::Write as _;

use mtld3d_shared::perf::perf_enabled;

#[cfg(perf_tracking)]
const SLOW_NS: u64 = 2_000_000;
#[cfg(perf_tracking)]
const SLOW_LIMIT: usize = 5;

#[cfg(test)]
mod tests;

/// One measured operation; parent totals include their nested children.
#[repr(usize)]
pub enum Kind {
    ShaderVs,
    ShaderPs,
    EmitVs,
    EmitPs,
    ShaderPreparation,
    Library,
    Function,
    CacheWrite,
    Pipeline,
    Sibling,
    PipelinePreparation,
    PipelineBuild,
    Depth,
    ResolveOther,
    PipelineOther,
}

#[cfg(perf_tracking)]
impl Kind {
    const COUNT: usize = Self::PipelineOther as usize + 1;
    const LABELS: [&'static str; Self::COUNT] = [
        "VS miss total",
        "PS miss total",
        "  emit VS",
        "  emit PS",
        "  shader setup",
        "  Metal library",
        "  function lookup",
        "  cache persist",
        "PSO primary total",
        "PSO sibling total",
        "  PSO setup",
        "  Metal PSO build",
        "depth state build",
        "resolve remainder",
        "pipeline remainder",
    ];
}

/// Per-device compilation accounting; storage vanishes without PERF.
#[derive(Default)]
pub struct CompilationPerf {
    #[cfg(perf_tracking)]
    frame: [Metric; Kind::COUNT],
    #[cfg(perf_tracking)]
    window: [Metric; Kind::COUNT],
    #[cfg(perf_tracking)]
    slow: Vec<SlowOperation>,
    #[cfg(perf_tracking)]
    frame_serial: u64,
}

impl CompilationPerf {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            #[cfg(perf_tracking)]
            frame: [const { Metric::new() }; Kind::COUNT],
            #[cfg(perf_tracking)]
            window: [const { Metric::new() }; Kind::COUNT],
            #[cfg(perf_tracking)]
            slow: Vec::new(),
            #[cfg(perf_tracking)]
            frame_serial: 0,
        }
    }

    /// Record elapsed work, evaluating identity only for a retained slow event.
    pub fn record(
        &mut self,
        kind: Kind,
        ns: u64,
        success: bool,
        seq: u64,
        identity: impl FnOnce() -> Identity,
    ) {
        if !perf_enabled() {
            return;
        }
        self.record_enabled(kind, ns, success, seq, identity);
    }

    #[cfg(not(perf_tracking))]
    fn record_enabled(
        &mut self,
        _kind: Kind,
        _ns: u64,
        _success: bool,
        _seq: u64,
        _identity: impl FnOnce() -> Identity,
    ) {
        *self = Self::new();
    }

    #[cfg(perf_tracking)]
    fn record_enabled(
        &mut self,
        kind: Kind,
        ns: u64,
        success: bool,
        seq: u64,
        identity: impl FnOnce() -> Identity,
    ) {
        let retain = !matches!(
            kind,
            Kind::ShaderVs | Kind::ShaderPs | Kind::Pipeline | Kind::Sibling
        );
        let index = kind as usize;
        let metric = &mut self.frame[index];
        metric.ns = metric.ns.saturating_add(ns);
        metric.calls = metric.calls.saturating_add(1);
        metric.failures = metric.failures.saturating_add(u64::from(!success));
        // Parent totals remain in the table; retain leaf operations only.
        if ns < SLOW_NS || !retain {
            return;
        }
        let position = self.slow.partition_point(|event| event.ns >= ns);
        if position >= SLOW_LIMIT {
            return;
        }
        if self.slow.len() == SLOW_LIMIT {
            self.slow.pop();
        }
        self.slow.insert(
            position,
            SlowOperation {
                kind: index,
                ns,
                success,
                seq,
                identity: identity(),
                serial: self.frame_serial,
                encoder_ns: None,
            },
        );
    }

    /// Fold native shader phases without treating successful preparation as a build result.
    pub fn shader_parts(
        &mut self,
        timings: &mtld3d_shared::perf::ShaderTimings,
        success: bool,
        seq: u64,
        identity: impl Fn() -> Identity,
    ) {
        if !perf_enabled() {
            return;
        }
        self.shader_parts_enabled(timings, success, seq, identity);
    }

    fn shader_parts_enabled(
        &mut self,
        timings: &mtld3d_shared::perf::ShaderTimings,
        success: bool,
        seq: u64,
        identity: impl Fn() -> Identity,
    ) {
        if timings.preparation_ns == 0 {
            return;
        }
        self.record_enabled(
            Kind::ShaderPreparation,
            timings.preparation_ns,
            success || timings.library_ns != 0,
            seq,
            &identity,
        );
        if timings.library_ns != 0 {
            self.record_enabled(
                Kind::Library,
                timings.library_ns,
                success || timings.function_ns != 0,
                seq,
                &identity,
            );
        }
        if timings.function_ns != 0 {
            self.record_enabled(Kind::Function, timings.function_ns, success, seq, identity);
        }
    }

    /// Close a submission, calculating residuals before taking window maxima.
    #[cfg(perf_tracking)]
    pub fn finish_frame(&mut self, resolve_ns: u64, pipeline_ns: u64, encoder_ns: u64) {
        let shaders = self.frame[Kind::ShaderVs as usize]
            .ns
            .saturating_add(self.frame[Kind::ShaderPs as usize].ns);
        let pipelines = self.frame[Kind::Pipeline as usize]
            .ns
            .saturating_add(self.frame[Kind::Sibling as usize].ns)
            .saturating_add(self.frame[Kind::Depth as usize].ns);
        self.frame[Kind::ResolveOther as usize].ns = resolve_ns.saturating_sub(shaders);
        self.frame[Kind::PipelineOther as usize].ns = pipeline_ns.saturating_sub(pipelines);
        for (window, frame) in self.window.iter_mut().zip(&mut self.frame) {
            window.ns = window.ns.saturating_add(frame.ns);
            window.calls = window.calls.saturating_add(frame.calls);
            window.failures = window.failures.saturating_add(frame.failures);
            window.peak_ns = window.peak_ns.max(frame.ns);
            *frame = Metric::new();
        }
        for event in &mut self.slow {
            if event.serial == self.frame_serial {
                event.encoder_ns = Some(encoder_ns);
            }
        }
        self.frame_serial = self.frame_serial.wrapping_add(1);
    }

    /// Render once with the existing PERF summary, then clear the window.
    #[cfg(perf_tracking)]
    pub fn append_window(&mut self, output: &mut String, frames: u32) {
        if self.window.iter().all(|metric| metric.calls == 0) {
            self.window = [const { Metric::new() }; Kind::COUNT];
            self.slow.clear();
            return;
        }
        let _ = writeln!(
            output,
            "\nCompilation (included in resolve/pipeline; nested rows are not additive)"
        );
        self.write_metrics(output, frames.max(1));
        self.write_slow(output);
        self.window = [const { Metric::new() }; Kind::COUNT];
        self.slow.clear();
    }

    /// Emit startup work separately; no gameplay frame denominator applies.
    #[cfg(perf_tracking)]
    pub fn log_startup(&mut self, device: u64) {
        if !perf_enabled() {
            return;
        }
        self.log_startup_enabled(device);
    }

    #[cfg(not(perf_tracking))]
    pub const fn log_startup(&mut self, _device: u64) {
        *self = Self::new();
    }

    #[cfg(perf_tracking)]
    fn log_startup_enabled(&mut self, device: u64) {
        if self.frame.iter().all(|metric| metric.calls == 0) {
            return;
        }
        self.window = std::mem::take(&mut self.frame);
        let mut output = format!(
            "shader prewarm PERF device={device:#x} (startup totals; outside gameplay windows)\n"
        );
        for (label, metric) in Kind::LABELS.iter().zip(&self.window) {
            if metric.calls != 0 {
                let _ = writeln!(
                    output,
                    "  {label:<20} {:>9.3} ms total  calls={:<5} failed={}",
                    ms(metric.ns),
                    metric.calls,
                    metric.failures
                );
            }
        }
        self.write_slow(&mut output);
        log::info!(target: super::LOG_TARGET, "{output}");
        self.window = [const { Metric::new() }; Kind::COUNT];
        self.slow.clear();
    }

    #[cfg(perf_tracking)]
    fn write_metrics(&self, output: &mut String, frames: u32) {
        for (label, metric) in Kind::LABELS.iter().zip(&self.window) {
            let _ = writeln!(
                output,
                "  {label:<20} {:>7.3} ms/frame  peak/frame {:>7.3} ms  total {:>9.3} ms  calls={:<5} failed={}",
                ms(metric.ns) / f64::from(frames),
                ms(metric.peak_ns),
                ms(metric.ns),
                metric.calls,
                metric.failures
            );
        }
    }

    #[cfg(perf_tracking)]
    fn write_slow(&self, output: &mut String) {
        for event in &self.slow {
            let _ = write!(
                output,
                "  slow {}: {:.3} ms/call seq={} success={} {}",
                Kind::LABELS[event.kind].trim(),
                ms(event.ns),
                event.seq,
                event.success,
                event.identity
            );
            if let Some(ns) = event.encoder_ns {
                let _ = write!(output, " encoder_ops_same_submission={:.3}ms", ms(ns));
            }
            output.push('\n');
        }
    }
}

#[cfg(perf_tracking)]
#[derive(Default)]
struct Metric {
    ns: u64,
    calls: u64,
    failures: u64,
    peak_ns: u64,
}

#[cfg(perf_tracking)]
impl Metric {
    const fn new() -> Self {
        Self {
            ns: 0,
            calls: 0,
            failures: 0,
            peak_ns: 0,
        }
    }
}

#[cfg(perf_tracking)]
struct SlowOperation {
    kind: usize,
    ns: u64,
    success: bool,
    seq: u64,
    identity: Identity,
    serial: u64,
    encoder_ns: Option<u64>,
}

#[cfg(perf_tracking)]
fn ms(ns: u64) -> f64 {
    mtld3d_shared::tsc::u64_to_f64_exact(ns) / 1_000_000.0
}

/// Convert only PE-owned counters using the PE runtime's calibrated frequency.
#[cfg(perf_tracking)]
#[must_use]
pub fn cycles_to_ns(cycles: u64) -> u64 {
    u64::try_from(
        u128::from(cycles) * 1_000_000_000 / u128::from(mtld3d_shared::tsc::tsc_hz().max(1)),
    )
    .unwrap_or(u64::MAX)
}

/// Owned metadata for a retained slow operation, formatted only at summary time.
pub enum Identity {
    Shader {
        device: u64,
        stage: &'static str,
        shader: super::PairShaderId,
    },
    Prewarm {
        device: u64,
        kind: crate::shader_cache::CachedKind,
        key: u64,
    },
    Pipeline {
        device: u64,
        vs: super::PairShaderId,
        ps: super::PairShaderId,
        snapshot: Box<crate::pipeline_state::PipelineSnapshot>,
        sibling: bool,
    },
    Depth {
        device: u64,
        key: u64,
    },
}

#[cfg(perf_tracking)]
impl std::fmt::Display for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shader {
                device,
                stage,
                shader,
            } => write!(
                f,
                "device={device:#x} stage={stage} shader={}",
                shader.tag()
            ),
            Self::Prewarm { device, kind, key } => write!(
                f,
                "device={device:#x} shader={} disk={key:#x}",
                kind.entry_name(*key)
            ),
            Self::Pipeline {
                device,
                vs,
                ps,
                snapshot,
                sibling,
            } => {
                write!(
                    f,
                    "device={device:#x} vs={} ps={} vdecl={:#x} layout=[",
                    vs.tag(),
                    ps.tag(),
                    snapshot.vdecl_hash
                )?;
                for (stream, layout) in snapshot.stream_layouts.iter().enumerate() {
                    if *layout != crate::pipeline_state::StreamLayout::UNUSED {
                        write!(f, "{stream}:{layout:?};")?;
                    }
                }
                write!(
                    f,
                    "] color={:?} attach={:?} samples={} rs={:?} extra={:?} ps_outputs={:#x} sibling={sibling}",
                    snapshot.color_format,
                    snapshot.attach,
                    snapshot.sample_count,
                    snapshot.rs,
                    snapshot.extra,
                    snapshot.ps_color_out_mask
                )
            }
            Self::Depth { device, key } => write!(f, "device={device:#x} depth_key={key:#x}"),
        }
    }
}
