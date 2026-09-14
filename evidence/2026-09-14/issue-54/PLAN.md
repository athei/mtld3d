# Issue 54 bounded experiment

Source baseline: cdf836d (the worktree's initial main revision). Native probe on the
local Apple GPU, no game, Wine launch, installation, presentation lifecycle change,
or #626 work. The retained MSL baseline is copied byte for byte from present.msl.

Compare baseline, hardware sRGB sampling, half BT.2446 inverse curve, half PQ
encode/decode, and both half changes. Use MSL 2.4 and Fast math as production does.
Half candidates keep final absolute-nits scale, ICtCp matrices, chroma ratio and
RGB multiply in float. PQ uses a representable denominator floor to avoid the
original float floor becoming zero. Record source hashes and exact modifications.

Use native Metal private textures with ShaderRead|RenderTarget (usage 5), no
PixelFormatView. Upload and readback outside timings. Fullscreen triangle writes
RGBA16Float with DontCare/Store, as the HDR pass does. Hardware decode uses the
sRGB view; SDR copy continues using the UNORM base. Offscreen destination is a
proxy for the present shader's render target, not the compositor or frame time.

Correctness covers 512-square deterministic RGB noise with a grayscale prefix,
all 256 encoded grayscale values, alternating black/white edges, alpha 0..255,
and 512/768/256-square output extents. Peaks 1, 2, 4, and 8 cover passthrough and
BT.2446. The grayscale gradients span both BT.2446 curve-segment boundaries;
scaled gradients add fractional filtered input values around them. Preserve raw
RGBA16Float output and inspect color, alpha, finiteness, RMSE, max absolute and
relative error and output-half ULP distances. A change that aims to be numerically
equivalent should fit the destination's rounding error (one output-half ULP);
larger discrepancies need explanation and cannot be declared safe from a loose
chosen threshold. No currently documented project error budget licenses visible
color differences merely because a timing improves. Hardware-before-filter and
shader-after-filter are explicitly different for unequal extents.

Timings alternate forward/reverse variant order over 24 rounds after 16 warmup
passes per variant, eight separately encoded fullscreen passes per measured
command buffer. Record GPUStartTime/GPUEndTime per-pass averages at 1920x1080 and
3840x2160, for both gradients and noise and peaks 1/2/4/8. Command-buffer duration
includes pass overhead and memory traffic, and does not isolate ALU alone. Native
API timing runs with capture/debug/HUD disabled during an exclusive session-wide
measurement window. Leave unrelated user apps running and record their processes.

Capture a five-pass 4K peak-4 command buffer with gpucapture, launched with
MTL_CAPTURE_ENABLED and MTLCAPTURE_WAIT_FOR_SIGNAL. Browse/profile with noninteractive
gpudebug, save shader/encoder rankings, output images, profile and heatmap when
available. Captured/replayed profile costs are separate from normal timings.
Terminate only this investigation's debugger sessions.

## Rules check

The experiment introduces only ignored evidence and scratch artifacts. No source
static, configuration key, wire field, dependency, derive, lint suppression,
concurrency mechanism or duplicated production path is introduced. It retains a
snapshot and explicit experimental shader variants solely to measure the proposals.
Current CONTRIBUTING/CONVENTIONS/ARCHITECTURE requirements apply to any subsequent
source patch; none is justified before measurements. Such a patch would need full
fmt/check/isolated test/conformance. An evidence-only report requires no conformance
run because it changes no render, shader, state or shared-crate source.
