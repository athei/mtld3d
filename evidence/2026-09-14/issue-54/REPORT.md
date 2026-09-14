# HDR present follow-up measurements (#54)

Retain the current present shader. The hardware-decode variant reduces GPU pass
cost in this offscreen experiment, but changes both near-black decode precision
and unequal-extent filtering. The tested half-precision variants either produce
large numerical errors or buy less than one microsecond per 4K pass. No production
source was changed and no PR is proposed from these measurements.

## Source and route audit

Baseline is `cdf836dbe97d9db46d3ed33a8d7cda3a7c37fc66`.
`unix/unix/src/metal/texture.rs::create_backbuffer` already creates an sRGB twin.
`texture_usage` omits PixelFormatView on Apple-family GPUs, retaining
ShaderRead|RenderTarget. `present.msl` still samples UNORM with a linear sampler,
then applies the piecewise sRGB EOTF. `command.rs::encode_hdr_present` selects
passthrough at peak <= 1 and BT.2446 above it.

`encode_hdr_present_upscaled` tone-maps into a render-resolution RGBA16Float
scratch before MetalFX's HDR upscale, so that first pass is a 1:1 sampling case.
The direct HDR fallback stretch can have unequal extents. SDR present copy and
readback copy consume encoded samples and cannot switch to an sRGB view with
unchanged output. Software cursor shaders share the transforms, but the probe
measured frame stages, not cursor compositing or its actual texture lifecycle.

The issue's lossless-compression premise is outdated on this Apple-family path.
The captured sRGB view reports `compressionType: Lossless`,
`allowGPUOptimizedContents: yes`, storage Private, usage ShaderRead|RenderTarget,
with the base texture as its parent. This is eligibility/metadata evidence; the
probe does not claim a measured compression ratio for a game's backbuffer.
Apple documents the transfer-function-only view exception in its
[pixelFormatView documentation](https://developer.apple.com/documentation/metal/mtltextureusage/pixelformatview).

## Workload and timing results

Apple M5 Max, 40 GPU cores, macOS 27.0 build 26A428. The probe uses exact current
MSL with language 2.4 and Fast math. All shaders render a fullscreen triangle from
a private BGRA8Unorm texture into private RGBA16Float, with DontCare/Store. The
hardware variant uses an sRGB view without adding PixelFormatView usage. Upload,
pipeline compile, readback, and capture are outside normal timing intervals.

Each workload runs 16 warmup passes per variant, then 24 alternating
forward/reverse rounds of eight separately encoded passes per command buffer.
GPUStartTime/GPUEndTime measures the command buffer, divided by eight. The two
complete runs cover 1920x1080 and 3840x2160, gradient and deterministic noise,
and headroom peaks 1, 2, 4 and 8. The parent paused competing builds/tests/probes
for this window. Unrelated user apps remained running; process snapshots are
retained. The initial workload can reflect GPU clock ramp, so paired differences
within a workload are preferable to comparing absolute times across workloads.

Representative 3840x2160, peak 4, in microseconds per pass:

| Input | Variant | First median | Repeat median | Repeat paired change | Repeat paired range |
|---|---|---:|---:|---:|---:|
| Gradient | Current float | 146.219 | 146.187 | reference | reference |
| Gradient | Hardware sRGB | 130.516 | 130.490 | -10.738% | -10.839% to -10.670% |
| Gradient | Half curve | 145.885 | 145.849 | -0.226% | -0.303% to -0.189% |
| Gradient | Half PQ | 153.091 | 153.065 | +4.705% | +4.637% to +4.799% |
| Gradient | Half both | 150.841 | 150.815 | +3.166% | +3.113% to +3.262% |
| Noise | Current float | 149.878 | 150.857 | reference | reference |
| Noise | Hardware sRGB | 141.367 | 142.687 | -5.408% | -5.673% to -5.041% |
| Noise | Half curve | 149.667 | 150.602 | -0.148% | -0.826% to +0.310% |
| Noise | Half PQ | 154.781 | 155.411 | +3.036% | +2.477% to +4.392% |
| Noise | Half both | 153.383 | 154.221 | +2.266% | +1.867% to +2.977% |

At peak 1 the first 4K gradient passthrough median changed 35.969 to 35.503 us
(-1.296% paired); noise changed 122.487 to 122.383 us (-0.072%, with paired
p10/p90 from -0.357% to +0.137%). Hardware decode is not a substantial universal
passthrough saving. Texture input and output traffic remain part of the pass.
These are repeated offscreen pass costs, not FPS or live frame pacing.

## Numerical and image evidence

Raw RGBA16Float readbacks cover 512x512 source images, output extents 512, 768 and
256, and peaks 1/2/4/8. Inputs contain deterministic dense RGB samples, all 256
encoded gray levels, alternating black/white texels, and all 256 alpha values.
Scaled gradients exercise fractional samples around the BT.2446 thresholds
(approximately gray byte 159.063 and 252.496 in the source float formulation).
`grayscale-details.json` retains the neighboring samples and monotonicity checks;
`numerical-comparison.png` and `.svg` plot measured grayscale output and the
checker filtering difference.
There are 117 candidate/reference comparisons and nine SDR-copy comparisons.
Every output was finite, alpha was bit-identical in all 117 HDR comparisons,
and all nine SDR copies were byte-identical when retaining the UNORM binding.

The existing output is RGBA16Float, so one output-half ULP is a useful reference
for rounding-sized differences. It is not a project-approved perceptual error
budget. Neither a loose relative threshold nor a favorable average establishes
acceptable image quality. Relative error can be large in a small color channel,
so retained results include absolute error and the actual worst pixels as well.

At peak 4, 1:1 dense RGB input, errors against current float output are:

| Variant | Max absolute RGB error | RGB RMSE | p99 absolute error |
|---|---:|---:|---:|
| Hardware sRGB | 0.0078125 | 0.0003770 | 0.0019531 |
| Half curve | 0.0683594 | 0.0063703 | 0.0253906 |
| Half PQ | 13.5498047 | 0.6894199 | 2.6139551 |
| Half both | 13.3564453 | 0.6819680 | 2.6113281 |

Values are normalized linear RGB, where 1.0 is the shader's paper-white
reference. At peak 8 the respective max errors are 0.0195313, 0.1328125,
24.984375, and 24.984375. For a peak-4 gray gradient, half-PQ changes one channel
from 2.7734375 to 3.869140625. A saturated RGB sample changes from
(1.1923828, 3.3144531, 2.7265625) to (14.7421875, 0.1722412, 5.3945313).
The half-PQ grayscale output decreases at 36 of the 255 adjacent gray-byte
transitions at peak 4; baseline, hardware and half-curve have no decreases there.
That cannot be justified as output rounding.

The curve candidate changes only the `yp_*` curve arithmetic to half; input
luminance, absolute-nit conversion, matrices, intensity/chroma ratio, and final
RGB multiplication remain float. Its peak-4 gray max error is 0.0488281, with
0.0091475 RMSE, for a timing difference around 0.3 us. The PQ candidate changes
PQ encode/decode arithmetic and constants to half, leaving float inputs/outputs
and matrices. Its denominator floor uses the smallest normal half value instead
of letting the float 1e-20 floor become zero. Half still poorly resolves the
subtraction/division and powers in the PQ round trip. Compiler statistics also
show that the hoped-for factor-of-two throughput does not follow from changing
these declarations. These findings reject these precise candidates; they do not
prove every possible mixed-precision formulation is unhelpful.

Hardware decode has a distinct resampling behavior: a 2:1 downsample of a
black/white checker gives 0.2139893 with current decode-after-filter versus 0.5
with hardware decode-before-filter, at peak 1. This is a different color-filtering
contract, not a numerical precision issue. Even 1:1 hardware sampling changes
low gray values measurably. At peak 4, encoded gray 1 produces 0.000449896 in
the current shader and 0.000360727 with the sRGB view; encoded gray 2 produces
0.000911236 versus 0.000729561. Absolute errors are small, but the approximately
20% near-black shift makes a claim of numerical equivalence inappropriate.

A narrow hardware-decode change would need explicit geometry/format routing,
retention of the encoded SDR and stretch behavior or approval of its new
filtering contract, an agreed near-black accuracy trade, and matched cursor and
MetalFX input validation. These measurements do not license adding an escape
hatch or silently changing all routes.

## GPU trace and shader profile

`hdr-five-variants.gputrace` captures one upload buffer and one command buffer
with five labeled 4K peak-4 render passes. The trace is about 192 MiB and carries
an embedded gpudebug profile collected with `--gpu-state high --exec serial`.
The raw profile log, per-encoder costs, compiler details, resource bindings and
extraction logs are retained. The corrected extraction produced four 4K output
PNGs plus a baseline fragment-cost heatmap and its Top100 JSON. The PNGs were
inspected for image content; they are normalized debugger previews, not calibrated
HDR display evidence. The numerical readbacks establish the color differences. This profile is a controlled replay of the capture,
not the uncaptured repeated-pass timing workload.

| Variant | Instructions | ALU | FP16 | FP32 | Temp registers | Profile fragment interval |
|---|---:|---:|---:|---:|---:|---:|
| Current float | 219 | 186 | 0 | 160 | 16 | 0.1622 ms |
| Hardware sRGB | 194 | 165 | 0 | 142 | 16 | 0.1665 ms |
| Half curve | 218 | 189 | 9 | 154 | 16 | 0.1704 ms |
| Half PQ | 220 | 189 | 21 | 142 | 20 | 0.1731 ms |
| Half both | 219 | 192 | 30 | 136 | 16 | 0.1724 ms |

Hardware decode removes 25 instructions and 21 ALU instructions in this compiler
output. The profile ranks its shader at 17.66% cost versus 18.26% for baseline,
but those percentages are shares of this multi-pass capture. The hardware
fragment interval is longer than baseline in this replay, despite the lower
cost share. Neither the rank nor those intervals demonstrate the uncaptured
speedup; the paired native command-buffer measurements above are its evidence.
Profile/capture overhead, replay scheduling, and the different one-pass workload
prevent substituting one metric for the other.

## Verification, limits, and Rules check

The native probe compiled successfully; all submitted command buffers completed.
All numerical outputs and timing records are retained, with a complete repeat.
Capture succeeded and its embedded GPU profile was inspected. Only this worker's
gpudebug session was terminated. No game, shared Wine installation, display-mode
change, frame-pacing or presentation split was involved. No worker processes
remain. No production diff or commit was created.

This does not measure MetalFX itself, the compositor, drawable acquisition,
concurrent game rendering, MSAA resolve costs, cursor compositing, Intel/AMD,
other Apple GPU generations, sustained thermal behavior, or live-game FPS.
Input is synthetic, and half precision is judged only for the retained variants.
The sRGB metadata is not an end-to-end measurement of framebuffer compression.

No statics, configuration keys, wire fields, dependencies, derives, lint
suppressions, or production duplication were introduced. Only ignored artifacts
exist. `make check`, `make test`, and conformance were not run because the
production tree is unchanged; those gates remain required for any later patch.

## Reproduction and artifact identity

See `REPRODUCE.md` for exact native build, timing, capture and profile commands.
`source-sha256.txt` contains all five MSL hashes and `SHA256SUMS` covers the evidence.
Baseline MSL SHA256 is
`e7868218bf20af43694381b9d57fcfe8c84f49fe323078364b5d7f2700dcbd7e`.
The numerical reference is the exact existing float shader, not an independently
implemented BT.2446 standard oracle. The checker difference also agrees with the
analytic decode(0.5) versus mean(decode(0), decode(1)) distinction.
