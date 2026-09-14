# Reproducing the offscreen measurements

Run on macOS with a Metal GPU and the Swift command-line tools. `Probe.swift`
resolves the five `.metal` files beside its source location, so keep the sources
together. The generator deliberately names the original owned worktree/evidence
paths; edit those two path assignments if reproducing elsewhere. Preserve a new
output directory when retaining a second investigation.

The probe uses no Wine, mtld3d installation or game process. Coordinate an idle
CPU/GPU window before timing or replay, and leave unrelated user apps alone.

Build and run, with the evidence directory as the working directory:

```sh
python3 generate_variants.py
xcrun swiftc -O Probe.swift -o probe
./probe correctness > correctness.log 2>&1
./probe timing > timing.log 2>&1
./probe timing > timing-repeat.log 2>&1
```

The original runs had no MTL, METAL or DYLD environment overrides. Correctness
writes explicit `pixels-<pattern>-<extent>-p<peak>-<variant>.bin` files as packed
little-endian RGBA16Float. SDR files are packed BGRA8Unorm. `analyze.py` requires
NumPy and produces full numerical and paired-timing summaries; the original run
used the bundled Python at
`/Users/alex/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`.
`plot.py` additionally requires Matplotlib and plots the measured values without
re-rendering or sampling the GPU. Its dependency target and font cache are in
`.codex/scratchpads/issue-54`.

For capture, launch in one terminal/process:

```sh
MTL_CAPTURE_ENABLED=1 MTLCAPTURE_WAIT_FOR_SIGNAL=1 ./probe capture > capture-probe.log 2>&1
```

It waits at device creation. Read the actual PID from `gpucapture list`, then:

```sh
gpucapture start --pid <PID> --until-exit --output <ABSOLUTE_PATH>/hdr-five-variants.gputrace
```

This captures initialization/upload and the five-pass command buffer, and exits.
Use absolute output paths and create the extraction directory before fetching:

```sh
mkdir -p gpudebug
gpudebug -q -t <TRACE> -o <ABSOLUTE_OUTPUT_DIRECTORY> \
  -c 'go commands' \
  -c 'profile run --gpu-state high --exec serial --embed'
```

Read the actual returned session ID, then use `gpudebug -s <ID> -c ...` to inspect
`/performance/encoders`, `/performance/shaders`, and the individual `info fragN`
compiler counts. In this trace baseline is fragment `frag3`/pipeline `rps0`,
hardware is `frag4`/`rps3`, half curve is `frag2`/`rps6`, half PQ is `frag1`/`rps9`,
and half both is `frag0`/`rps12`. Indices are capture-specific: check pipeline
labels before interpreting them in a new capture.

Frame encoders are `/commands/cb1/re0` through `re4` in the baseline, hardware,
half-curve, half-PQ, half-both order. Navigate to `draw0`, inspect `pipeline`,
and fetch `color0 --out <ABSOLUTE_PNG_PATH>`. The hardware source is
`/commands/cb1/re1/draw0/fragment/tex[0]`. Its inspected metadata establishes the
sRGB view format, parent, usage and lossless-compression setting.

The baseline heatmap is under
`/performance/commands/draw3/heatmap/fragment`; `fetch cost` writes its PNG and
Top100 JSON. End the specific owned session with `gpudebug --terminate <ID>`.
Never terminate all sessions on a shared machine.

The first extraction attempt returned a zero process status despite failed image
writes to a missing explicit subdirectory. The original failed log is retained;
`gpudebug-extract-retry.log` records successful fetches after directory creation.
Check output files and tool diagnostics, not only the command's exit status.

`source-sha256.txt` identifies the exact five MSL inputs; `SHA256SUMS` covers the
retained source, outputs, reports, trace contents and images. Baseline source SHA256
is `e7868218bf20af43694381b9d57fcfe8c84f49fe323078364b5d7f2700dcbd7e`.
