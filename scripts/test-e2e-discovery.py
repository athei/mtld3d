#!/usr/bin/env python3
"""Check executable discovery and benchmark launch settings at their Makefile consumers."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


MAKEFILE = Path(__file__).resolve().parent.parent / "Makefile"
PE_TARGETS = {
    "i686": "i686-pc-windows-msvc",
    "x86_64": "x86_64-pc-windows-msvc",
    "aarch64": "aarch64-pc-windows-msvc",
}
# The legs of `make test` in each mode, each with the PE target of the test
# binaries it runs, and the unix `.so` install leaf the mode's legs take: the
# arm64 mode's 64-bit leg runs the x86_64 binaries against the ARM64X build,
# the x86_64 mode's against the x86_64 build.
MODES = {
    "arm64": ({"i686": PE_TARGETS["i686"], "arm64x": PE_TARGETS["x86_64"]}, "arm64"),
    "x86_64": ({"i686": PE_TARGETS["i686"], "x86_64": PE_TARGETS["x86_64"]}, "x64"),
}
# The variables of the two-Wine layout, which the Makefile refuses; a shell
# that still exports one would stop every run here.
OLD_VARIABLES = ("WINE_X86_64", "WINE_ARM64", "WINE_SDK_ARM64X", "EC", "ARM64")


class E2eDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mtld3d-e2e-discovery-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        # The mode is read from the Wine's `lib/wine`: a tree the Makefile
        # cannot read is taken for the arm64 mode, one holding `x86_64-unix`
        # for the x86_64 mode, the one CI's Intel image runs.
        self.wine_sdks = {"arm64": self.root / "wine-arm64", "x86_64": self.root / "wine-x86_64"}
        self.wine_sdks["arm64"].mkdir()
        (self.wine_sdks["x86_64"] / "lib/wine/x86_64-unix").mkdir(parents=True)
        self.test_exes = {
            arch: self.root / f"{target}.exe" for arch, target in PE_TARGETS.items()
        }
        for test_exe in self.test_exes.values():
            test_exe.touch()
        self.fake_build = self.root / "fake-build"
        self.fake_build.write_text(
            "#!/bin/sh\n"
            "target=$1\n"
            "previous=\n"
            "for argument in \"$@\"; do\n"
            "    if [ \"$previous\" = --target ]; then target=$argument; break; fi\n"
            "    previous=$argument\n"
            "done\n"
            f"printf '{{\"reason\":\"compiler-artifact\",\"executable\":\"%s/%s.exe\"}}\\n' "
            f"{shlex.quote(str(self.root))} \"$target\"\n"
            "[ \"$target\" != \"$FAIL_TARGET\" ] || exit 23\n"
        )
        self.fake_build.chmod(0o755)
        (self.root / "cargo").symlink_to(self.fake_build)
        self.environment = os.environ.copy()
        for name in ("MAKEFLAGS", "MFLAGS", "GNUMAKEFLAGS", "MAKEOVERRIDES", "MAKELEVEL",
                     "ISOLATED", "STAGE", "WINE_INSTALL_DIR", *OLD_VARIABLES):
            self.environment.pop(name, None)
        self.environment["PATH"] = f"{self.root}{os.pathsep}{self.environment['PATH']}"

    def run_make(self, mode, *arguments, fail_target):
        environment = self.environment.copy()
        environment["FAIL_TARGET"] = fail_target
        result = subprocess.run(
            [
                "make",
                "--no-print-directory",
                "-f",
                str(MAKEFILE),
                f"WINE_SDK={self.wine_sdks[mode]}",
                f"ISOLATED_ROOT={self.root / '.wine-isolated'}",
                "ISOLATED_REGISTRY=",
                "BUILD_ID=e2e-discovery-test",
                f"E2E_EXES_BUILD={self.fake_build} $(1)",
                *arguments,
            ],
            cwd=MAKEFILE.parent,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        return result

    def assert_discovery_failed(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("Error 23", result.stdout)

    def test_partial_output_does_not_reach_either_runner(self):
        for mode, (legs, unix) in MODES.items():
            for leg, target in legs.items():
                with self.subTest(mode=mode, leg=leg):
                    runner_marker = self.root / f"runner-{mode}-{leg}-ran"
                    runner = self.root / f"runner-{mode}-{leg}"
                    runner.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(runner_marker))}\n")
                    runner.chmod(0o755)
                    result = self.run_make(
                        mode,
                        "-o",
                        f"install-windows-{leg}",
                        "-o",
                        f"install-unix-{unix}",
                        "MAKE=true",
                        f"E2E_RUNNER={runner}",
                        "E2E_RUNNER_DIR=.",
                        f"test-e2e-{leg}",
                        fail_target=target,
                    )
                    self.assertFalse(runner_marker.exists(), result.stdout)
                    self.assert_discovery_failed(result)

    def test_benchmark_logging_does_not_inherit_shell_filters(self):
        suite = self.root / "e2e-fixture.exe"
        suite.touch()
        log_dir = self.root / "bench-logs"
        marker = self.root / "bench-environment"
        runner = self.root / "bench-runner"
        runner.write_text(
            "#!/bin/sh\n"
            f"printf '%s|%s\\n' \"$RUST_LOG\" \"$__CX_UNIX_RUST_LOG\" > {shlex.quote(str(marker))}\n"
            f"touch {shlex.quote(str(log_dir / 'bench-fixture.txt'))}\n"
        )
        runner.chmod(0o755)
        for mode, (_, unix) in MODES.items():
            for pe, unix_filter in [("warn", "off"), ("off", "mtld3d::perf=off")]:
                with self.subTest(mode=mode, pe=pe, unix=unix_filter):
                    self.environment["RUST_LOG"] = pe
                    self.environment["__CX_UNIX_RUST_LOG"] = unix_filter
                    marker.unlink(missing_ok=True)
                    result = self.run_make(
                        mode,
                        "-o",
                        "install-windows-i686",
                        "-o",
                        f"install-unix-{unix}",
                        "MAKE=true",
                        f"E2E_RUNNER={runner}",
                        "E2E_RUNNER_DIR=.",
                        f"BENCH_EXES={suite}",
                        f"LOG_DIR={log_dir}",
                        "bench",
                        fail_target="",
                    )
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(marker.read_text(), "info|info\n")

    def test_partial_output_is_not_staged(self):
        outputs = self.root / "outputs"
        out_i386 = outputs / "i386"
        out_x64 = outputs / "x86_64"
        out_arm64x = outputs / "arm64x"
        out_unix_x64 = outputs / "unix-x64"
        out_unix_arm64 = outputs / "unix-arm64"
        for directory in (out_i386, out_x64, out_arm64x, out_unix_x64, out_unix_arm64):
            directory.mkdir(parents=True)
        for directory in (out_i386, out_x64, out_arm64x):
            for name in ("mtld3d.dll", "mtld3d.pdb", "mtld3d.fake.dll", "d3d9.dll", "d3d9.pdb"):
                (directory / name).touch()
        for directory in (out_unix_x64, out_unix_arm64):
            (directory / "mtld3d.so").touch()
            (directory / "mtld3d.so.dSYM").mkdir()
        for mode in MODES:
            # The test binaries the stage takes in this mode, in its order: the
            # aarch64 ones, which the native ARM64 leg runs, in the arm64 mode alone.
            order = ["i686", "x86_64", *(["aarch64"] if mode == "arm64" else [])]
            for failed_arch, failed_target in PE_TARGETS.items():
                if failed_arch not in order:
                    continue
                with self.subTest(mode=mode, failed_arch=failed_arch):
                    stage_dir = self.root / f"stage-{mode}-{failed_arch}"
                    result = self.run_make(
                        mode,
                        "-o",
                        "all",
                        f"OUT_i386={out_i386}",
                        f"OUT_x64={out_x64}",
                        f"OUT_arm64x={out_arm64x}",
                        f"OUT_unix_x64={out_unix_x64}",
                        f"OUT_unix_arm64={out_unix_arm64}",
                        f"STAGE_DIR={stage_dir}",
                        f"STAGE_OUT={self.root / f'stage-{mode}-{failed_arch}.tar'}",
                        "stage",
                        fail_target=failed_target,
                    )
                    for arch, test_exe in self.test_exes.items():
                        staged = stage_dir / f"tests/{arch}" / test_exe.name
                        # Only the binaries of the arches staged before the one that failed.
                        before = arch in order and order.index(arch) < order.index(failed_arch)
                        self.assertEqual(staged.exists(), before, result.stdout)
                    # The ARM64X pair is staged in the arm64 mode alone.
                    self.assertEqual((stage_dir / "aarch64-windows/d3d9.dll").exists(),
                                     mode == "arm64", result.stdout)
                    self.assertEqual((stage_dir / "tests/aarch64").is_dir(), mode == "arm64", result.stdout)
                    self.assertTrue((stage_dir / "x86_64-windows/d3d9.dll").exists())
                    self.assert_discovery_failed(result)


if __name__ == "__main__":
    unittest.main()
