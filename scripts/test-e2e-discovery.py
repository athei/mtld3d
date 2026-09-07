#!/usr/bin/env python3
"""Check that failed e2e executable discovery stops every consumer."""

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
}


class E2eDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mtld3d-e2e-discovery-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.wine_sdk = self.root / "wine-sdk"
        self.wine_sdk.mkdir()
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
        for name in ("MAKEFLAGS", "MFLAGS", "GNUMAKEFLAGS", "MAKEOVERRIDES", "MAKELEVEL"):
            self.environment.pop(name, None)
        self.environment["PATH"] = f"{self.root}{os.pathsep}{self.environment['PATH']}"

    def run_make(self, *arguments, fail_target):
        environment = self.environment.copy()
        environment["FAIL_TARGET"] = fail_target
        result = subprocess.run(
            [
                "make",
                "--no-print-directory",
                "-f",
                str(MAKEFILE),
                f"WINE_SDK={self.wine_sdk}",
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
        for arch, target in PE_TARGETS.items():
            with self.subTest(arch=arch):
                runner_marker = self.root / f"runner-{arch}-ran"
                runner = self.root / f"runner-{arch}"
                runner.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(runner_marker))}\n")
                runner.chmod(0o755)
                result = self.run_make(
                    "-o",
                    f"install-windows-{arch}",
                    "-o",
                    "install-unix-x64",
                    "MAKE=true",
                    "SDK_UNIX_ARCH=x64",
                    f"E2E_RUNNER={runner}",
                    "E2E_RUNNER_DIR=.",
                    f"test-e2e-{arch}",
                    fail_target=target,
                )
                self.assertFalse(runner_marker.exists(), result.stdout)
                self.assert_discovery_failed(result)

    def test_partial_output_is_not_staged(self):
        outputs = self.root / "outputs"
        out_i386 = outputs / "i386"
        out_x64 = outputs / "x86_64"
        out_unix_x64 = outputs / "unix-x64"
        out_unix_arm64 = outputs / "unix-arm64"
        for directory in (out_i386, out_x64, out_unix_x64, out_unix_arm64):
            directory.mkdir(parents=True)
        for directory in (out_i386, out_x64):
            for name in ("mtld3d.dll", "mtld3d.pdb", "mtld3d.fake.dll", "d3d9.dll", "d3d9.pdb"):
                (directory / name).touch()
        for directory in (out_unix_x64, out_unix_arm64):
            (directory / "mtld3d.so").touch()
            (directory / "mtld3d.so.dSYM").mkdir()
        for failed_arch, failed_target in PE_TARGETS.items():
            with self.subTest(failed_arch=failed_arch):
                stage_dir = self.root / f"stage-{failed_arch}"
                result = self.run_make(
                    "-o",
                    "all",
                    f"OUT_i386={out_i386}",
                    f"OUT_x64={out_x64}",
                    f"OUT_unix_x64={out_unix_x64}",
                    f"OUT_unix_arm64={out_unix_arm64}",
                    f"STAGE_DIR={stage_dir}",
                    f"STAGE_OUT={self.root / f'stage-{failed_arch}.tar'}",
                    "stage",
                    fail_target=failed_target,
                )
                for arch, test_exe in self.test_exes.items():
                    staged = stage_dir / f"tests/{arch}" / test_exe.name
                    if arch == "i686" and failed_arch == "x86_64":
                        self.assertTrue(staged.exists())
                    else:
                        self.assertFalse(staged.exists(), result.stdout)
                self.assert_discovery_failed(result)


if __name__ == "__main__":
    unittest.main()
