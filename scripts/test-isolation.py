#!/usr/bin/env python3
"""Check production Makefile path resolution without building or installing."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


MAKEFILE = Path(__file__).resolve().parent.parent / "Makefile"
FIELDS = ("WINE_SDK", "WINE_INSTALL_DIR", "WINEPREFIX", "WINE_LOADER", "WINEBUILD",
          "WINE_SERVER", "INSTALL_DIRS", "TEST_PREFIX")
MODE_FIELDS = ("WINE_SDK_MODE", "PE64_BUILD", "UNIX_ARCH", "ARM64X_LEG")
# The variables of the two-Wine layout, which the Makefile refuses at its top
# level; a shell that still exports one would stop every run here.
OLD_VARIABLES = ("WINE_X86_64", "WINE_ARM64", "WINE_SDK_ARM64X", "EC", "ARM64")


class MakeTestCase(unittest.TestCase):
    """The Makefile included by a wrapper, run in a scratch tree with its Wine inputs faked."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mtld3d-isolation-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.work = self.root / "work"
        self.work.mkdir()
        self.sources = {
            "WINE_SDK": str(self.root / "sdk-source"),
            "WINE_INSTALL_DIR": str(self.root / "install-source"),
            "WINEPREFIX": str(self.root / "prefix-source"),
        }
        for name, directory in self.sources.items():
            source = Path(directory)
            source.mkdir()
            (source / "sentinel").write_text(name)
        self.wrapper = self.work / "inspect.mk"
        self.wrapper.write_text(
            f"include {MAKEFILE}\n"
            ".PHONY: inspect recurse recurse-old\n"
            "inspect:\n"
            "\t@printf '%s\\n' "
            + " ".join(f"'resolved:{name}=$({name})'" for name in FIELDS)
            + "\nrecurse: inspect\n"
            "\t+$(MAKE) --no-print-directory -f $(firstword $(MAKEFILE_LIST)) inspect\n"
            "recurse-old:\n"
            "\t+$(MAKE) --no-print-directory -f $(firstword $(MAKEFILE_LIST)) "
            "WINE_X86_64=/old WINE_ARM64=/old WINE_SDK_ARM64X=/old EC=1 ARM64=1 inspect\n"
            "mode:\n"
            "\t@printf '%s\\n' "
            + " ".join(f"'resolved:{name}=$({name})'" for name in MODE_FIELDS)
            + "\nrequire:\n"
            "\t@$(WINE_SDK_REQUIRE)\n"
        )
        self.environment = os.environ.copy()
        for name in ("MAKEFLAGS", "MFLAGS", "GNUMAKEFLAGS", "MAKEOVERRIDES",
                     "MAKELEVEL", "ISOLATED", "STAGE", "WINE_TESTS", *OLD_VARIABLES, *self.sources):
            self.environment.pop(name, None)
        self.environment.update(self.sources)

    def run_make(self, *arguments, succeeds=True):
        isolation_root = self.work / ".wine-isolated"
        result = subprocess.run(
            ["make", "--no-print-directory", "-f", str(self.wrapper),
             f"ISOLATED_ROOT={isolation_root}", "ISOLATED_REGISTRY=",
             "BUILD_ID=isolation-test", *arguments],
            cwd=MAKEFILE.parent, env=self.environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
        )
        self.assertEqual(result.returncode == 0, succeeds, result.stdout)
        return result.stdout

    def resolved(self, output):
        return dict(line[len("resolved:"):].split("=", 1) for line in output.splitlines()
                    if line.startswith("resolved:"))


class IsolationTests(MakeTestCase):
    def assert_paths(self, output, isolated, copies=1):
        sdk = str(self.work / ".wine-isolated/sdk") if isolated else self.sources["WINE_SDK"]
        prefix = str(self.work / ".wine-isolated/prefix") if isolated else self.sources["WINEPREFIX"]
        install = sdk if isolated else self.sources["WINE_INSTALL_DIR"]
        expected = dict(WINE_SDK=sdk, WINE_INSTALL_DIR=install, WINEPREFIX=prefix,
                        WINE_LOADER=f"{sdk}/bin/wine", WINEBUILD=f"{sdk}/bin/winebuild",
                        WINE_SERVER=f"{sdk}/bin/wineserver", TEST_PREFIX=prefix,
                        INSTALL_DIRS=" ".join(sorted({sdk, install})))
        actual = [line for line in output.splitlines() if line.startswith("resolved:")]
        self.assertEqual(actual, [f"resolved:{name}={expected[name]}" for name in FIELDS] * copies,
                         output)
        if isolated:
            banner = (f"==> ISOLATED=1: WINE_SDK={sdk} WINE_INSTALL_DIR={install} "
                      f"WINEPREFIX={prefix}")
            actual_banners = [line for line in output.splitlines()
                              if line.startswith("==> ISOLATED=1:")]
            self.assertEqual(actual_banners, [banner] * copies, output)
            for directory, source in (("sdk", "WINE_SDK"), ("prefix", "WINEPREFIX")):
                self.assertEqual((self.work / ".wine-isolated" / directory / "sentinel").read_text(),
                                 source)
            self.assertEqual(sorted(path.name for path in (self.work / ".wine-isolated").iterdir()),
                             ["prefix", "sdk", "sdk.source"])
            record = (self.work / ".wine-isolated/sdk.source").read_text()
            self.assertTrue(record.startswith(f"{self.sources['WINE_SDK']} unread "), record)
        else:
            self.assertFalse((self.work / ".wine-isolated").exists())
        for name, directory in self.sources.items():
            self.assertEqual(list(Path(directory).iterdir()), [Path(directory) / "sentinel"])
            self.assertEqual((Path(directory) / "sentinel").read_text(), name)

    def test_environment_inputs(self):
        self.assert_paths(self.run_make("ISOLATED=1", "inspect"), True)

    def test_command_line_inputs_and_recursive_make(self):
        arguments = [f"{name}={value}" for name, value in self.sources.items()]
        self.assert_paths(self.run_make("ISOLATED=1", *arguments, "recurse"), True, copies=2)

    def test_environment_override_mode(self):
        self.assert_paths(self.run_make("-e", "ISOLATED=1", "recurse"), True, copies=2)

    def test_nonisolated_paths_are_preserved(self):
        self.assert_paths(self.run_make("inspect"), False)

    def test_cleaner_only_goals_do_not_clone(self):
        arguments = [f"{name}={value}" for name, value in self.sources.items()]
        for goals in (("clean-isolated",), ("clean-isolated-orphans",),
                      ("clean-isolated", "clean-isolated-orphans")):
            with self.subTest(goals=goals):
                self.run_make("-n", "ISOLATED=1", *arguments, *goals)
                self.assertFalse((self.work / ".wine-isolated").exists())

    def test_the_bench_comparisons_refuse_isolation_before_cloning(self):
        for goal in ("bench-ab", "bench-variants"):
            with self.subTest(goal=goal):
                output = self.run_make("ISOLATED=1", "BASE=HEAD", goal, succeeds=False)
                self.assertIn("run it without ISOLATED=1", output)
                self.assertFalse((self.work / ".wine-isolated").exists())


class VariableTests(MakeTestCase):
    """WINE_SDK as the one Wine input, the refusals, and what a bench-ab leg is given."""

    def test_an_old_variable_stops_the_run(self):
        for name in OLD_VARIABLES:
            for where in ("environment", "command line"):
                with self.subTest(name=name, where=where):
                    if where == "environment":
                        self.environment[name] = "1"
                        output = self.run_make("inspect", succeeds=False)
                        del self.environment[name]
                    else:
                        output = self.run_make(f"{name}=1", "inspect", succeeds=False)
                    self.assertIn(f"no longer read: {name}.", output)
                    self.assertIn("WINE_SDK names the one Wine", output)
                    self.assertIn("an arm64 or an x86_64 Wine", output)
                    self.assertIn("the ARM64X build is built whenever it is the arm64 one", output)

    def test_old_variables_given_to_a_sub_make_stop_it_too(self):
        # Only a BASE of an older layout is ever given them (`make bench-ab`).
        output = self.run_make("recurse-old", succeeds=False)
        self.assertIn("no longer read: WINE_X86_64 WINE_ARM64 WINE_SDK_ARM64X EC ARM64.", output)

    def test_without_wine_sdk_the_makefile_names_it(self):
        self.environment.pop("WINE_SDK")
        output = self.run_make("inspect", succeeds=False)
        self.assertIn("WINE_SDK is not set", output)

    def test_an_install_dir_naming_wine_sdk_is_written_once(self):
        sdk = self.sources["WINE_SDK"]
        for install in (sdk, f"{sdk}/", f"{sdk}/../sdk-source"):
            with self.subTest(install=install):
                values = self.resolved(self.run_make(f"WINE_INSTALL_DIR={install}", "inspect"))
                self.assertEqual(values["INSTALL_DIRS"], sdk)

    def test_only_a_bench_ab_base_of_an_older_layout_gets_the_old_variables(self):
        wrapper = self.root / "show.mk"
        wrapper.write_text(f"include {MAKEFILE}\nshow:\n"
                           "\t@printf 'cand:%s\\nbase:%s\\n' \"$(BENCH_LEG_MAKE)\" \"$(BENCH_BASE_MAKE)\"\n")
        sdk = self.sources["WINE_SDK"]
        old = (f"WINE_ARM64='{sdk}'", f"WINE_SDK_ARM64X='{sdk}'", "EC=1")
        for one_wine in ("", "1"):
            with self.subTest(base_reads_wine_sdk_alone=one_wine):
                output = subprocess.run(
                    ["make", "--no-print-directory", "-f", str(wrapper),
                     f"BENCH_BASE_ONE_WINE={one_wine}", "show"],
                    cwd=MAKEFILE.parent, env=self.environment, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30, check=True,
                ).stdout
                cand, base = (line.split(":", 1)[1] for line in output.splitlines()[-2:])
                for line in (cand, base):
                    self.assertIn(f"WINE_SDK='{sdk}'", line)
                    self.assertIn("PROD=1 PERF=1", line)
                    self.assertNotIn("ISOLATED", line)
                for assignment in old:
                    self.assertNotIn(assignment, cand)
                    self.assertEqual(assignment in base, not one_wine, base)

    def test_a_benchmark_takes_the_i686_or_the_arm64x_build(self):
        output = self.run_make("ARCH=x86_64", "BASE=HEAD", "bench-ab", succeeds=False)
        self.assertIn("ARCH=x86_64: a benchmark measures the i686 build", output)
        output = self.run_make("ARCH=i686", "bench-variants", succeeds=False)
        self.assertIn("takes no ARCH", output)


class FakeWineTestCase(MakeTestCase):
    """A fake arm64 Wine as WINE_SDK, a fake x86_64 Wine beside it, and a stage of every build."""

    X86_64_DIRS = ("bin", "lib/wine/x86_64-unix", "lib/wine/i386-windows", "lib/wine/x86_64-windows",
                   "lib/wine/d3d9/mtld3d/i386-windows", "lib/wine/d3d9/mtld3d/x86_64-windows",
                   "lib/wine/d3d9/mtld3d/x86_64-unix")
    ARM64_DIRS = ("bin", "lib/wine/aarch64-unix", "lib/wine/i386-windows", "lib/wine/aarch64-windows",
                  "lib/wine/x86_64-windows", "lib/wine/d3d9/mtld3d/i386-windows",
                  "lib/wine/d3d9/mtld3d/aarch64-windows", "lib/wine/d3d9/mtld3d/aarch64-unix")
    SDK_FILES = ("lib/wine/aarch64-unix/compatdb.so", "lib/wine/i386-windows/libwinecrt0.a",
                 "lib/wine/x86_64-windows/libwinecrt0.a", "lib/wine/aarch64-windows/libwinecrt0.a",
                 "lib/wine/aarch64-windows/libntdll.a")
    X86_64_SDK_FILES = ("lib/wine/i386-windows/libwinecrt0.a", "lib/wine/x86_64-windows/libwinecrt0.a")

    def setUp(self):
        super().setUp()
        self.x86_64 = self.make_wine("x86_64", self.X86_64_DIRS)
        self.arm64 = self.make_wine("arm64", self.ARM64_DIRS)
        for wine, files in ((self.arm64, self.SDK_FILES), (self.x86_64, self.X86_64_SDK_FILES)):
            for name in files:
                (wine / name).touch()
            # winebuild writes its arguments into the file `-o` names and leaves
            # everything else as it is.
            winebuild = wine / "bin/winebuild"
            winebuild.write_text("#!/bin/sh\nall=\"$*\"\n"
                                 "while [ $# -gt 0 ]; do [ \"$1\" = -o ] && printf '%s\\n' \"$all\" > \"$2\"; shift; done\n")
            winebuild.chmod(0o755)
        self.stage = self.root / "stage"
        for arch in ("i386-windows", "x86_64-windows", "aarch64-windows"):
            (self.stage / arch).mkdir(parents=True)
            for name in ("d3d9.dll", "d3d9.pdb", "mtld3d.dll", "mtld3d.pdb"):
                (self.stage / arch / name).write_text(arch)
        for arch in ("x86_64-unix", "aarch64-unix"):
            (self.stage / arch / "mtld3d.so.dSYM").mkdir(parents=True)
            (self.stage / arch / "mtld3d.so").write_text(arch)
        self.environment.pop("WINE_INSTALL_DIR")
        self.environment["WINE_SDK"] = str(self.arm64)

    def make_wine(self, name, directories):
        root = self.root / name
        for directory in directories:
            (root / directory).mkdir(parents=True)
        for program in ("wine", "wineserver"):
            path = root / "bin" / program
            path.write_text("#!/bin/sh\n")
            path.chmod(0o755)
        return root

    def directories(self, root):
        return sorted(str(path.relative_to(root)) for path in root.rglob("*")
                      if path.is_dir() and not path.name.endswith(".dSYM"))

    def install(self, *arguments, succeeds=True):
        return self.run_make(f"STAGE={self.stage}", *arguments, "install", succeeds=succeeds)


class ModeTests(FakeWineTestCase):
    """The two modes, read from WINE_SDK's own `lib/wine`, and what each one's aggregates run."""

    def dry_run(self, wine, *goals):
        # `-n` prints every recipe line, a sub-make's included, and runs none.
        return self.run_make(f"WINE_SDK={wine}", "-n", *goals)

    def test_the_mode_follows_the_wine(self):
        expected = {self.arm64: ("arm64", "arm64x", "arm64", "arm64x"),
                    self.x86_64: ("x86_64", "x86_64", "x64", "")}
        for wine, values in expected.items():
            with self.subTest(wine=wine.name):
                resolved = self.resolved(self.run_make(f"WINE_SDK={wine}", "mode"))
                self.assertEqual(tuple(resolved[name] for name in MODE_FIELDS), values)
        # A tree that is no Wine yet is held to the arm64 mode's requirements.
        resolved = self.resolved(self.run_make(f"WINE_SDK={self.root / 'none'}", "mode"))
        self.assertEqual(resolved["WINE_SDK_MODE"], "arm64")

    def test_the_x86_64_mode_says_so_once(self):
        line = "==> WINE_SDK is an x86_64 Wine: x86_64 mode, in which the ARM64X build is not built"
        output = self.run_make(f"WINE_SDK={self.x86_64}", "recurse")
        self.assertEqual([entry for entry in output.splitlines() if entry.startswith("==> WINE_SDK")],
                         [f"{line}; the legs are i686 and x86_64"])
        self.assertNotIn("==> WINE_SDK", self.run_make("recurse"))

    def test_the_arm64x_build_is_built_linted_and_bundled_in_the_arm64_mode_alone(self):
        for goals in (("windows",), ("check",), ("-o", "all", "bundle"), ("-o", "all", "stage")):
            with self.subTest(goals=goals):
                arm64 = self.dry_run(self.arm64, *goals)
                x86_64 = self.dry_run(self.x86_64, *goals)
                self.assertTrue("arm64x" in arm64 or "aarch64-pc-windows-msvc" in arm64, arm64)
                for word in ("arm64x", "aarch64-pc-windows-msvc", "arm64ec", "LLVM_MINGW",
                             "lld-link", "aarch64-windows"):
                    self.assertNotIn(word, x86_64)
        self.assertIn("native/aarch64-windows", self.dry_run(self.arm64, "-o", "all", "bundle"))

    def test_make_test_and_conformance_run_the_modes_own_legs(self):
        arm64 = self.dry_run(self.arm64, "test")
        x86_64 = self.dry_run(self.x86_64, "test")
        self.assertIn("arm64x/release/d3d9.dll", arm64)
        self.assertNotIn("x86_64-pc-windows-msvc/release/d3d9.dll", arm64)
        self.assertIn("x86_64-pc-windows-msvc/release/d3d9.dll", x86_64)
        self.assertNotIn("arm64x", x86_64)
        for output in (arm64, x86_64):
            self.assertNotIn(".wine-isolated/leg-", output)
        self.assertIn(".wine-isolated/leg-x86_64", self.dry_run(self.arm64, "test-e2e-x86_64"))
        self.assertNotIn(".wine-isolated", self.dry_run(self.x86_64, "test-e2e-x86_64"))
        # Each Wine records under keys of its own, so neither rewrites the other's.
        legs = {self.arm64: ["i686-arm64", "arm64x"], self.x86_64: ["i686", "x86_64"]}
        for wine, expected in legs.items():
            # `conformance-baseline` records the native, intel and scale variants of both.
            for goal, variant, rounds in (("conformance", "", 1),
                                          ("conformance-baseline", "--update-baseline", 3),
                                          ("conformance-intel", "--variant intel", 1),
                                          ("conformance-scale", "--variant scale", 1)):
                with self.subTest(wine=wine.name, goal=goal):
                    output = self.dry_run(wine, goal)
                    runs = [line for line in output.splitlines() if "--arch " in line]
                    self.assertEqual([re.search(r"--arch (\S+)", line)[1] for line in runs],
                                     expected * rounds)
                    for line in runs:
                        self.assertIn(variant, line)

    def test_every_leg_runs_the_tests_wine_tests_names(self):
        # One test build for both Wines: the arm64 Wine's own by default, and
        # under an x86_64 Wine only the directory WINE_TESTS names.
        tests = self.root / "arm64-release-tests"
        for wine, arguments, expected in ((self.arm64, (), self.arm64 / "lib/wine/tests"),
                                          (self.arm64, (f"WINE_TESTS={tests}",), tests),
                                          (self.x86_64, (f"WINE_TESTS={tests}",), tests)):
            with self.subTest(wine=wine.name, arguments=arguments):
                output = self.dry_run(wine, *arguments, "conformance")
                exes = re.findall(r"--exe (\S+)", output)
                self.assertEqual(exes, [f"{expected}/i386-windows/d3d9_test.exe",
                                        f"{expected}/x86_64-windows/d3d9_test.exe"])
        output = self.run_make(f"WINE_SDK={self.x86_64}", "-o", "install-windows-i686", "-o",
                               "install-unix-x64", "MAKE=true", "conformance-i686", succeeds=False)
        self.assertIn("WINE_TESTS is not set: in the x86_64 mode", output)
        self.assertNotIn("--exe", output)

    def test_conformance_isolate_takes_a_leg_of_the_mode(self):
        for wine, good, bad in ((self.arm64, ("i686", "arm64x"), "x86_64"),
                                (self.x86_64, ("i686", "x86_64"), "arm64x")):
            with self.subTest(wine=wine.name):
                output = self.run_make(f"WINE_SDK={wine}", f"ARCH={bad}", "-n", "conformance-isolate",
                                       succeeds=False)
                self.assertIn(f"ARCH={bad}: `make conformance-isolate` runs a leg of the", output)
                labels = [re.search(r"--arch (\S+)", line)[1]
                          for arch in good
                          for line in self.dry_run(wine, f"ARCH={arch}", "conformance-isolate").splitlines()
                          if "--arch " in line]
                self.assertEqual(labels, ["i686-arm64" if wine == self.arm64 else "i686", good[1]])

    def test_the_other_modes_legs_refuse(self):
        for wine, goal in ((self.arm64, "conformance-x86_64"), (self.arm64, "conformance-baseline-intel-x86_64"),
                           (self.x86_64, "conformance-arm64x"), (self.x86_64, "test-e2e-arm64x")):
            with self.subTest(wine=wine.name, goal=goal):
                output = self.run_make(f"WINE_SDK={wine}", goal, succeeds=False)
                self.assertIn(f"{goal} runs the", output)
        for goal in ("windows-arm64x", "clippy-pe-arm64x"):
            with self.subTest(goal=goal):
                output = self.run_make(f"WINE_SDK={self.x86_64}", goal, succeeds=False)
                self.assertIn(f"{goal} needs an arm64 Wine as WINE_SDK", output)
        output = self.run_make(f"WINE_SDK={self.x86_64}", "bench-variants", succeeds=False)
        self.assertIn("needs an arm64 Wine as WINE_SDK", output)
        output = self.run_make(f"WINE_SDK={self.x86_64}", "ARCH=arm64x", "BASE=HEAD", "bench-ab",
                               succeeds=False)
        self.assertIn("ARCH=x86_64 in the x86_64 mode", output)

    def test_a_bench_ab_base_gets_the_variables_of_its_own_layout(self):
        # BENCH_BASE_ONE_WINE= stands for a BASE whose Makefile predates WINE_SDK
        # as the one Wine; HEAD reads WINE_SDK alone and gets none of the old set.
        for wine, expected, absent in ((self.arm64, ("WINE_ARM64=", "EC=1"), "WINE_X86_64="),
                                       (self.x86_64, ("WINE_X86_64=",), "EC=1")):
            with self.subTest(wine=wine.name):
                output = self.dry_run(wine, "BASE=HEAD", "BENCH_BASE_ONE_WINE=", "bench-ab")
                base = next(line for line in output.splitlines() if " -C " in line and "PROD=1" in line)
                for word in expected:
                    self.assertIn(word, base)
                self.assertNotIn(absent, base)
                output = self.dry_run(wine, "BASE=HEAD", "bench-ab")
                base = next(line for line in output.splitlines() if " -C " in line and "PROD=1" in line)
                for word in ("WINE_ARM64=", "WINE_X86_64=", "WINE_SDK_ARM64X=", "EC=1"):
                    self.assertNotIn(word, base)

    def test_an_isolated_clone_of_another_wine_is_refused(self):
        clone = self.work / ".wine-isolated/sdk"
        self.run_make("ISOLATED=1", "mode")
        self.assertTrue(clone.is_dir())
        resolved = self.resolved(self.run_make("ISOLATED=1", "mode"))
        self.assertEqual(resolved["WINE_SDK_MODE"], "arm64")
        # Another Wine, a redeploy of the same one, and a clone with no record.
        wineserver = self.arm64 / "bin/wineserver"
        for change in ("other wine", "redeploy", "no record"):
            with self.subTest(change=change):
                arguments = []
                if change == "other wine":
                    arguments.append(f"WINE_SDK={self.x86_64}")
                elif change == "redeploy":
                    saved = wineserver.read_text()
                    wineserver.unlink()
                    wineserver.write_text(saved)
                    wineserver.chmod(0o755)
                else:
                    (self.work / ".wine-isolated/sdk.source").unlink()
                output = self.run_make("ISOLATED=1", *arguments, "mode", succeeds=False)
                self.assertIn(f"ISOLATED=1: {clone} was cloned from", output)
                self.assertIn("`make clean-isolated` removes the clones", output)
                self.run_make("ISOLATED=1", "clean-isolated")
                self.assertFalse(clone.exists())
                self.run_make("ISOLATED=1", "mode")


class InstallLayoutTests(FakeWineTestCase):
    """`make install` against fake Wines: what goes where, and no arch directory created."""

    def test_the_arm64_wine_gets_the_arm64x_build_and_no_x86_64_piece(self):
        before = self.directories(self.arm64)
        self.install()
        self.assertEqual(self.directories(self.arm64), before)
        tree = self.arm64 / "lib/wine/d3d9/mtld3d"
        self.assertEqual((tree / "aarch64-windows/d3d9.dll").read_text(), "aarch64-windows")
        self.assertEqual((tree / "aarch64-windows/mtld3d.dll").read_text(), "aarch64-windows")
        self.assertEqual((tree / "i386-windows/mtld3d.dll").read_text(), "i386-windows")
        self.assertEqual((tree / "aarch64-unix/mtld3d.so").read_text(), "aarch64-unix")
        self.assertTrue((self.arm64 / "lib/wine/aarch64-windows/d3d9.dll").exists())
        self.assertFalse((self.arm64 / "lib/wine/x86_64-windows/d3d9.dll").exists())
        self.assertFalse(list(self.arm64.rglob("x86_64-unix")))

    def test_an_x86_64_install_dir_gets_its_own_pieces_beside_the_arm64_wine(self):
        before = {wine: self.directories(wine) for wine in (self.x86_64, self.arm64)}
        self.install(f"WINE_INSTALL_DIR={self.x86_64}")
        for wine in (self.x86_64, self.arm64):
            self.assertEqual(self.directories(wine), before[wine])
        tree = self.x86_64 / "lib/wine/d3d9/mtld3d"
        self.assertEqual((tree / "x86_64-windows/d3d9.dll").read_text(), "x86_64-windows")
        self.assertEqual((tree / "i386-windows/d3d9.dll").read_text(), "i386-windows")
        self.assertEqual((tree / "x86_64-unix/mtld3d.so").read_text(), "x86_64-unix")
        self.assertTrue((self.x86_64 / "lib/wine/x86_64-windows/d3d9.dll").exists())
        self.assertFalse(list(self.x86_64.rglob("aarch64-*")))
        tree = self.arm64 / "lib/wine/d3d9/mtld3d"
        self.assertEqual((tree / "aarch64-windows/d3d9.dll").read_text(), "aarch64-windows")

    def test_a_missing_arch_directory_fails_the_install_and_is_not_created(self):
        missing = self.arm64 / "lib/wine/d3d9/mtld3d/aarch64-windows"
        missing.rmdir()
        output = self.install(succeeds=False)
        self.assertIn(f"{missing} does not exist", output)
        self.assertFalse(missing.exists())

    def test_the_64_bit_markers_are_x64_ones_in_both_wines(self):
        self.install(f"WINE_INSTALL_DIR={self.x86_64}")
        for marker in (self.x86_64 / "lib/wine/x86_64-windows/d3d9.dll",
                       self.arm64 / "lib/wine/aarch64-windows/d3d9.dll",
                       self.arm64 / "lib/wine/aarch64-windows/mtld3d.dll"):
            self.assertIn("-b x86_64-windows", marker.read_text())
        self.assertIn("-b i686-windows", (self.arm64 / "lib/wine/i386-windows/d3d9.dll").read_text())

    def test_an_isolated_install_writes_the_clone_and_not_the_wines(self):
        before = {wine: sorted(str(path) for path in wine.rglob("*")) for wine in (self.x86_64, self.arm64)}
        self.install("ISOLATED=1", f"WINE_INSTALL_DIR={self.x86_64}")
        for wine in (self.x86_64, self.arm64):
            self.assertEqual(sorted(str(path) for path in wine.rglob("*")), before[wine])
        clones = self.work / ".wine-isolated"
        self.assertEqual(sorted(path.name for path in clones.iterdir()), ["prefix", "sdk", "sdk.source"])
        self.assertEqual((clones / "sdk/lib/wine/d3d9/mtld3d/aarch64-windows/d3d9.dll").read_text(),
                         "aarch64-windows")

    def test_a_flat_x86_64_wine_gets_the_build_in_its_default_dirs(self):
        for directory in ("i386-windows", "x86_64-windows", "x86_64-unix"):
            (self.x86_64 / "lib/wine/d3d9/mtld3d" / directory).rmdir()
        (self.x86_64 / "lib/wine/d3d9/mtld3d").rmdir()
        (self.x86_64 / "lib/wine/d3d9").rmdir()
        before = self.directories(self.x86_64)
        self.install(f"WINE_INSTALL_DIR={self.x86_64}")
        self.assertEqual(self.directories(self.x86_64), before)
        self.assertEqual((self.x86_64 / "lib/wine/x86_64-windows/d3d9.dll").read_text(), "x86_64-windows")
        self.assertEqual((self.x86_64 / "lib/wine/i386-windows/mtld3d.dll").read_text(), "i386-windows")
        self.assertEqual((self.x86_64 / "lib/wine/x86_64-unix/mtld3d.so").read_text(), "x86_64-unix")

    def test_a_wine_with_both_unix_dirs_is_refused_naming_the_leftover(self):
        (self.arm64 / "lib/wine/x86_64-unix").mkdir()
        output = self.install(succeeds=False)
        self.assertIn("has both lib/wine/x86_64-unix and lib/wine/aarch64-unix", output)
        self.assertIn("can be removed", output)

    def test_an_arm64_wine_without_the_mtld3d_tree_is_refused(self):
        for directory in ("i386-windows", "aarch64-windows", "aarch64-unix"):
            (self.arm64 / "lib/wine/d3d9/mtld3d" / directory).rmdir()
        (self.arm64 / "lib/wine/d3d9/mtld3d").rmdir()
        output = self.install(succeeds=False)
        self.assertIn("is an arm64 Wine without lib/wine/d3d9/mtld3d", output)
        self.assertFalse((self.arm64 / "lib/wine/d3d9").joinpath("mtld3d").exists())

    def test_a_build_against_a_wine_missing_a_piece_refuses_naming_it(self):
        for target in ("windows-i686", "windows-arm64x", "clippy-pe-x86_64", "clippy-pe-arm64x",
                       "doc-windows"):
            for missing in ("bin/winebuild", "lib/wine/aarch64-windows/libntdll.a",
                            "lib/wine/i386-windows/libwinecrt0.a"):
                with self.subTest(target=target, missing=missing):
                    path = self.arm64 / missing
                    saved = path.read_bytes()
                    path.unlink()
                    output = self.run_make(target, succeeds=False)
                    path.write_bytes(saved)
                    path.chmod(0o755)
                    self.assertIn(f"WINE_SDK={self.arm64} ", output)
                    self.assertIn(f"no {missing}", output)

    def test_each_mode_requires_its_own_pieces(self):
        # A complete Wine of either arch passes; the x86_64 mode asks for no
        # ARM64X archive, compatdb.so or mtld3d tree, and the arm64 mode for all.
        for wine in (self.arm64, self.x86_64):
            with self.subTest(wine=wine.name):
                self.run_make(f"WINE_SDK={wine}", "require")
        for wine, missing in ((self.x86_64, "lib/wine/x86_64-windows/libwinecrt0.a"),
                              (self.arm64, "lib/wine/aarch64-unix/compatdb.so")):
            with self.subTest(wine=wine.name, missing=missing):
                (wine / missing).unlink()
                output = self.run_make(f"WINE_SDK={wine}", "require", succeeds=False)
                (wine / missing).touch()
                self.assertIn(f"holds no {missing}", output)
        shutil.rmtree(self.arm64 / "lib/wine/d3d9/mtld3d")
        output = self.run_make("require", succeeds=False)
        self.assertIn("has no lib/wine/d3d9/mtld3d tree", output)

    def test_the_x86_64_mode_installs_the_x86_64_build(self):
        before = self.directories(self.x86_64)
        self.install(f"WINE_SDK={self.x86_64}")
        self.assertEqual(self.directories(self.x86_64), before)
        tree = self.x86_64 / "lib/wine/d3d9/mtld3d"
        self.assertEqual((tree / "x86_64-windows/d3d9.dll").read_text(), "x86_64-windows")
        self.assertEqual((tree / "i386-windows/d3d9.dll").read_text(), "i386-windows")
        self.assertEqual((tree / "x86_64-unix/mtld3d.so").read_text(), "x86_64-unix")
        self.assertFalse(list(self.x86_64.rglob("aarch64-*")))

    def test_an_arm64_install_dir_beside_an_x86_64_wine_sdk_gets_the_x86_64_build(self):
        # The x86_64 mode builds no ARM64X pair, so the arm64 tree takes the
        # x86_64 one, and its PE and unix halves come from one build.
        tree = self.arm64 / "lib/wine/d3d9/mtld3d"
        (tree / "aarch64-windows/d3d9.dll").write_text("an older ARM64X build")
        before = self.directories(self.arm64)
        self.install(f"WINE_SDK={self.x86_64}", f"WINE_INSTALL_DIR={self.arm64}")
        self.assertEqual(self.directories(self.arm64), before)
        self.assertEqual((tree / "aarch64-windows/d3d9.dll").read_text(), "x86_64-windows")
        self.assertEqual((tree / "aarch64-windows/mtld3d.dll").read_text(), "x86_64-windows")
        self.assertEqual((tree / "i386-windows/d3d9.dll").read_text(), "i386-windows")
        self.assertEqual((tree / "aarch64-unix/mtld3d.so").read_text(), "aarch64-unix")
        self.assertIn("-b x86_64-windows", (self.arm64 / "lib/wine/aarch64-windows/d3d9.dll").read_text())
        self.assertEqual((self.x86_64 / "lib/wine/d3d9/mtld3d/x86_64-windows/d3d9.dll").read_text(),
                         "x86_64-windows")

    def test_the_x86_64_leg_installs_the_x86_64_build_into_a_clone_of_its_own(self):
        before = sorted(str(path) for path in self.arm64.rglob("*"))
        (self.stage / "tests/x86_64").mkdir(parents=True)
        (self.stage / "tests/x86_64/e2e-fixture.exe").touch()
        runner = self.stage / "e2e" / os.uname().machine / "mtld3d-e2e"
        runner.parent.mkdir(parents=True)
        runner.write_text("#!/bin/sh\nprintf 'runner:%s\\n' \"$*\"\n")
        runner.chmod(0o755)
        output = self.run_make(f"STAGE={self.stage}", "LEG_SUBMAKE=true", "test-e2e-x86_64")
        self.assertEqual(sorted(str(path) for path in self.arm64.rglob("*")), before)
        clone = self.work / ".wine-isolated/leg-x86_64/sdk"
        tree = clone / "lib/wine/d3d9/mtld3d"
        self.assertEqual((tree / "aarch64-windows/d3d9.dll").read_text(), "x86_64-windows")
        self.assertEqual((tree / "i386-windows/d3d9.dll").read_text(), "i386-windows")
        self.assertEqual((tree / "aarch64-unix/mtld3d.so").read_text(), "aarch64-unix")
        self.assertIn("-b x86_64-windows", (clone / "lib/wine/aarch64-windows/d3d9.dll").read_text())
        self.assertIn(f"--wine {clone}/bin/wine", output)


USER_REG = """WINE REGISTRY Version 2
;; All keys relative to \\\\User\\\\S-1-5-21-0-0-0-1000

#arch=win64

[Software\\\\Wine\\\\Mac Driver] 1787051437
#time=1dd2f02325c1100
"RetinaMode"="{retina}"

[Software\\\\Wine\\\\WineDbg] 1787047719
#time=1dd2ef98a532d16
"ShowCrashDialog"=dword:00000000

[Software\\\\Wine\\\\X11 Driver] 1787051373
#time=1dd2f020c8a89fc
"EmulateModeset"="Y"
"""


class ConfigureTests(MakeTestCase):
    """Which way `configure-test-prefix-locked` goes, with Wine played by scripts."""

    def configure_decision(self, server_running, user_reg):
        prefix = Path(self.sources["WINEPREFIX"])
        if user_reg is not None:
            (prefix / "user.reg").write_text(user_reg)
        wineserver = self.work / "wineserver"
        wineserver.write_text(f"#!/bin/sh\n[ \"$1\" = -k0 ] && exit {0 if server_running else 1}\nexit 0\n")
        wine = self.work / "wine"
        wine.write_text("#!/bin/sh\nexit 1\n")
        for script in (wineserver, wine):
            script.chmod(0o755)
        output = self.run_make(f"WINE_SERVER={wineserver}", f"WINE_LOADER={wine}", "MAKE=echo sub-make",
                               "configure-test-prefix-locked")
        return [line for line in output.splitlines() if line.startswith("sub-make ")]

    def test_a_prefix_whose_file_holds_the_keys_only_boots(self):
        self.assertEqual(self.configure_decision(False, USER_REG.format(retina="Y")),
                         ["sub-make configure-test-prefix-boot"])

    def test_a_prefix_missing_a_key_or_its_file_is_configured(self):
        self.assertEqual(self.configure_decision(False, USER_REG.format(retina="N")),
                         ["sub-make configure-test-prefix-session"])
        (Path(self.sources["WINEPREFIX"]) / "user.reg").unlink()
        self.assertEqual(self.configure_decision(False, None),
                         ["sub-make configure-test-prefix-session"])

    def fake_wine(self, wineboot_status, server_after):
        wineserver = self.work / "wineserver"
        wineserver.write_text(f"#!/bin/sh\n[ \"$1\" = -k0 ] && exit {0 if server_after else 1}\nexit 0\n")
        wine = self.work / "wine"
        wine.write_text(f"#!/bin/sh\necho 'wineboot said this'\nexit {wineboot_status}\n")
        for script in (wineserver, wine):
            script.chmod(0o755)
        return [f"WINE_SERVER={wineserver}", f"WINE_LOADER={wine}"]

    def test_the_boot_alone_fails_loudly(self):
        self.run_make(*self.fake_wine(0, True), "configure-test-prefix-boot")
        for status, server in ((1, True), (0, False)):
            with self.subTest(wineboot=status, server=server):
                output = self.run_make(*self.fake_wine(status, server),
                                       "configure-test-prefix-boot", succeeds=False)
                self.assertIn("failed or left no server", output)
                self.assertIn("wineboot said this", output)

    def test_a_running_server_is_asked_rather_than_its_file(self):
        # The server's keys do not read back (the fake `reg query` fails), and the
        # file is not trusted while a server may be about to rewrite it.
        self.assertEqual(self.configure_decision(True, USER_REG.format(retina="Y")),
                         ["sub-make configure-test-prefix-session"])


if __name__ == "__main__":
    unittest.main()
