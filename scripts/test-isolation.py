#!/usr/bin/env python3
"""Check production Makefile path resolution without building or installing."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


MAKEFILE = Path(__file__).resolve().parent.parent / "Makefile"
FIELDS = ("WINE_SDK", "WINE_INSTALL_DIR", "WINEPREFIX", "WINE_LOADER", "WINEBUILD",
          "WINE_SERVER", "INSTALL_DIRS", "TEST_PREFIX")
MODE_FIELDS = ("WINE_SDK_MODE", "PE64_BUILD", "UNIX_ARCH", "ARM64X_LEG")
# The variables of the two-Wine layout, which the Makefile refuses at its top
# level; a shell that still exports one would stop every run here.
OLD_VARIABLES = ("WINE_X86_64", "WINE_ARM64", "WINE_SDK_ARM64X", "EC", "ARM64")

# A fake PE image: the PE header at 0x40, as its offset at 0x3c says, with the
# signature, the COFF machine, the PE32+ magic 24 bytes past the signature
# and, 216 bytes past it, the load-config entry of the data directory (an RVA
# and a size, zero for none), then a label. Source text, because the fake winebuild runs it too.
PE_HEADER = 288
FAKE_PE = """\
def fake_pe(machine, label, load_config=False):
    header = bytearray(288)
    header[0:2] = b'MZ'
    header[60:64] = (64).to_bytes(4, 'little')
    header[64:68] = b'PE\\0\\0'
    header[68:70] = machine.to_bytes(2, 'little')
    header[88:90] = (0x20b).to_bytes(2, 'little')
    if load_config:
        header[280:288] = (0x2000).to_bytes(4, 'little') + (0x140).to_bytes(4, 'little')
    return bytes(header) + label.encode()
"""
exec(FAKE_PE)

# The fake winebuild writes into the file `-o` names a placeholder of the
# machine `-b` names, labelled with its arguments, and with the load config a
# hybrid one carries when `-marm64x` is given and `bin/no-arm64ec` is absent;
# that file stands for the winebuild of a Wine that predates hybrid
# placeholders. With `bin/truncated-arm64x` a `-marm64x` placeholder ends
# before its data directory. Without `-o` it leaves everything as it is.
FAKE_WINEBUILD = FAKE_PE + """\
import os, sys
args = sys.argv[1:]
flag = lambda name: os.path.exists(os.path.join(os.path.dirname(sys.argv[0]), name))
if '-o' in args:
    targets = {'i686-windows': 0x014c, 'x86_64-windows': 0x8664, 'aarch64-windows': 0xaa64}
    machine = targets[args[args.index('-b') + 1]]
    hybrid = '-marm64x' in args
    data = fake_pe(machine, ' '.join(args) + '\\n', hybrid and not flag('no-arm64ec'))
    with open(args[args.index('-o') + 1], 'wb') as out:
        out.write(data[:200] if hybrid and flag('truncated-arm64x') else data)
"""


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
    MACHINES = {"i386-windows": 0x014c, "x86_64-windows": 0x8664, "aarch64-windows": 0xaa64}
    PE_HEADER = PE_HEADER
    FAKE_WINEBUILD = FAKE_WINEBUILD
    fake_pe = staticmethod(fake_pe)

    def content(self, path):
        # A fake PE's label, or a text file as it is.
        data = path.read_bytes()
        return data[self.PE_HEADER:].decode() if data.startswith(b"MZ") else data.decode()

    def machine(self, path):
        return int.from_bytes(path.read_bytes()[68:70], "little")

    @staticmethod
    def printed(output, start):
        # The lines a recipe printed that begin with `start`. Under CI the
        # Makefile echoes every recipe (it is silent only without CI), and a
        # recipe's text holds its own messages with their shell variables
        # unexpanded, so a message is matched by the start of a line, which an
        # echoed recipe never has, rather than anywhere in the output.
        return [line for line in output.splitlines() if line.startswith(start)]

    def winebuild_lines(self, output):
        # The line the install fails with when the Wine's winebuild cannot write
        # a hybrid placeholder.
        return self.printed(output, f"{self.arm64}/bin/winebuild is too old")

    def setUp(self):
        super().setUp()
        self.x86_64 = self.make_wine("x86_64", self.X86_64_DIRS)
        self.arm64 = self.make_wine("arm64", self.ARM64_DIRS)
        for wine, files in ((self.arm64, self.SDK_FILES), (self.x86_64, self.X86_64_SDK_FILES)):
            for name in files:
                (wine / name).touch()
            winebuild = wine / "bin/winebuild"
            winebuild.write_text(f"#!{sys.executable} -I\n{self.FAKE_WINEBUILD}")
            winebuild.chmod(0o755)
        # Each staged DLL is a PE header of its build's machine (the ARM64X
        # build says ARM64) followed by the name of its directory, which
        # `content` reads back.
        self.stage = self.root / "stage"
        for arch in ("i386-windows", "x86_64-windows", "aarch64-windows"):
            (self.stage / arch).mkdir(parents=True)
            for name in ("d3d9.dll", "mtld3d.dll"):
                (self.stage / arch / name).write_bytes(self.fake_pe(self.MACHINES[arch], arch))
            for name in ("d3d9.pdb", "mtld3d.pdb"):
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
                self.assertNotIn("system32-arm64", x86_64)
        bundle = self.dry_run(self.arm64, "-o", "all", "bundle")
        self.assertIn("native/aarch64-windows", bundle)
        # An arm64 Wine's 64-bit prefix marker is the ARM64X build's hybrid one.
        self.assertRegex(bundle, r"cp -c \S*/arm64x/\w+/mtld3d\.fake\.dll \S*/prefix-markers/system32-arm64/mtld3d\.dll")
        self.assertIn("--fake-module -o windows/target/arm64x/release/mtld3d.fake.dll -b aarch64-windows -marm64x",
                      self.dry_run(self.arm64, "windows"))

    def test_make_test_and_conformance_run_the_modes_own_legs(self):
        arm64 = self.dry_run(self.arm64, "test")
        x86_64 = self.dry_run(self.x86_64, "test")
        self.assertIn("arm64x/release/d3d9.dll", arm64)
        self.assertNotIn("x86_64-pc-windows-msvc/release/d3d9.dll", arm64)
        self.assertIn("x86_64-pc-windows-msvc/release/d3d9.dll", x86_64)
        self.assertNotIn("arm64x", x86_64)
        # The arm64 mode's `make test` adds the native ARM64 leg, in a tree and
        # prefix of its own, running the aarch64 test binaries; the x86_64 mode
        # has none.
        self.assertIn(".wine-isolated/leg-arm64/sdk/bin/wine", arm64)
        self.assertIn("--target aarch64-pc-windows-msvc --message-format", arm64)
        self.assertNotIn(".wine-isolated/leg-x86_64", arm64)
        self.assertNotIn(".wine-isolated/leg-", x86_64)
        self.assertNotIn("aarch64-pc-windows-msvc", x86_64)
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

    def test_the_runner_is_told_the_wine_version_the_tests_came_from(self):
        # WINE_TESTS's VERSION first, then the `wine --version` of the tree the
        # tests lie in, else no version: never the Wine that runs them.
        (self.arm64 / "bin/wine").write_text(
            "#!/bin/sh\n[ \"$1\" = --version ] && echo wine-11.19-tree\n")
        alone = self.root / "tests-alone"
        stamped = self.root / "tests-stamped"
        stamped.mkdir()
        (stamped / "VERSION").write_text("wine-11.20-stamped\nanything after the first line\n")
        inside = self.arm64 / "lib/wine/tests"
        for wine, tests, expected in ((self.arm64, None, "wine-11.19-tree"),
                                      (self.arm64, stamped, "wine-11.20-stamped"),
                                      (self.x86_64, stamped, "wine-11.20-stamped"),
                                      (self.x86_64, inside, "wine-11.19-tree"),
                                      (self.x86_64, alone, None)):
            with self.subTest(wine=wine.name, tests=tests and tests.name):
                arguments = (f"WINE_TESTS={tests}",) if tests else ()
                for goal in ("conformance", "conformance-baseline"):
                    runs = [line for line in self.dry_run(wine, *arguments, goal).splitlines()
                            if "--arch " in line]
                    self.assertTrue(runs)
                    for line in runs:
                        if expected:
                            self.assertIn(f"--tests-wine-version '{expected}'", line)
                        else:
                            self.assertNotIn("--tests-wine-version", line)

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

    def test_the_native_arm64_leg_needs_the_aarch64_splat(self):
        # Without the splat's aarch64 libraries the leg stops naming
        # `make setup-xwin`, before it sets up its tree or links anything.
        output = self.run_make(f"WINE_SDK={self.arm64}", f"XWIN_ROOT={self.root / 'no-xwin'}",
                               "-o", "windows-i686", "-o", "windows-arm64x", "-o", "unix-arm64",
                               "test-e2e-arm64", succeeds=False)
        self.assertIn(f"{self.root / 'no-xwin'} has no crt/lib/aarch64, the libraries the native ARM64 "
                      "test executables link against; `make setup-xwin` adds them", output)
        self.assertFalse((self.work / ".wine-isolated/leg-arm64").exists())

    def test_the_other_modes_legs_refuse(self):
        for wine, goal in ((self.arm64, "conformance-x86_64"), (self.arm64, "conformance-baseline-intel-x86_64"),
                           (self.x86_64, "conformance-arm64x"), (self.x86_64, "test-e2e-arm64x"),
                           (self.x86_64, "test-e2e-arm64")):
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
        self.assertEqual(self.content(tree / "aarch64-windows/d3d9.dll"), "aarch64-windows")
        self.assertEqual(self.content(tree / "aarch64-windows/mtld3d.dll"), "aarch64-windows")
        self.assertEqual(self.content(tree / "i386-windows/mtld3d.dll"), "i386-windows")
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
        self.assertEqual(self.content(tree / "x86_64-windows/d3d9.dll"), "x86_64-windows")
        self.assertEqual(self.content(tree / "i386-windows/d3d9.dll"), "i386-windows")
        self.assertEqual((tree / "x86_64-unix/mtld3d.so").read_text(), "x86_64-unix")
        self.assertTrue((self.x86_64 / "lib/wine/x86_64-windows/d3d9.dll").exists())
        self.assertFalse(list(self.x86_64.rglob("aarch64-*")))
        tree = self.arm64 / "lib/wine/d3d9/mtld3d"
        self.assertEqual(self.content(tree / "aarch64-windows/d3d9.dll"), "aarch64-windows")

    def test_a_missing_arch_directory_fails_the_install_and_is_not_created(self):
        missing = self.arm64 / "lib/wine/d3d9/mtld3d/aarch64-windows"
        missing.rmdir()
        output = self.install(succeeds=False)
        self.assertIn(f"{missing} does not exist", output)
        self.assertFalse(missing.exists())

    def test_the_arm64x_pair_gets_hybrid_markers_and_the_x86_64_pair_amd64_ones(self):
        output = self.install(f"WINE_INSTALL_DIR={self.x86_64}")
        self.assertEqual(self.winebuild_lines(output), [])
        for name in ("d3d9", "mtld3d"):
            marker = self.arm64 / f"lib/wine/aarch64-windows/{name}.dll"
            self.assertIn("-b aarch64-windows -marm64x", self.content(marker))
            self.assertEqual(self.machine(marker), 0xaa64)
            self.assertIn("-b x86_64-windows", self.content(self.x86_64 / f"lib/wine/x86_64-windows/{name}.dll"))
            for wine in (self.arm64, self.x86_64):
                self.assertIn("-b i686-windows", self.content(wine / f"lib/wine/i386-windows/{name}.dll"))

    def snapshot(self, root):
        # Every file under `root` with its bytes, and every directory.
        return {str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
                for path in sorted(root.rglob("*"))}

    def test_a_winebuild_without_hybrid_placeholders_fails_the_install(self):
        # arm64-11.19-1's winebuild takes -marm64x but writes no load config,
        # which an x64 process refuses: the probe ahead of the first leaf
        # stops the install, before the i686 leaf has touched anything.
        (self.arm64 / "bin/no-arm64ec").touch()
        output = self.install(succeeds=False)
        lines = self.winebuild_lines(output)
        self.assertEqual(len(lines), 1, output)
        self.assertRegex(lines[0], r"^\S+ is too old: it writes a -marm64x placeholder without the "
                         r"ARM64EC metadata of a hybrid one, which an x64 process refuses "
                         r"\(\S+/probe\.dll\); WINE_SDK has to be wine-build arm64-11\.19-2 or later$")
        self.assertFalse((self.arm64 / "lib/wine/aarch64-windows/mtld3d.dll").exists())
        self.assertFalse((self.arm64 / "lib/wine/d3d9/mtld3d/i386-windows/d3d9.dll").exists())

    def test_an_old_winebuild_leaves_the_wine_as_it_was(self):
        # A Wine with a working install: its pair in the mtld3d tree and AMD64
        # placeholders the x64 programs load behind. Neither the probe ahead of
        # the leaves nor, without it, the ARM64X leaf's own check of the
        # placeholders it stamps elsewhere first may change a byte of it.
        tree = self.arm64 / "lib/wine/d3d9/mtld3d"
        for directory, machine in (("i386-windows", 0x014c), ("aarch64-windows", 0xaa64)):
            for name in ("d3d9", "mtld3d"):
                (tree / directory / f"{name}.dll").write_bytes(self.fake_pe(machine, "the working pair"))
        for name in ("d3d9", "mtld3d"):
            (self.arm64 / f"lib/wine/aarch64-windows/{name}.dll").write_bytes(
                self.fake_pe(0x8664, "a working AMD64 placeholder"))
            (self.arm64 / f"lib/wine/i386-windows/{name}.dll").write_bytes(
                self.fake_pe(0x014c, "a working i386 placeholder"))
        (self.arm64 / "bin/no-arm64ec").touch()
        before = self.snapshot(self.arm64)
        for arguments in ((), ("HYBRID_WINEBUILD_CHECK=",)):
            with self.subTest(arguments=arguments):
                goal = "install" if not arguments else "install-windows-arm64x"
                output = self.run_make(f"STAGE={self.stage}", *arguments, goal, succeeds=False)
                lines = self.winebuild_lines(output)
                self.assertEqual(len(lines), 1, output)
                # The probe's scratch file, or the leaf's scratch directory.
                self.assertIn("/mtld3d-markers." if arguments else "/mtld3d-probe.", lines[0])
                self.assertEqual(self.snapshot(self.arm64), before, output)

    def test_a_hybrid_marker_whose_load_config_cannot_be_read_fails_the_install_naming_it(self):
        (self.arm64 / "bin/truncated-arm64x").touch()
        output = self.install(succeeds=False)
        lines = [line for line in output.splitlines() if ", the placeholder " in line
                 and line.startswith("/")]
        self.assertEqual(len(lines), 1, output)
        self.assertRegex(lines[0], rf"^\S+/probe\.dll, the placeholder {re.escape(str(self.arm64))}"
                         r"/bin/winebuild just wrote, is no PE32\+ image whose load config make can read$")
        self.assertEqual(self.winebuild_lines(output), [])

    def test_a_d3d9_tree_with_an_x86_64_build_keeps_the_d3d9_marker_amd64(self):
        # wine-build's rule: a name's marker is hybrid only when every tree of
        # the family that has the DLL holds an ARM64 image of it.
        wined3d = self.arm64 / "lib/wine/d3d9/wined3d/aarch64-windows"
        wined3d.mkdir(parents=True)
        for machine, expected in ((0xaa64, "-b aarch64-windows -marm64x"), (0x8664, "-b x86_64-windows")):
            with self.subTest(machine=hex(machine)):
                (wined3d / "d3d9.dll").write_bytes(self.fake_pe(machine, "wined3d"))
                self.install()
                marker = self.content(self.arm64 / "lib/wine/aarch64-windows/d3d9.dll")
                self.assertIn(expected, marker)
                self.assertNotIn("x86_64-windows" if "marm64x" in expected else "marm64x", marker)
                # wined3d has no mtld3d.dll, so that name follows the ARM64X pair alone.
                self.assertIn("-b aarch64-windows -marm64x",
                              self.content(self.arm64 / "lib/wine/aarch64-windows/mtld3d.dll"))

    def test_a_d3d9_tree_dll_whose_machine_cannot_be_read_fails_the_install_naming_it(self):
        # Counting it as not ARM64 would silently stamp AMD64 and cut native
        # ARM64 processes off d3d9.dll; the bundle's rule stops instead.
        wined3d = self.arm64 / "lib/wine/d3d9/wined3d/aarch64-windows"
        wined3d.mkdir(parents=True)
        dll = wined3d / "d3d9.dll"
        whole = self.fake_pe(0xaa64, "wined3d")
        for name, data in (("truncated before the machine", whole[:66]),
                           ("truncated before the PE offset", whole[:40]),
                           ("no PE signature", whole[:64] + b"NE\0\0" + whole[68:]),
                           ("not a PE at all", b"just text, no header")):
            with self.subTest(name=name):
                dll.write_bytes(data)
                output = self.install(succeeds=False)
                lines = self.printed(output, f"{dll} is no PE image whose machine make install can read")
                self.assertEqual(len(lines), 1, output)

    def test_an_isolated_install_writes_the_clone_and_not_the_wines(self):
        before = {wine: sorted(str(path) for path in wine.rglob("*")) for wine in (self.x86_64, self.arm64)}
        self.install("ISOLATED=1", f"WINE_INSTALL_DIR={self.x86_64}")
        for wine in (self.x86_64, self.arm64):
            self.assertEqual(sorted(str(path) for path in wine.rglob("*")), before[wine])
        clones = self.work / ".wine-isolated"
        self.assertEqual(sorted(path.name for path in clones.iterdir()), ["prefix", "sdk", "sdk.source"])
        self.assertEqual(self.content(clones / "sdk/lib/wine/d3d9/mtld3d/aarch64-windows/d3d9.dll"),
                         "aarch64-windows")

    def test_a_flat_x86_64_wine_gets_the_build_in_its_default_dirs(self):
        for directory in ("i386-windows", "x86_64-windows", "x86_64-unix"):
            (self.x86_64 / "lib/wine/d3d9/mtld3d" / directory).rmdir()
        (self.x86_64 / "lib/wine/d3d9/mtld3d").rmdir()
        (self.x86_64 / "lib/wine/d3d9").rmdir()
        before = self.directories(self.x86_64)
        self.install(f"WINE_INSTALL_DIR={self.x86_64}")
        self.assertEqual(self.directories(self.x86_64), before)
        self.assertEqual(self.content(self.x86_64 / "lib/wine/x86_64-windows/d3d9.dll"), "x86_64-windows")
        self.assertEqual(self.content(self.x86_64 / "lib/wine/i386-windows/mtld3d.dll"), "i386-windows")
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
        self.assertEqual(self.content(tree / "x86_64-windows/d3d9.dll"), "x86_64-windows")
        self.assertEqual(self.content(tree / "i386-windows/d3d9.dll"), "i386-windows")
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
        self.assertEqual(self.content(tree / "aarch64-windows/d3d9.dll"), "x86_64-windows")
        self.assertEqual(self.content(tree / "aarch64-windows/mtld3d.dll"), "x86_64-windows")
        self.assertEqual(self.content(tree / "i386-windows/d3d9.dll"), "i386-windows")
        self.assertEqual((tree / "aarch64-unix/mtld3d.so").read_text(), "aarch64-unix")
        for name in ("d3d9", "mtld3d"):
            marker = self.content(self.arm64 / f"lib/wine/aarch64-windows/{name}.dll")
            self.assertIn("-b x86_64-windows", marker)
            self.assertNotIn("marm64x", marker)
        self.assertEqual(self.content(self.x86_64 / "lib/wine/d3d9/mtld3d/x86_64-windows/d3d9.dll"),
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
        self.assertEqual(self.content(tree / "aarch64-windows/d3d9.dll"), "x86_64-windows")
        self.assertEqual(self.content(tree / "i386-windows/d3d9.dll"), "i386-windows")
        self.assertEqual((tree / "aarch64-unix/mtld3d.so").read_text(), "aarch64-unix")
        for name in ("d3d9", "mtld3d"):
            marker = self.content(clone / f"lib/wine/aarch64-windows/{name}.dll")
            self.assertIn("-b x86_64-windows", marker)
            self.assertNotIn("marm64x", marker)
        self.assertIn(f"--wine {clone}/bin/wine", output)

    def test_the_native_arm64_leg_runs_the_aarch64_binaries_in_a_tree_of_its_own(self):
        # The ARM64X build in a clone with a prefix booted after the install,
        # so the prefix's system32 takes the hybrid placeholders a native
        # ARM64 process needs.
        before = sorted(str(path) for path in self.arm64.rglob("*"))
        (self.stage / "tests/aarch64").mkdir(parents=True)
        (self.stage / "tests/aarch64/e2e-fixture.exe").touch()
        runner = self.stage / "e2e" / os.uname().machine / "mtld3d-e2e"
        runner.parent.mkdir(parents=True)
        runner.write_text("#!/bin/sh\nprintf 'runner:%s\\n' \"$*\"\n")
        runner.chmod(0o755)
        output = self.run_make(f"STAGE={self.stage}", "LEG_SUBMAKE=true", "test-e2e-arm64")
        self.assertEqual(sorted(str(path) for path in self.arm64.rglob("*")), before)
        root = self.work / ".wine-isolated/leg-arm64"
        tree = root / "sdk/lib/wine/d3d9/mtld3d"
        self.assertEqual(self.content(tree / "aarch64-windows/d3d9.dll"), "aarch64-windows")
        self.assertEqual(self.content(tree / "i386-windows/d3d9.dll"), "i386-windows")
        self.assertEqual((tree / "aarch64-unix/mtld3d.so").read_text(), "aarch64-unix")
        for name in ("d3d9", "mtld3d"):
            marker = root / f"sdk/lib/wine/aarch64-windows/{name}.dll"
            self.assertIn("-b aarch64-windows -marm64x", self.content(marker))
        runs = self.printed(output, "runner:")
        self.assertEqual(len(runs), 1, output)
        self.assertIn(f"--wine {root}/sdk/bin/wine", runs[0])
        self.assertIn(f"{self.stage}/tests/aarch64/e2e-fixture.exe", runs[0])


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
