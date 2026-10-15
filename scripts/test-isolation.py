#!/usr/bin/env python3
"""Check production Makefile path resolution without building or installing."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


MAKEFILE = Path(__file__).resolve().parent.parent / "Makefile"
FIELDS = ("WINE_X86_64", "WINE_SDK", "WINE_INSTALL_DIR", "WINEPREFIX", "WINE",
          "WINEBUILD", "WINESERVER", "INSTALL_DIRS", "TEST_PREFIX")


class MakeTestCase(unittest.TestCase):
    """The Makefile included by a wrapper, run in a scratch tree with its Wine inputs faked."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mtld3d-isolation-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.work = self.root / "work"
        self.work.mkdir()
        self.sources = {
            "WINE_X86_64": str(self.root / "sdk-source"),
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
            ".PHONY: inspect recurse\n"
            "inspect:\n"
            "\t@printf '%s\\n' "
            + " ".join(f"'resolved:{name}=$({name})'" for name in FIELDS)
            + "\nrecurse: inspect\n"
            "\t+$(MAKE) --no-print-directory -f $(firstword $(MAKEFILE_LIST)) inspect\n"
        )
        self.environment = os.environ.copy()
        for name in ("MAKEFLAGS", "MFLAGS", "GNUMAKEFLAGS", "MAKEOVERRIDES",
                     "MAKELEVEL", "ISOLATED", "EC", "ARM64", "WINE_ARM64", "STAGE",
                     "WINE_SDK", "WINE_SDK_ARM64X", *self.sources):
            self.environment.pop(name, None)
        self.environment.update(self.sources)

    def run_make(self, *arguments, succeeds=True):
        isolation_root = self.work / ".wine-isolated"
        result = subprocess.run(
            ["make", "--no-print-directory", "-f", str(self.wrapper),
             f"ISOLATED_ROOT={isolation_root}", "ISOLATED_REGISTRY=",
             "BUILD_ID=isolation-test", "SDK_UNIX_ARCH=x64", *arguments],
            cwd=MAKEFILE.parent, env=self.environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
        )
        self.assertEqual(result.returncode == 0, succeeds, result.stdout)
        return result.stdout


class IsolationTests(MakeTestCase):
    def assert_paths(self, output, isolated, copies=1):
        sdk = str(self.work / ".wine-isolated/sdk") if isolated else self.sources["WINE_X86_64"]
        prefix = str(self.work / ".wine-isolated/prefix") if isolated else self.sources["WINEPREFIX"]
        install = sdk if isolated else self.sources["WINE_INSTALL_DIR"]
        expected = dict(WINE_X86_64=sdk, WINE_SDK=sdk, WINE_INSTALL_DIR=install, WINEPREFIX=prefix,
                        WINE=f"{sdk}/bin/wine", WINEBUILD=f"{sdk}/bin/winebuild",
                        WINESERVER=f"{sdk}/bin/wineserver", TEST_PREFIX=prefix,
                        INSTALL_DIRS=" ".join(sorted({sdk, install})))
        actual = [line for line in output.splitlines() if line.startswith("resolved:")]
        self.assertEqual(actual, [f"resolved:{name}={expected[name]}" for name in FIELDS] * copies,
                         output)
        if isolated:
            banner = (f"==> ISOLATED=1: WINE_X86_64={sdk} WINE_INSTALL_DIR={install} "
                      f"WINEPREFIX={prefix}")
            actual_banners = [line for line in output.splitlines()
                              if line.startswith("==> ISOLATED=1:")]
            self.assertEqual(actual_banners, [banner] * copies, output)
            for directory, source in (("sdk", "WINE_X86_64"), ("prefix", "WINEPREFIX")):
                self.assertEqual((self.work / ".wine-isolated" / directory / "sentinel").read_text(),
                                 source)
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


class WineChoiceTests(MakeTestCase):
    """Which Wine the builds, the installs and the bench-ab legs take, and the refusals."""

    def setUp(self):
        super().setUp()
        self.arm64 = self.root / "arm64"
        (self.arm64 / "bin").mkdir(parents=True)
        (self.arm64 / "bin" / "wine").touch()

    def arm64_alone(self):
        for name in ("WINE_X86_64", "WINE_INSTALL_DIR"):
            self.environment.pop(name)
        self.environment["WINE_ARM64"] = str(self.arm64)

    def resolved(self, output):
        return dict(line[len("resolved:"):].split("=", 1) for line in output.splitlines()
                    if line.startswith("resolved:"))

    def test_the_x86_64_wine_is_the_sdk_when_both_are_set(self):
        values = self.resolved(self.run_make(f"WINE_ARM64={self.arm64}", "inspect"))
        self.assertEqual(values["WINE_SDK"], self.sources["WINE_X86_64"])
        self.assertEqual(values["WINEBUILD"], f"{self.sources['WINE_X86_64']}/bin/winebuild")

    def test_the_arm64_wine_alone_serves_the_build(self):
        self.arm64_alone()
        values = self.resolved(self.run_make("inspect"))
        self.assertEqual(values["WINE_SDK"], str(self.arm64))
        self.assertEqual(values["WINEBUILD"], f"{self.arm64}/bin/winebuild")
        self.assertEqual(values["INSTALL_DIRS"], "")

    def test_isolation_with_the_arm64_wine_alone_clones_no_x86_64_tree(self):
        self.arm64_alone()
        values = self.resolved(self.run_make("ISOLATED=1", "inspect"))
        self.assertEqual(values["WINE_X86_64"], "")
        self.assertEqual(values["WINE_INSTALL_DIR"], "")
        self.assertEqual(values["INSTALL_DIRS"], "")
        self.assertEqual(values["WINE_SDK"], str(self.arm64))
        self.assertFalse((self.work / ".wine-isolated/sdk").exists())
        values = self.resolved(self.run_make("ISOLATED=1", "ARM64=1", "inspect"))
        self.assertEqual(values["WINE_SDK"], str(self.work / ".wine-isolated/sdk-arm64"))
        self.assertTrue((self.work / ".wine-isolated/sdk-arm64/bin/wine").exists())

    def test_a_wine_sdk_from_the_command_line_is_ignored(self):
        values = self.resolved(self.run_make(f"WINE_SDK={self.root / 'elsewhere'}", "inspect"))
        self.assertEqual(values["WINE_SDK"], self.sources["WINE_X86_64"])

    def test_an_old_variable_in_the_environment_stops_the_run_without_wine_x86_64(self):
        self.arm64_alone()
        for name in ("WINE_SDK", "WINE_SDK_ARM64X"):
            with self.subTest(name=name):
                self.environment[name] = str(self.root / "old")
                output = self.run_make("inspect", succeeds=False)
                self.assertIn("WINE_SDK and WINE_SDK_ARM64X are no longer read", output)
                self.assertIn("set WINE_X86_64", output)
                del self.environment[name]

    def test_an_old_variable_beside_wine_x86_64_is_ignored(self):
        self.environment["WINE_SDK"] = str(self.root / "old")
        values = self.resolved(self.run_make("inspect"))
        self.assertEqual(values["WINE_SDK"], self.sources["WINE_X86_64"])

    def test_without_either_wine_the_makefile_names_both(self):
        self.environment.pop("WINE_X86_64")
        output = self.run_make("inspect", succeeds=False)
        self.assertIn("neither WINE_X86_64 nor WINE_ARM64 is set", output)

    def test_the_exported_sdk_does_not_trip_the_old_variable_check_in_a_sub_make(self):
        # The Makefile exports WINE_SDK, so a sub-make sees it in its environment
        # while WINE_X86_64 is unset; the check holds only at the top level.
        self.arm64_alone()
        output = self.run_make("recurse")
        sdks = [line for line in output.splitlines() if line.startswith("resolved:WINE_SDK=")]
        self.assertEqual(sdks, [f"resolved:WINE_SDK={self.arm64}"] * 2, output)

    def test_an_x86_install_leaf_without_a_tree_refuses(self):
        self.arm64_alone()
        output = self.run_make(f"STAGE={self.root / 'stage'}", "install-unix-x64", succeeds=False)
        self.assertIn("no x86_64 Wine tree to install into", output)

    def test_a_build_against_a_runtime_only_wine_refuses(self):
        for target, arch in (("windows-i686", "i386-windows"), ("clippy-pe-x86_64", "x86_64-windows")):
            with self.subTest(target=target):
                output = self.run_make(target, succeeds=False)
                self.assertIn(f"WINE_X86_64={self.sources['WINE_X86_64']} is no Wine SDK for {arch}",
                              output)

    def test_bench_ab_legs_name_both_variable_sets(self):
        wrapper = self.root / "show.mk"
        wrapper.write_text(f"include {MAKEFILE}\nshow:\n\t@printf '%s\\n' \"$(BENCH_LEG_MAKE)\"\n")
        output = subprocess.run(
            ["make", "--no-print-directory", "-f", str(wrapper), f"WINE_ARM64={self.arm64}", "show"],
            cwd=MAKEFILE.parent, env=self.environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30, check=True,
        ).stdout
        x86 = self.sources["WINE_X86_64"]
        for assignment in (f"WINE_X86_64='{x86}'", f"WINE_ARM64='{self.arm64}'",
                           f"WINE_SDK='{x86}'", f"WINE_SDK_ARM64X='{self.arm64}'"):
            self.assertIn(assignment, output)


class InstallLayoutTests(MakeTestCase):
    """`make install` against two fake Wines: what goes where, and no arch directory created."""

    X86_64_DIRS = ("bin", "lib/wine/x86_64-unix", "lib/wine/i386-windows", "lib/wine/x86_64-windows",
                   "lib/wine/d3d9/mtld3d/i386-windows", "lib/wine/d3d9/mtld3d/x86_64-windows",
                   "lib/wine/d3d9/mtld3d/x86_64-unix")
    ARM64_DIRS = ("bin", "lib/wine/aarch64-unix", "lib/wine/i386-windows", "lib/wine/aarch64-windows",
                  "lib/wine/x86_64-windows", "lib/wine/d3d9/mtld3d/i386-windows",
                  "lib/wine/d3d9/mtld3d/aarch64-windows", "lib/wine/d3d9/mtld3d/aarch64-unix")

    def setUp(self):
        super().setUp()
        self.x86_64 = self.make_wine("x86_64", self.X86_64_DIRS)
        self.arm64 = self.make_wine("arm64", self.ARM64_DIRS)
        (self.arm64 / "lib/wine/aarch64-unix/compatdb.so").touch()
        # winebuild writes its arguments into the file `-o` names and leaves
        # everything else as it is.
        winebuild = self.x86_64 / "bin/winebuild"
        winebuild.write_text("#!/bin/sh\nall=\"$*\"\n"
                             "while [ $# -gt 0 ]; do [ \"$1\" = -o ] && printf '%s\\n' \"$all\" > \"$2\"; shift; done\n")
        winebuild.chmod(0o755)
        self.arm64x = self.root / "arm64x"
        self.arm64x.mkdir()
        for name in ("d3d9.dll", "d3d9.pdb", "mtld3d.dll", "mtld3d.pdb"):
            (self.arm64x / name).write_text("arm64x")
        self.stage = self.root / "stage"
        for arch in ("i386-windows", "x86_64-windows"):
            (self.stage / arch).mkdir(parents=True)
            for name in ("d3d9.dll", "d3d9.pdb", "mtld3d.dll", "mtld3d.pdb"):
                (self.stage / arch / name).write_text(arch)
        for arch in ("x86_64-unix", "aarch64-unix"):
            (self.stage / arch / "mtld3d.so.dSYM").mkdir(parents=True)
            (self.stage / arch / "mtld3d.so").write_text(arch)
        self.environment.pop("WINE_INSTALL_DIR")
        self.environment["WINE_X86_64"] = str(self.x86_64)

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

    def test_each_wine_gets_its_own_pieces_and_no_new_arch_directory(self):
        before = {wine: self.directories(wine) for wine in (self.x86_64, self.arm64)}
        self.install(f"WINE_ARM64={self.arm64}")
        for wine in (self.x86_64, self.arm64):
            self.assertEqual(self.directories(wine), before[wine])
        tree = self.x86_64 / "lib/wine/d3d9/mtld3d"
        self.assertEqual((tree / "x86_64-windows/d3d9.dll").read_text(), "x86_64-windows")
        self.assertEqual((tree / "i386-windows/d3d9.dll").read_text(), "i386-windows")
        self.assertEqual((tree / "x86_64-unix/mtld3d.so").read_text(), "x86_64-unix")
        self.assertTrue((self.x86_64 / "lib/wine/x86_64-windows/d3d9.dll").exists())
        tree = self.arm64 / "lib/wine/d3d9/mtld3d"
        self.assertEqual((tree / "aarch64-windows/d3d9.dll").read_text(), "x86_64-windows")
        self.assertEqual((tree / "i386-windows/mtld3d.dll").read_text(), "i386-windows")
        self.assertEqual((tree / "aarch64-unix/mtld3d.so").read_text(), "aarch64-unix")
        self.assertTrue((self.arm64 / "lib/wine/aarch64-windows/d3d9.dll").exists())
        self.assertFalse((self.arm64 / "lib/wine/x86_64-windows/d3d9.dll").exists())

    def test_the_x86_64_wine_alone_is_installed_without_the_arm64_pieces(self):
        before = self.directories(self.x86_64)
        self.install()
        self.assertEqual(self.directories(self.x86_64), before)
        self.assertFalse(list(self.x86_64.rglob("aarch64-*")))

    def test_a_missing_arch_directory_fails_the_install_and_is_not_created(self):
        missing = self.x86_64 / "lib/wine/d3d9/mtld3d/x86_64-windows"
        missing.rmdir()
        output = self.install(succeeds=False)
        self.assertIn(f"{missing} does not exist", output)
        self.assertFalse(missing.exists())

    def test_the_64_bit_markers_are_x64_ones_in_both_wines(self):
        self.install(f"WINE_ARM64={self.arm64}")
        for marker in (self.x86_64 / "lib/wine/x86_64-windows/d3d9.dll",
                       self.arm64 / "lib/wine/aarch64-windows/d3d9.dll",
                       self.arm64 / "lib/wine/aarch64-windows/mtld3d.dll"):
            self.assertIn("-b x86_64-windows", marker.read_text())
        self.assertIn("-b i686-windows", (self.arm64 / "lib/wine/i386-windows/d3d9.dll").read_text())

    def test_ec_puts_the_arm64x_pair_into_aarch64_windows_behind_x64_markers(self):
        self.install(f"WINE_ARM64={self.arm64}", "EC=1", f"OUT_arm64x={self.arm64x}")
        tree = self.arm64 / "lib/wine/d3d9/mtld3d"
        self.assertEqual((tree / "aarch64-windows/d3d9.dll").read_text(), "arm64x")
        self.assertEqual((tree / "aarch64-windows/mtld3d.dll").read_text(), "arm64x")
        self.assertEqual((tree / "i386-windows/d3d9.dll").read_text(), "i386-windows")
        self.assertIn("-b x86_64-windows", (self.arm64 / "lib/wine/aarch64-windows/d3d9.dll").read_text())

    def test_an_isolated_install_writes_the_clones_and_not_the_wines(self):
        before = {wine: sorted(str(path) for path in wine.rglob("*")) for wine in (self.x86_64, self.arm64)}
        self.install("ISOLATED=1", f"WINE_ARM64={self.arm64}")
        for wine in (self.x86_64, self.arm64):
            self.assertEqual(sorted(str(path) for path in wine.rglob("*")), before[wine])
        clones = self.work / ".wine-isolated"
        self.assertEqual((clones / "sdk/lib/wine/d3d9/mtld3d/x86_64-windows/d3d9.dll").read_text(),
                         "x86_64-windows")
        self.assertEqual((clones / "sdk-arm64/lib/wine/d3d9/mtld3d/aarch64-windows/d3d9.dll").read_text(),
                         "x86_64-windows")

    def test_a_flat_x86_64_wine_gets_the_build_in_its_default_dirs(self):
        for directory in ("i386-windows", "x86_64-windows", "x86_64-unix"):
            (self.x86_64 / "lib/wine/d3d9/mtld3d" / directory).rmdir()
        (self.x86_64 / "lib/wine/d3d9/mtld3d").rmdir()
        (self.x86_64 / "lib/wine/d3d9").rmdir()
        before = self.directories(self.x86_64)
        self.install()
        self.assertEqual(self.directories(self.x86_64), before)
        self.assertEqual((self.x86_64 / "lib/wine/x86_64-windows/d3d9.dll").read_text(), "x86_64-windows")
        self.assertEqual((self.x86_64 / "lib/wine/i386-windows/mtld3d.dll").read_text(), "i386-windows")
        self.assertEqual((self.x86_64 / "lib/wine/x86_64-unix/mtld3d.so").read_text(), "x86_64-unix")

    def test_a_wine_with_both_unix_dirs_is_refused_naming_the_leftover(self):
        (self.x86_64 / "lib/wine/aarch64-unix").mkdir()
        output = self.install(succeeds=False)
        self.assertIn("has both lib/wine/x86_64-unix and lib/wine/aarch64-unix", output)
        self.assertIn("can be removed", output)

    def test_an_install_dir_naming_the_arm64_wine_is_written_once(self):
        output = self.run_make(f"WINE_ARM64={self.arm64}", f"WINE_INSTALL_DIR={self.arm64}", "inspect")
        self.assertIn(f"resolved:INSTALL_DIRS={self.x86_64}\n", output)

    def test_an_arm64_wine_without_the_mtld3d_tree_is_refused(self):
        for directory in ("i386-windows", "aarch64-windows", "aarch64-unix"):
            (self.arm64 / "lib/wine/d3d9/mtld3d" / directory).rmdir()
        (self.arm64 / "lib/wine/d3d9/mtld3d").rmdir()
        output = self.run_make(f"STAGE={self.stage}", f"WINE_ARM64={self.arm64}", "install-arm64",
                               succeeds=False)
        self.assertIn("no lib/wine/d3d9/mtld3d tree", output)
        self.assertFalse((self.arm64 / "lib/wine/d3d9").joinpath("mtld3d").exists())


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
        output = self.run_make(f"WINESERVER={wineserver}", f"WINE={wine}", "MAKE=echo sub-make",
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
        return [f"WINESERVER={wineserver}", f"WINE={wine}"]

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
