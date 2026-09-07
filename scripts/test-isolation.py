#!/usr/bin/env python3
"""Check production Makefile path resolution without building or installing."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


MAKEFILE = Path(__file__).resolve().parent.parent / "Makefile"
FIELDS = ("WINE_SDK", "WINE_INSTALL_DIR", "WINEPREFIX", "WINE",
          "WINEBUILD", "WINESERVER", "INSTALL_DIRS", "TEST_PREFIX")


class IsolationTests(unittest.TestCase):
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
            ".PHONY: inspect recurse\n"
            "inspect:\n"
            "\t@printf '%s\\n' "
            + " ".join(f"'resolved:{name}=$({name})'" for name in FIELDS)
            + "\nrecurse: inspect\n"
            "\t+$(MAKE) --no-print-directory -f $(firstword $(MAKEFILE_LIST)) inspect\n"
        )
        self.environment = os.environ.copy()
        for name in ("MAKEFLAGS", "MFLAGS", "GNUMAKEFLAGS", "MAKEOVERRIDES",
                     "MAKELEVEL", "ISOLATED", *self.sources):
            self.environment.pop(name, None)
        self.environment.update(self.sources)

    def run_make(self, *arguments):
        isolation_root = self.work / ".wine-isolated"
        result = subprocess.run(
            ["make", "--no-print-directory", "-f", str(self.wrapper),
             f"ISOLATED_ROOT={isolation_root}", "ISOLATED_REGISTRY=",
             "BUILD_ID=isolation-test", "SDK_UNIX_ARCH=x64", *arguments],
            cwd=MAKEFILE.parent, env=self.environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def assert_paths(self, output, isolated, copies=1):
        sdk = str(self.work / ".wine-isolated/sdk") if isolated else self.sources["WINE_SDK"]
        prefix = str(self.work / ".wine-isolated/prefix") if isolated else self.sources["WINEPREFIX"]
        install = sdk if isolated else self.sources["WINE_INSTALL_DIR"]
        expected = dict(WINE_SDK=sdk, WINE_INSTALL_DIR=install, WINEPREFIX=prefix,
                        WINE=f"{sdk}/bin/wine", WINEBUILD=f"{sdk}/bin/winebuild",
                        WINESERVER=f"{sdk}/bin/wineserver", TEST_PREFIX=prefix,
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


if __name__ == "__main__":
    unittest.main()
