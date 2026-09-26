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
