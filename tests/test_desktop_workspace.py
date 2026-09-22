"""Tests for KDE virtual-desktop pin helpers."""
from __future__ import annotations

import unittest

from ascent_player.env.desktop_workspace import (
    build_kwin_pin_script,
    parse_kwinrc_desktops,
    workspace_name,
)


class DesktopWorkspaceTests(unittest.TestCase):
    def test_parse_kwinrc_desktops(self) -> None:
        text = (
            "[Desktops]\n"
            "Id_1=aaa\n"
            "Id_2=bbb\n"
            "Name_1=Main\n"
            "Name_2=Ascent\n"
            "Number=2\n"
            "\n"
            "[NightColor]\n"
            "Active=true\n"
        )
        desks = parse_kwinrc_desktops(text)
        self.assertEqual(desks, [("aaa", "Main"), ("bbb", "Ascent")])

    def test_script_escapes_and_matches(self) -> None:
        js = build_kwin_pin_script(
            desktop_name="Ascent",
            pids=(63849, 64280),
            title_prefixes=("Ascent — v2 climb",),
            resource_hints=("playwright_chromiumdev_profile",),
        )
        self.assertIn('"desktop": "Ascent"', js)
        self.assertIn("63849", js)
        self.assertIn("playwright_chromiumdev_profile", js)
        self.assertIn("windowAdded", js)
        self.assertNotIn("Cursor", js)

    def test_workspace_name_override_and_disable(self) -> None:
        self.assertEqual(workspace_name("Ascent"), "Ascent")
        self.assertEqual(workspace_name("  "), "")
        self.assertEqual(workspace_name(""), "")


if __name__ == "__main__":
    unittest.main()
