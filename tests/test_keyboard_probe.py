from __future__ import annotations

import unittest

from ascent_player.demo.keyboard_probe import (
    _INSTALL_LISTENERS_JS,
    install_keyboard_probe,
)


class _FakeContext:
    def __init__(self) -> None:
        self.init_scripts: list[str] = []

    async def add_init_script(self, script: str) -> None:
        self.init_scripts.append(script)


class _FakePage:
    def __init__(self) -> None:
        self._context = _FakeContext()
        self.init_scripts: list[str] = []
        self.evaluations: list[str] = []

    def context(self) -> _FakeContext:
        return self._context

    async def add_init_script(self, script: str) -> None:
        self.init_scripts.append(script)

    async def evaluate(self, script: str) -> bool:
        self.evaluations.append(script)
        return True


class KeyboardProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_install_keyboard_probe_persists_across_navigations(self) -> None:
        page = _FakePage()

        await install_keyboard_probe(page)

        self.assertEqual(page.context().init_scripts, [_INSTALL_LISTENERS_JS])
        self.assertEqual(page.init_scripts, [_INSTALL_LISTENERS_JS])
        self.assertEqual(page.evaluations, [_INSTALL_LISTENERS_JS])


if __name__ == "__main__":
    unittest.main()
