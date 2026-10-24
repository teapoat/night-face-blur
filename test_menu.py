"""Arrow keys must move the list, not quit. Runs in a pty."""

import os
import pty
import select
import unittest
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)


def _pick(keys):
    parent, child = pty.openpty()
    pid = os.fork()
    if pid == 0:
        os.close(parent)
        os.dup2(child, 0)
        os.dup2(child, 1)
        os.dup2(child, 2)
        os.close(child)
        import sys

        sys.path.insert(0, ROOT)
        from menu import choose

        pick = choose("title", ["one", "two"])
        print(f"PICK {pick}", flush=True)
        os._exit(0)
    os.close(child)
    data = b""
    while b"up/down" not in data:
        ready, _, _ = select.select([parent], [], [], 3)
        if not ready:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            raise AssertionError(data)
        data = data + os.read(parent, 4096)
    os.write(parent, keys)
    while b"PICK " not in data:
        ready, _, _ = select.select([parent], [], [], 3)
        if not ready:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            raise AssertionError(data)
        chunk = os.read(parent, 4096)
        if not chunk:
            break
        data = data + chunk
    os.close(parent)
    os.waitpid(pid, 0)
    return data


class MenuKeyTests(unittest.TestCase):
    def test_down_then_enter(self):
        self.assertIn(b"PICK 1", _pick(b"\x1b[B\r"))

    def test_up_wraps_then_enter(self):
        self.assertIn(b"PICK 1", _pick(b"\x1b[A\r"))


if __name__ == "__main__":
    unittest.main()
