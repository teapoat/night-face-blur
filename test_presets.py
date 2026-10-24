"""Preset files and the video list. No model and no video card."""

import tempfile
import unittest
from pathlib import Path

from presets import BY_NAME, load_current, save_choice
from scan import list_videos, probe_times, spread_times


class PresetTests(unittest.TestCase):
    def test_same_level_is_not_written_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            first = save_choice(folder, BY_NAME["medium"])
            second = save_choice(folder, BY_NAME["medium"])
            self.assertEqual(first, second)
            self.assertEqual(load_current(folder).name, "medium")
            self.assertEqual(list(Path(tmp, "conf").glob("*.txt")), [first])

    def test_changed_level_keeps_the_old_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            save_choice(folder, BY_NAME["medium"])
            save_choice(folder, BY_NAME["soft"])
            again = save_choice(folder, BY_NAME["soft"])
            names = sorted(path.name for path in Path(tmp, "conf").glob("*.txt"))
            self.assertEqual(names, ["medium.txt", "soft.txt"])
            self.assertEqual(again.name, "soft.txt")
            self.assertEqual(load_current(folder).name, "soft")

    def test_same_name_new_numbers_gets_a_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            save_choice(folder, BY_NAME["soft"])
            other = BY_NAME["soft"].__class__(
                "soft", pad=0.2, kernel_frac=0.2, passes=1, feather=9
            )
            path = save_choice(folder, other)
            self.assertEqual(path.name, "soft-2.txt")
            self.assertEqual(load_current(folder).pad, 0.2)

    def test_list_skips_finished_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "clip.mp4").write_bytes(b"")
            (folder / "clip-blur.mp4").write_bytes(b"")
            (folder / "note.txt").write_bytes(b"")
            (folder / "nested").mkdir()
            (folder / "nested" / "inner.mp4").write_bytes(b"")
            names = [path.name for path in list_videos(folder)]
            self.assertEqual(names, ["clip.mp4"])


class ScanTests(unittest.TestCase):
    def test_probe_times_cover_the_file(self):
        times = probe_times(20, step=3)
        self.assertGreaterEqual(times[0], 0)
        self.assertLess(times[-1], 20)
        self.assertGreaterEqual(len(times), 6)

    def test_spread_keeps_first_and_last_face(self):
        hits = [2, 5, 9, 12, 18, 21, 30]
        self.assertEqual(spread_times(hits, 4), [2, 9, 18, 30])

    def test_spread_returns_short_lists(self):
        self.assertEqual(spread_times([4, 8], 4), [4, 8])
        self.assertEqual(spread_times([], 4), [])


if __name__ == "__main__":
    unittest.main()
