"""List video files in one folder. No subfolders."""

import json
import subprocess
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


def list_videos(folder):
    found = []
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in VIDEO_EXT:
            continue
        if path.stem.endswith("-blur"):
            continue
        found.append(path)
    return found


def duration_seconds(path):
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    value = json.loads(raw).get("format", {}).get("duration")
    if value is None:
        return 0.0
    return float(value)


def format_duration(seconds):
    whole = int(round(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def format_size(nbytes):
    return f"{nbytes / 1048576:.0f}M"


def probe_times(duration, step=3.0):
    """Times to scan for a face. Not a random sample of the whole file."""
    if duration <= 0:
        return [0.0]
    times = []
    stamp = min(1.0, duration / 2)
    while stamp < duration - 0.05:
        times.append(stamp)
        stamp += step
    if not times:
        times.append(0.0)
    return times


def spread_times(hits, count=4):
    """Keep up to `count` hit times, spread from the first face to the last."""
    if len(hits) <= count:
        return list(hits)
    if count <= 1:
        return [hits[0]]
    last = len(hits) - 1
    picked = []
    for index in range(count):
        stamp = hits[round(index * last / (count - 1))]
        if not picked or stamp != picked[-1]:
            picked.append(stamp)
    return picked
