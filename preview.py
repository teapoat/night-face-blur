"""Blurred stills of frames that actually contain a face. Not a full encode."""

import subprocess
from pathlib import Path

import cv2

from blur_ops import apply_blur
from scan import format_duration, probe_times, spread_times


def preview_dir(src):
    return Path(src).with_name(f"{Path(src).stem}_preview")


def _stamp_name(index, seconds):
    whole = int(round(seconds))
    return f"{index:02d}-{whole // 60:02d}m{whole % 60:02d}s.jpg"


def _grab(src, folder, stamp):
    raw = folder / "_probe.jpg"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{stamp:.3f}",
            "-i",
            str(src),
            "-frames:v",
            "1",
            str(raw),
        ]
    )
    frame = cv2.imread(str(raw))
    raw.unlink(missing_ok=True)
    return frame


def _blur(frame, dets, preset):
    for det in dets:
        apply_blur(
            frame,
            det,
            pad=preset.pad,
            kernel_frac=preset.kernel_frac,
            passes=preset.passes,
            feather=preset.feather,
        )


def make_preview(src, detector, preset, duration):
    """Check one frame every 3s. Save up to four where a face was found.

    The stills are seconds apart, so the video tracker is not used here.
    """
    folder = preview_dir(src)
    folder.mkdir(exist_ok=True)
    for old in folder.glob("*.jpg"):
        old.unlink()
    print("Looking for frames with a face, one every 3s...", flush=True)
    hits = []
    times = probe_times(duration)
    for index, stamp in enumerate(times, start=1):
        frame = _grab(src, folder, stamp)
        if frame is None:
            continue
        if detector.detect(frame):
            hits.append(stamp)
        if index % 5 == 0 or index == len(times):
            print(
                f"  checked {format_duration(stamp)}, frames with a face {len(hits)}",
                flush=True,
            )
    chosen = spread_times(hits, 4)
    if not chosen:
        print("No face in the frames I checked.")
        return folder
    for index, stamp in enumerate(chosen, start=1):
        frame = _grab(src, folder, stamp)
        if frame is None:
            continue
        dets = detector.detect(frame)
        if not dets:
            continue
        _blur(frame, dets, preset)
        name = _stamp_name(index, stamp)
        cv2.imwrite(
            str(folder / name),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        print(f"  {name}  {len(dets)} faces", flush=True)
    return folder


def open_folder(folder):
    subprocess.Popen(
        ["xdg-open", str(folder)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
