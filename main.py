"""Pick a video, look at stills that contain a face, then blur the whole file."""

import sys
from pathlib import Path

import numpy as np

from encode import output_path, process
from menu import choose, wait_action
from presets import BY_NAME, LEVELS, load_current, save_choice
from preview import make_preview, open_folder
from scan import duration_seconds, format_duration, format_size, list_videos
from weights import weight_path


def load_detector():
    from yolo_face import YoloFace

    model = weight_path()
    try:
        detector = YoloFace(conf=0.32, size=1280, provider="cuda", model=model)
        detector.detect(np.zeros((360, 640, 3), np.uint8))
        if detector.provider != "CUDAExecutionProvider":
            raise RuntimeError(detector.provider)
        print("Detection: GPU")
        return detector
    except Exception:
        print("CUDA is not available. Detection will run on CPU.")
        return YoloFace(conf=0.32, size=1280, provider="cpu", model=model)


def _level_labels(current_name):
    labels = []
    for item in LEVELS:
        note = ""
        if item.name == "medium":
            note = "  (checked night blur)"
        if item.name == current_name:
            note += "  current"
        labels.append(item.name + note)
    return labels


def _pick_level(current_name):
    selected = 0
    if current_name in BY_NAME:
        selected = [item.name for item in LEVELS].index(current_name)
    index = choose(
        "Blur level",
        _level_labels(current_name),
        selected=selected,
        hint="up/down, enter. q quits.",
    )
    if index is None:
        return None
    return LEVELS[index]


def _file_labels(videos):
    labels = []
    for path in videos:
        seconds = duration_seconds(path)
        labels.append(
            f"{path.name}    {format_duration(seconds)}    {format_size(path.stat().st_size)}"
        )
    return labels


def _preview_then_run(src, preset, folder, detector):
    while True:
        print(f"Preview with {preset.name}...", flush=True)
        stills = make_preview(src, detector, preset, duration_seconds(src))
        open_folder(stills)
        print(f"Preview: {stills}")
        print("enter runs the whole video. n picks another level. q quits.")
        action = wait_action()
        if action == "q":
            return
        if action == "n":
            picked = _pick_level(preset.name)
            if picked is None:
                return
            preset = picked
            continue
        saved = save_choice(folder, preset)
        dst = output_path(src)
        print(f"Preset: {saved.name}. Output: {dst}")
        process(src, dst, detector, preset)
        return


def interactive(folder):
    videos = list_videos(folder)
    if not videos:
        raise SystemExit("No video files in this folder.")
    index = choose("night-face-blur", _file_labels(videos))
    if index is None:
        return
    src = videos[index]
    current = load_current(folder)
    if current is None:
        preset = _pick_level("medium")
        if preset is None:
            return
    else:
        preset = current
        print(f"Using {preset.name} from conf/{(folder / 'conf' / 'current').name}.")
    detector = load_detector()
    _preview_then_run(src, preset, folder, detector)


def _parse_args(argv):
    level = None
    files = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--level":
            if index + 1 >= len(argv):
                raise SystemExit("--level needs soft, medium, or hard")
            level = argv[index + 1]
            if level not in BY_NAME:
                raise SystemExit(f"Unknown level: {level}")
            index += 2
            continue
        files.append(Path(arg))
        index += 1
    return level, files


def run_files(files, level_name, folder):
    if level_name:
        preset = BY_NAME[level_name]
    else:
        preset = load_current(folder) or BY_NAME["medium"]
    save_choice(folder, preset)
    detector = load_detector()
    for src in files:
        if not src.is_file():
            raise SystemExit(f"Not a file: {src}")
        process(src, output_path(src), detector, preset)


def main():
    folder = Path.cwd()
    level_name, files = _parse_args(sys.argv[1:])
    if files:
        run_files(files, level_name, folder)
        return
    if level_name:
        raise SystemExit("--level needs a video file")
    interactive(folder)


if __name__ == "__main__":
    main()
