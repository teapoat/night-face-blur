"""Locate the face model. Download it when the local file is missing."""

import hashlib
import urllib.request
from pathlib import Path

WEIGHT_NAME = "yolov8n-face.onnx"
SHA256 = "33f3951af7fc0c4d9b321b29cdcd8c9a59d0a29a8d4bdc01fcb5507d5c714809"
URL = "https://github.com/teapoat/night-face-blur/releases/download/weights/yolov8n-face.onnx"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matching(path):
    return path.is_file() and _sha256(path) == SHA256


def weight_path():
    root = Path(__file__).resolve().parent
    for path in (root / WEIGHT_NAME, root / "weights" / WEIGHT_NAME):
        if _matching(path):
            return path
    dest = root / "weights" / WEIGHT_NAME
    dest.parent.mkdir(exist_ok=True)
    try:
        urllib.request.urlretrieve(URL, dest)
    except OSError as exc:
        raise SystemExit(
            f"Could not download the face model from {URL} ({exc}). "
            f"Place {WEIGHT_NAME} next to the script."
        ) from exc
    if not _matching(dest):
        dest.unlink(missing_ok=True)
        raise SystemExit(f"Downloaded model hash does not match. Removed {dest}.")
    return dest
