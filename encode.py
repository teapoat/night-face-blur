"""Blur faces in one video. Audio is copied when the file has an audio track."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from blur_ops import apply_blur, update_tracks


def _as_int(value):
    if value is None or value == "N/A":
        return None
    return int(value)


def _as_float(value):
    if value is None or value == "N/A":
        return None
    return float(value)


def probe(src):
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(src),
        ]
    )
    data = json.loads(raw)
    streams = data["streams"]
    video = next(item for item in streams if item["codec_type"] == "video")
    audio = next((item for item in streams if item["codec_type"] == "audio"), None)
    width = int(video["width"])
    height = int(video["height"])
    rotation = 0
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(side["rotation"])
    tag = video.get("tags", {}).get("rotate")
    if tag is not None:
        rotation = int(tag)
    if abs(rotation) % 180 == 90:
        width, height = height, width
    frames = _as_int(video.get("nb_frames"))
    video_duration = _as_float(video.get("duration"))
    if video_duration is None:
        video_duration = _as_float(data.get("format", {}).get("duration"))
    if frames is None:
        rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        if not fps or not video_duration:
            raise SystemExit(f"cannot read frame count of {src.name}")
        frames = int(round(fps * video_duration))
    audio_duration = _as_float(audio.get("duration")) if audio else None
    clock = audio_duration or video_duration
    if not clock:
        raise SystemExit(f"cannot read duration of {src.name}")
    return {
        "width": width,
        "height": height,
        "frames": frames,
        "audio_duration": audio_duration,
        "audio_codec": audio.get("codec_name") if audio else None,
        "rotation": rotation,
        "fps": frames / clock,
        "has_audio": audio is not None and audio_duration is not None,
    }


def readable_frames(path):
    """Container nb_frames can be higher than the frames a decoder can emit."""
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    value = json.loads(raw)["streams"][0].get("nb_read_frames")
    if value is None or value == "N/A":
        raise SystemExit(f"cannot count frames in {path}")
    return int(value)


def output_path(src):
    src = Path(src)
    dst = src.with_name(f"{src.stem}-blur{src.suffix}")
    if dst.resolve() == src.resolve():
        raise SystemExit(f"refusing to overwrite {src}")
    return dst


def process(src, dst, detector, preset):
    src = Path(src)
    dst = Path(dst)
    info = probe(src)
    width, height = info["width"], info["height"]
    frame_bytes = width * height * 3
    audio_note = info["audio_codec"] if info["has_audio"] else "none"
    if not info["has_audio"]:
        print(f"{src.name} has no audio track. The output will be video only.")
    print(
        f"{src.name} -> {dst.name} {width}x{height} frames={info['frames']} "
        f"level={preset.name} audio={audio_note} rotation={info['rotation']}",
        flush=True,
    )
    dec_log_path = dst.with_suffix(".decode.log")
    enc_log_path = dst.with_suffix(".encode.log")
    dec_log = open(dec_log_path, "wb")
    enc_log = open(enc_log_path, "wb")
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=dec_log,
    )
    encode_cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{info['fps']:.8f}",
        "-i",
        "pipe:0",
    ]
    if info["has_audio"]:
        encode_cmd.extend(
            [
                "-i",
                str(src),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:a",
                "copy",
            ]
        )
    else:
        encode_cmd.extend(["-map", "0:v:0"])
    encode_cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )
    encoder = subprocess.Popen(
        encode_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=enc_log,
    )
    tracks = []
    index = 0
    started = time.perf_counter()
    try:
        while True:
            buf = bytearray()
            while len(buf) < frame_bytes:
                chunk = decoder.stdout.read(frame_bytes - len(buf))
                if not chunk:
                    break
                buf.extend(chunk)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(height, width, 3).copy()
            dets = detector.detect(frame)
            update_tracks(tracks, dets)
            for box in (track.box for track in tracks if track.miss <= 6):
                apply_blur(
                    frame,
                    box,
                    pad=preset.pad,
                    kernel_frac=preset.kernel_frac,
                    passes=preset.passes,
                    feather=preset.feather,
                )
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
            index += 1
            if index % 400 == 0:
                elapsed = time.perf_counter() - started
                left = (info["frames"] - index) * (elapsed / index)
                print(
                    f"  {src.name} {index}/{info['frames']} "
                    f"{elapsed:.0f}s left {left:.0f}s",
                    flush=True,
                )
    finally:
        encoder.stdin.close()
        decoder.stdout.close()
    dec_code = decoder.wait()
    enc_code = encoder.wait()
    dec_log.close()
    enc_log.close()
    elapsed = time.perf_counter() - started
    print(
        f"DONE {dst.name} frames={index} expected={info['frames']} "
        f"seconds={elapsed:.1f} bytes={dst.stat().st_size} "
        f"decode={dec_code} encode={enc_code}",
        flush=True,
    )
    mismatch = abs(index - info["frames"]) > 2
    if dec_code == 0 and enc_code == 0 and mismatch:
        counted = readable_frames(src)
        print(
            f"container frames={info['frames']} readable={counted} written={index}",
            flush=True,
        )
        mismatch = abs(index - counted) > 2
    if dec_code != 0 or enc_code != 0 or mismatch:
        print(dec_log_path.read_text(errors="replace")[-1500:])
        print(enc_log_path.read_text(errors="replace")[-1500:])
        raise SystemExit(1)
    dec_log_path.unlink(missing_ok=True)
    enc_log_path.unlink(missing_ok=True)
