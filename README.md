# night-face-blur

Blur faces in a night video.

3 blur sizes. The detector stays the same.
Soft: smaller spot. Medium: the night setting. Hard: wider spot.
Misses a face turned down, the back of a head, and license plates.

![Blurred face on a night street](card.jpg)

## Run

From the video folder. That folder only.

```
/path/to/night-face-blur
```

Text menu. Arrows, number keys, Enter.
Pick a file, then soft, medium, or hard.
One frame every 3 seconds. Up to 4 stills with a face, in `<name>_preview/`. The folder opens.
Enter runs the whole file. `n` picks another size. `q` quits.

Output: `<name>-blur`, next to the original. The original stays.
The next video starts from the saved size and still gets its own stills.
Saved size: `conf/current`. A new file only when the size changes (`medium.txt`, `soft.txt`, `soft-2.txt`).

No menu:

```
/path/to/night-face-blur --level medium video.mp4
```

## Install

ffmpeg. Python 3.14. Other versions were not tried.

```
python3 -m venv .venv
```

```
.venv/bin/pip install -r requirements.txt
```

NVIDIA is optional. Tested: RTX 3060 Ti, driver 580.
No CUDA: one line, then detection on CPU. Blur and encode stay on CPU.
Two short clips: CPU was about 3 times slower. A machine with no NVIDIA card was not tried.

First run downloads `yolov8n-face.onnx` (12 MB) from this repo's release, if the file is not next to the script. The hash is checked.

Weight source. No license file:
https://github.com/yakhyo/yolov8-face-onnx-inference

Training project. GPL-3.0:
https://github.com/derronqi/yolov8-face

This script is MIT. The weight is not in git.
