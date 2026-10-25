# night-face-blur

Blurs faces in a night video.

![Blurred face on a night street](card.jpg)

Console program. You control it from the keyboard. Run it from the folder with the videos, pick a video, then pick how strong the blur should be. The strength does not change face detection. The preview takes a few seconds, longer when the video is longer. You get `<name>_preview/` with the faces covered. If that looks right, run the whole video.

```
/path/to/night-face-blur
```

No menu:

```
/path/to/night-face-blur --level medium video.mp4
```

The full run can take about 10 times the length of the video, sometimes more. It depends on the CPU and the graphics card.

Three strengths: soft, medium, hard. Medium is the one checked on night video. Faces looking down, the back of a head, and license plates stay visible. Daylight caught faces too, but that test was small.

The original file stays. The result is `<name>-blur` next to it. The next video in the same folder starts from the saved strength and still gets its own preview. The choice is saved in `conf/`. A new file is written only when you change the strength.

## Install

ffmpeg and Python 3.14. Other Python versions were not tried.

```
python3 -m venv .venv
```

```
.venv/bin/pip install -r requirements.txt
```

This install is about 2.2 GB. Most of it is NVIDIA libraries, and pip downloads them even on a PC with no NVIDIA card. A smaller install without those libraries was not tried.

A graphics card is optional. Tried on an RTX 3060 Ti, driver 580. If CUDA does not start, the program says so and runs detection on the CPU. Blur and encoding stay on the CPU. On two short clips the CPU was about 3 times slower.

The first run downloads `yolov8n-face.onnx` (12 MB) from this repo's release, if the file is not already next to the script. The hash is checked.

Weight file. That repo has no license file:
https://github.com/yakhyo/yolov8-face-onnx-inference

Training project, GPL-3.0:
https://github.com/derronqi/yolov8-face

This script is MIT. The weight is not in git.
