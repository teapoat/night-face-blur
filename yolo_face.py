"""YOLOv8-Face ONNX detector. NumPy only, no torch."""

import sys
from pathlib import Path


def _prepare_cudnn():
    """The cuDNN wheel ships libcudnn.so.9 and often no unversioned libcudnn.so.

    Setting LD_LIBRARY_PATH here is too late: the dynamic loader does not pick
    up a change made after the process has started. The night-face-blur script
    exports the directory before exec. This only adds the missing soname symlink.
    """
    for entry in sys.path:
        lib = Path(entry) / "nvidia" / "cudnn" / "lib"
        if not lib.is_dir():
            continue
        soname = lib / "libcudnn.so.9"
        link = lib / "libcudnn.so"
        if soname.is_file() and not link.exists():
            link.symlink_to(soname.name)
        return


_prepare_cudnn()

import cv2
import numpy as np
import onnxruntime as ort

STRIDES = (8, 16, 32)


def letterbox(img, new_shape=(640, 640)):
    height, width = img.shape[:2]
    ratio = min(new_shape[0] / height, new_shape[1] / width)
    resized = (int(round(width * ratio)), int(round(height * ratio)))
    dw = (new_shape[1] - resized[0]) / 2
    dh = (new_shape[0] - resized[1]) / 2
    out = cv2.resize(img, resized, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(
        out, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return out, ratio, left, top


def softmax(values, axis=-1):
    shifted = np.exp(values - np.max(values, axis=axis, keepdims=True))
    return shifted / np.sum(shifted, axis=axis, keepdims=True)


def nms(boxes, scores, iou_threshold):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


class YoloFace:
    def __init__(self, conf=0.4, iou=0.45, size=640, provider="cpu", model=None):
        self.conf = conf
        self.iou = iou
        if provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(provider)
        if model is None:
            model = Path(__file__).with_name("yolov8n-face.onnx")
        self.session = ort.InferenceSession(str(model), providers=providers)
        self.provider = self.session.get_providers()[0]
        if provider == "cuda" and self.provider != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA was requested, session uses {self.provider}")
        self.input_name = self.session.get_inputs()[0].name
        # Training size is 640. Larger input keeps small faces detectable.
        self.img_size = (size, size)

    def _decode(self, outputs):
        boxes_all = []
        scores_all = []
        for pred, stride in zip(outputs, STRIDES):
            _, channels, height, width = pred.shape
            pred = pred.reshape(1, channels, -1).transpose(0, 2, 1)[0]
            grid_y, grid_x = np.meshgrid(
                np.arange(height) + 0.5, np.arange(width) + 0.5, indexing="ij"
            )
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()
            dist = softmax(pred[:, :64].reshape(-1, 4, 16), axis=-1) @ np.arange(16)
            boxes = np.stack(
                [
                    (grid_x - dist[:, 0]) * stride,
                    (grid_y - dist[:, 1]) * stride,
                    (grid_x + dist[:, 2]) * stride,
                    (grid_y + dist[:, 3]) * stride,
                ],
                axis=-1,
            )
            scores = 1 / (1 + np.exp(-pred[:, 64]))
            boxes_all.append(boxes)
            scores_all.append(scores)
        boxes = np.concatenate(boxes_all, axis=0)
        scores = np.concatenate(scores_all, axis=0)
        mask = scores >= self.conf
        boxes, scores = boxes[mask], scores[mask]
        if len(boxes) == 0:
            return []
        keep = nms(boxes, scores, self.iou)[:50]
        return [(boxes[i], float(scores[i])) for i in keep]

    def detect(self, frame):
        canvas, ratio, left, top = letterbox(frame, self.img_size)
        tensor = canvas[:, :, ::-1].astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]
        tensor = np.ascontiguousarray(tensor)
        outputs = self.session.run(None, {self.input_name: tensor})
        height, width = frame.shape[:2]
        found = []
        for box, score in self._decode(outputs):
            x1 = int(np.clip((box[0] - left) / ratio, 0, width - 1))
            y1 = int(np.clip((box[1] - top) / ratio, 0, height - 1))
            x2 = int(np.clip((box[2] - left) / ratio, 0, width - 1))
            y2 = int(np.clip((box[3] - top) / ratio, 0, height - 1))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            found.append((x1, y1, x2 - x1, y2 - y1, score))
        return found
