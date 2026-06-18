"""
Minimal CPU inference example for the X-ray body-part classifier (ONNX).

Preprocessing MUST match training: convert to RGB, resize the shorter edge to 224,
center-crop 224x224, scale to [0,1], normalize with ImageNet mean/std, layout NCHW.
The ONNX graph ends in a softmax, so the "probs" output is already a probability
distribution over the classes in classes.txt (same order).

    pip install -r requirements.txt
    python inference_example.py path/to/xray.jpg
"""

import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_classes(path="classes.txt"):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def preprocess(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = IMG_SIZE / min(w, h)                      # resize shorter edge to 224
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))   # center crop
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)[None]                    # HWC -> NCHW + batch dim
    return np.ascontiguousarray(x, dtype=np.float32)


def main(image_path, model_path="model.onnx", topk=5):
    classes = load_classes()
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    probs = sess.run(["probs"], {"images": preprocess(image_path)})[0][0]
    topk = min(topk, len(classes))
    order = probs.argsort()[::-1][:topk]
    print(f"Top-{topk} predictions for {image_path}:")
    for i in order:
        print(f"  {classes[i]:<20s} {probs[i] * 100:5.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python inference_example.py <image.(jpg|png)>")
        sys.exit(1)
    main(sys.argv[1])
