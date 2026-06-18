---
license: apache-2.0
library_name: onnx
pipeline_tag: image-classification
tags:
  - medical
  - radiology
  - x-ray
  - radiograph
  - anatomy
  - body-part
  - image-classification
  - onnx
metrics:
  - accuracy
---

# X-Ray Body-Part Classifier (ConvNeXt-Tiny, ONNX)

A CPU-friendly **body-part / anatomy classifier for plain radiographs (X-ray)**. Given a single
rendered X-ray frame it predicts the imaged anatomy across **33 classes** (CHEST, KNEE, LUMBAR_SPINE,
ABDOMEN, …). Exported to **ONNX** with a built-in softmax, so the output is a ready-to-use probability
distribution and it runs anywhere with `onnxruntime` — no GPU required.

It was built to fill the "vision gap" in a radiology workflow: suggesting the likely anatomy when the
text order / DICOM tags are missing, opaque, or mislabelled. **It is a decision-support suggestion
model, not a diagnostic device.**

## ⚠️ Intended use & limitations

- **Intended use:** a *suggestion/assist* signal — surface the likely body part to a human reviewer,
  ideally as a ranked top-k list behind a confidence threshold.
- **NOT for clinical or diagnostic use.** It classifies *anatomy*, not pathology, and must never drive
  an unsupervised clinical decision.
- **Coarse labels with known overlap.** Several classes are hierarchical / overlapping
  (`HEAD`↔`SKULL`, `KUB`↔`ABDOMEN`, `SPINE`↔`LUMBAR/CERVICAL/DORSAL_SPINE`,
  `EXTREMITY`↔`ARM`/`LEG`/`FOREARM`). This caps top-1 (a `KUB` image read as `ABDOMEN` is "wrong" but
  practically correct), which is why **top-5 (0.94) is the more meaningful number than top-1 (0.70)**.
- **Weak on rare / overlapping classes** (see per-class table) — `FINGER`, `HEEL`, `KUB`, `ARM`, `HIP`
  have few samples and/or collapse into larger classes. Use confidence thresholding in production.
- Trained on adult-population radiographs from routine practice; behaviour on paediatric, exotic, or
  heavily-processed images is unverified.

## Performance

Held-out validation: **7,354 images**, 33 classes.

| Metric | Score |
|---|---|
| Top-1 accuracy | **0.704** |
| Top-5 accuracy | **0.940** |

### Per-class recall (validation)

| Class | Recall | n | | Class | Recall | n |
|---|---:|---:|---|---|---:|---:|
| SHOULDER | 0.98 | 355 | | SPINE | 0.60 | 272 |
| KNEE | 0.97 | 400 | | NECK | 0.60 | 400 |
| ABDOMEN | 0.89 | 400 | | LEG | 0.60 | 45 |
| CERVICAL_SPINE | 0.84 | 376 | | WRIST | 0.58 | 202 |
| CHEST | 0.82 | 400 | | UPPER_EXTREMITY | 0.57 | 400 |
| FOOT | 0.80 | 400 | | PELVIS | 0.57 | 400 |
| LUMBAR_SPINE | 0.80 | 400 | | LOWER_EXTREMITY | 0.55 | 400 |
| PNS | 0.79 | 199 | | FOREARM | 0.53 | 95 |
| ANKLE | 0.78 | 292 | | HEAD | 0.50 | 400 |
| ELBOW | 0.77 | 237 | | SI_JOINT | 0.38 | 8 |
| SKULL | 0.77 | 400 | | FEMUR | 0.29 | 34 |
| DORSAL_SPINE | 0.75 | 101 | | EXTREMITY | 0.27 | 56 |
| HAND | 0.73 | 390 | | NASOPHARYNX | 0.24 | 62 |
| TEMPORAL_BONE | 0.71 | 17 | | HIP | 0.16 | 32 |
| TIBIA | 0.67 | 57 | | ARM | 0.04 | 24 |
| | | | | KUB | 0.00 | 71 |
| | | | | FINGER | 0.00 | 15 |
| | | | | HEEL | 0.00 | 14 |

The high-volume, visually distinct anatomies are strong (0.77–0.98); the weak rows are the
overlapping/hierarchical and low-sample classes. Merging those into a cleaner ~15–18-class taxonomy is
the obvious path to a substantially higher-accuracy v2.

## Model details

- **Architecture:** `convnext_tiny` (timm), ImageNet-pretrained, fine-tuned.
- **Input:** RGB image, resize shorter edge to 224, center-crop 224×224, scale to `[0,1]`, normalize
  with ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`, layout `NCHW`. (No
  horizontal-flip augmentation — it would corrupt left/right laterality.)
- **ONNX I/O:** input `images` `[N,3,224,224]` float32 → output `probs` `[N,33]` (softmax). Class order
  is `classes.txt`.
- **Files:** `model.onnx` (FP32) · `best.pt` (PyTorch state dict, for fine-tuning).

## Usage

```bash
pip install -r requirements.txt
python inference_example.py path/to/xray.jpg
```

```python
import numpy as np, onnxruntime as ort
from PIL import Image

classes = [c.strip() for c in open("classes.txt")]
MEAN, STD = np.float32([0.485,0.456,0.406]), np.float32([0.229,0.224,0.225])

img = Image.open("xray.jpg").convert("RGB")
s = 224 / min(img.size); img = img.resize((round(img.size[0]*s), round(img.size[1]*s)))
w, h = img.size; img = img.crop(((w-224)//2, (h-224)//2, (w-224)//2+224, (h-224)//2+224))
x = ((np.asarray(img, np.float32)/255 - MEAN)/STD).transpose(2,0,1)[None]

probs = ort.InferenceSession("model.onnx").run(["probs"], {"images": x})[0][0]
top = probs.argsort()[::-1][:5]
print([(classes[i], round(float(probs[i]), 3)) for i in top])
```

## Training data

~37k de-identified plain-radiograph frames from routine clinical practice, one representative frame per
study/series, labelled from the `body_part_examined` DICOM tag (with the confirmed procedure as a
fallback) and normalized through an anatomy lexicon. Manual uploads, multi-body-part studies, and
conflicting-label images were excluded; classes were balanced (cap 2,000/class). The training dataset
is **not** released. The dataset owner has confirmed the rights to publish this derived model.

## Author

Created by **Istiak Hassan Emon** — GitHub [@emon5122](https://github.com/emon5122).

If you use this model, please credit:

```
Istiak Hassan Emon, "X-Ray Body-Part Classifier (ConvNeXt-Tiny)", 2026.
https://huggingface.co/emon5122/xray-bodypart-classifier
```

## License

`apache-2.0` (matches the ConvNeXt backbone). © 2026 Istiak Hassan Emon. The model is provided
**as-is, with no warranty, and not for clinical use.**
