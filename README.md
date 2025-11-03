Indian Classical Dance Image Classification

End-to-end image classification project to recognize 8 Indian classical dance forms using TensorFlow/Keras. The workflow covers data preparation, tf.data pipelines, two model baselines (a Simple CNN and a MobileNetV2 transfer-learning model), training, evaluation, and inference on unseen images.

This README summarizes the full notebook in `project.ipynb` and provides quick-start instructions to reproduce results locally.

Dataset
- Source layout (after unzipping `archive.zip` into this folder):
  - `dataset/train/` — training images
  - `dataset/test/` — unlabeled test images
  - `dataset/train.csv` — image filename and label per row
  - `dataset/test.csv` — image filename per row (no labels)

- Classes (8 total): `bharatanatyam`, `kathak`, `kathakali`, `kuchipudi`, `manipuri`, `mohiniyattam`, `odissi`, `sattriya`.

Project Structure
- `project.ipynb` — main notebook (data prep → training → evaluation → inference)
- `mobilenet_transfer_model.keras` — saved best model (MobileNetV2 head)
- `simple_cnn_model.keras` — saved baseline CNN
- `dataset/` — images and CSVs (expected after unzip)

Environment Setup
1) Create a virtual environment (recommended):
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install dependencies:
```bash
pip install tensorflow pandas scikit-learn matplotlib seaborn
```

3) Unzip the data archive:
```bash
```

How It Works (High Level)
1) Load `dataset/train.csv`, inspect class distribution.
2) Build train/val/test split from the labeled training data (stratified).
3) Create performant `tf.data` pipelines (decode → resize → batch → prefetch; optional cache and shuffle for train).
4) Train two models:
   - Simple CNN (from scratch) with basic data augmentation + rescaling.
   - MobileNetV2 transfer learning (frozen backbone, custom classification head) with preprocessing + light dropout.
5) Evaluate both models on a held-out test split with Accuracy and Weighted F1.
6) Visualize curves and confusion matrix; save trained models to `.keras` files.

Key Training Details
- Image size: 160×160×3
- Batch size: 32
- Data augmentation: RandomFlip, RandomRotation(0.1), RandomZoom(0.1)
- Optimizer: Adam
- Loss: SparseCategoricalCrossentropy
- Epochs: 50 (adjustable)

Results (Held-out Test Split)
From `project.ipynb` final evaluation:

| Model                    | Accuracy | Weighted F1-Score | Train Time (s) |
|--------------------------|----------|-------------------|----------------|
| Simple CNN               | 0.5818   | 0.5768            | 403.19         |
| MobileNetV2 (Transfer)   | 0.7091   | 0.7030            | 200.26         |

MobileNetV2 transfer learning outperforms the scratch CNN and is saved as `mobilenet_transfer_model.keras`.

Reproduce Training
Option A — Run the notebook end-to-end:
1) Open `project.ipynb`.
2) Run cells sequentially (ensure `dataset/` exists). Models will be saved at the end.

Option B — Scriptify (optional):
- Convert the core cells to a Python script if you prefer CLI execution.

Inference on an Unseen Image
The notebook includes a demo for a random image from `dataset/test/`.

Minimal example (Python):
```python
import tensorflow as tf
import numpy as np

IMAGE_SIZE = (160, 160)
class_names = [
    'bharatanatyam','kathak','kathakali','kuchipudi',
    'manipuri','mohiniyattam','odissi','sattriya'
]

model = tf.keras.models.load_model('mobilenet_transfer_model.keras')

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.expand_dims(img, 0)
    return img

img = load_image('dataset/test/145.jpg')  # replace with any test image
probs = model.predict(img)[0]
pred_idx = int(np.argmax(probs))
print('Predicted:', class_names[pred_idx], 'Confidence:', float(np.max(probs)))
```


