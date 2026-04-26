# Plant Disease Detection (Deep Learning)

A deep-learning project for **plant leaf disease detection and classification** using **Keras / TensorFlow** and **MobileNetV2** (a lightweight CNN architecture designed for efficient vision models).

> Repo note: this repository is primarily composed of Jupyter Notebooks. The core training/inference logic and experiments are expected to live inside the notebooks.

## Project overview

- **Goal:** classify plant leaf images into disease/healthy categories.
- **Approach:** transfer learning with **MobileNetV2**.
- **Framework:** Keras (TensorFlow backend).
- **Why MobileNetV2?** It is compute-efficient and commonly used for mobile / edge vision applications.

## Repository structure

The repository currently contains mostly notebooks. Common files/folders you may see or want to add:

- `*.ipynb` — training, evaluation, and experiments
- `data/` — dataset location (often gitignored)
- `models/` — saved models / checkpoints (often gitignored)
- `outputs/` — figures, metrics, predictions (optional)

If your repo uses different paths, update this section to match your actual layout.

## Getting started

### 1) Prerequisites

- Python 3.9+ (3.10/3.11 usually works as well)
- pip or conda
- (Recommended) a GPU runtime (NVIDIA CUDA) for faster training

### 2) Install dependencies

If you have a `requirements.txt`, use it:

```bash
pip install -r requirements.txt
```

If you don’t, this is a typical baseline for Keras + notebooks:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
```

### 3) Launch the notebooks

```bash
jupyter notebook
```

Open the training/evaluation notebook(s) and run the cells in order.

## Data

This project expects a dataset of labeled plant leaf images.

- Put your dataset in a local folder (commonly `data/`).
- Organize it by class (typical `ImageDataGenerator.flow_from_directory` layout):

```
data/
  train/
    class_1/
    class_2/
    ...
  val/
    class_1/
    class_2/
    ...
  test/
    class_1/
    class_2/
    ...
```

If you used a specific dataset source (e.g., PlantVillage or a Kaggle dataset), add the link and citation here.

## Training (high level)

Typical workflow used in the notebooks:

1. Load images with a Keras data pipeline (e.g., `ImageDataGenerator` or `tf.data`).
2. Initialize **MobileNetV2** with ImageNet weights.
3. Replace the classification head for your target classes.
4. Train the head (optionally freeze the base model first).
5. Fine-tune selected layers.
6. Evaluate with accuracy, confusion matrix, and sample predictions.

## Inference

Once a model is trained and saved, you can load it and run predictions on new images.

Example (illustrative):

```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("path/to/saved_model")
img = tf.keras.utils.load_img("leaf.jpg", target_size=(224, 224))
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)

pred = model.predict(x)
print(pred)
```

> Update input size, preprocessing, and class label mapping to match your notebook implementation.

## Results

Add your key results here (recommended):

- Final validation accuracy: **TBD**
- Test accuracy: **TBD**
- Example predictions / confusion matrix screenshot(s): **TBD**

## Reproducibility

To improve reproducibility, consider adding:

- `requirements.txt` (or `environment.yml`)
- fixed random seeds in notebooks
- clear dataset download + preprocessing steps
- saved model artifacts or instructions to reproduce them

## Roadmap / improvements

- [ ] Add `requirements.txt`
- [ ] Add a script or notebook for inference on a folder of images
- [ ] Export the model to TensorFlow Lite for mobile deployment
- [ ] Add model evaluation report (confusion matrix, per-class metrics)

## License

Choose a license for your project (MIT, Apache-2.0, etc.) and add a `LICENSE` file.

## Acknowledgements

- MobileNetV2: *Sandler et al., 2018*
- Keras / TensorFlow community

---

If you share:
- the dataset source/link,
- the notebook filenames,
- the number of classes,

I can tailor this README to your exact project and add concrete commands/paths and reported metrics.
