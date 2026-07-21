"""
Shared evaluation utilities for degraded and clean evaluation.

 This matters methodologically: it guarantees identical
preprocessing, identical prediction logic, and identical metric
calculations whether the test images are clean or degraded. Any
difference in the results is then attributable to the image conditions
themselves, not to inconsistencies in how they were measured.
"""

from pathlib import Path
 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tensorflow.keras.applications.efficientnet import preprocess_input
 
 
def infer_class_names(directory):
    """
    Returns class names sorted alphabetically matching the index order
    Keras assigns during training.
 
    Must point at the TRAINING directory
    """
    directory = Path(directory)
    return sorted(p.name for p in directory.iterdir() if p.is_dir())
 
 
def load_and_prep(img_bgr, size):
    """Take a raw image straight off disk (as OpenCV loads it) and turn
    it into the exact shape and format the model expects."""
    # OpenCV loads images as BGR (blue-green-red) by convention, but the  model was trained on RGB images. Swap the channel order.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Resize to whatever input size the model was trained on.
    img_resized = cv2.resize(img_rgb, size)
    # The model expects a "batch" of images, even a batch of one This adds a new first dimension: (224, 224, 3) -> (1, 224, 224, 3).
    batch = np.expand_dims(img_resized.astype(np.float32), axis=0)
    # EfficientNet's own preprocessing step
    return preprocess_input(batch)
 
 
def predict_single(model, img_bgr, idx_to_class, size=(224, 224)):
    """Run one image through the model. Returns (predicted_label, confidence_pct)."""
    batch = load_and_prep(img_bgr, size)
    # model.predict returns a probability for each class
    # verbose=0 just silences Keras's default progress bar.
    probs = model.predict(batch, verbose=0)[0]
    # argmax finds the index of the highest probability = the model's guess.
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    return pred_label, confidence
 
 
def evaluate_directory(model, root_dir, idx_to_class, size=(224, 224)):
    """
    Walk a folder structured as:
        root_dir/
          ClassA/img1.jpg, img2.jpg, ...
          ClassB/img1.jpg, ...
          ClassC/img1.jpg, ...
 
    Run every image through the model and record the true label (the
    folder name) against the model's guess.
 
    Returns three parallel lists: true labels, predicted labels, confidences.
    """
    root_dir = Path(root_dir)
    true_labels, pred_labels, confidences = [], [], []
    count = 0
 
    # Each subfolder name IS the true class — that's the whole point of
    # organizing image data this way instead of a separate labels file.
    for class_folder in sorted(root_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        true_class = class_folder.name
 
        for img_path in sorted(class_folder.iterdir()):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                # Skip unreadable files rather than crashing the whole run
                # over one corrupted image.
                print(f"Warning: could not read {img_path}, skipping")
                continue
 
            pred_label, confidence = predict_single(model, img_bgr, idx_to_class, size)
 
            true_labels.append(true_class)
            pred_labels.append(pred_label)
            confidences.append(confidence)
 
            count += 1
            if count % 50 == 0:
                print(f"  ...{count} images evaluated so far")
 
    return true_labels, pred_labels, confidences
 
 
def compute_metrics(true_labels, pred_labels, class_names):
    """
    Turn raw true/predicted label lists into the numbers that go in the
    paper: overall accuracy, plus precision/recall/F1 both per-class and
    macro-averaged (a simple mean across classes, each weighted equally —
    stops a large class from hiding a small class's poor performance).
    """
    accuracy = accuracy_score(true_labels, pred_labels)
 
    # Passing labels=class_names guarantees the output arrays are in
    # the same class order every time, regardless of which classes
    # happened to appear first in this particular test set.
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=class_names, zero_division=0
    )
 
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
 
    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }
 
    return {
        "accuracy": float(accuracy),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
 
 
def save_confusion_matrix(true_labels, pred_labels, class_names, output_path):
    """Save a confusion matrix heatmap as a PNG."""
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Save BEFORE show — plt.show() can clear the figure buffer in some
    # environments, so saving after would silently produce a blank image.
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()