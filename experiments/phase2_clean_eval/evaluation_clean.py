"""
Phase 2 — Clean Evaluation

Runs the frozen, trained model against the clean test set to produce
baseline accuracy/precision/recall/F1 and a confusion matrix.
"""

import json
import sys
from pathlib import Path

import tensorflow as tf

# experiments/common holds evaluation code shared across phases.
common_path = Path(__file__).resolve().parents[1] / "experiments_common"
sys.path.append(str(common_path))
from eval_utils import (    
    infer_class_names,
    evaluate_directory,
    compute_metrics,
    save_confusion_matrix,
)

#Change model and results path based on baseline/transfer learning model
MODEL_PATH = "tomato_leaf_disease_model.h5"
TRAIN_DIR = "data/Tomato_Leaf_Data_Split/train" 
TEST_DIR = "data/Tomato_Leaf_Data_Split/test"
RESULTS_PATH = "results/tables/phase2_clean_eval_scratch.json"
CONFUSION_MATRIX_PATH = "results/figures/phase2_confusion_matrix_scratch.png"


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    # Derived from folder names rather than a saved file — see infer_class_names().
    class_names = infer_class_names(TRAIN_DIR)
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    print(f"Detected classes (index order): {class_names}")

    true_labels, pred_labels, confidences = evaluate_directory(
        model, TEST_DIR, idx_to_class, (128,128)
    )

    metrics = compute_metrics(true_labels, pred_labels, class_names)

    print(f"Evaluated {len(true_labels)} images")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(true_labels, pred_labels, class_names, CONFUSION_MATRIX_PATH)

    print(f"Saved metrics to {RESULTS_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()