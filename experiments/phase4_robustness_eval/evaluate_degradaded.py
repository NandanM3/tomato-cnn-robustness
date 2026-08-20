"""
Phase 4 — Robustness Evaluation
 
Runs the same frozen model (no retraining) against all synthetic
degraded test sets generated in Phase 3, one condition/severity
combination at a time.
"""
 
import sys
from pathlib import Path
 
import pandas as pd
import tensorflow as tf
 
common_path = Path(__file__).resolve().parents[1] / "experiments_common"
sys.path.append(str(common_path))
sys.path.append(str(Path(__file__).resolve().parents[1] / "phase3_degradation_scripts"))
 
from eval_utils import infer_class_names, evaluate_directory, compute_metrics
from degradation import DEGRADATION_CONFIG
 
# adjust these to match your repo
# change model path and results path to change model
MODEL_PATH = r"tomato_leaf_disease_model.h5"
TRAIN_DIR = "data/Tomato_Leaf_Data_Split/train"  # source of truth for class index order
SYNTHETIC_ROOT = "data/synthetic"
RESULTS_PATH = "results/tables/phase4_robustness_eval_scratch.csv"
 
 
def main():
    # Loaded once, outside the loop — same frozen weights used for all 12 evaluations.
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = infer_class_names(TRAIN_DIR)
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    print(f"Detected classes (index order): {class_names}")
 
    rows = []
 
    # Imported from Phase 3 rather than retyped, so Phase 4 always matches
    # whatever conditions/severities Phase 3 actually generated.
    for condition, severities in DEGRADATION_CONFIG.items():
        for severity in severities:
            test_dir = Path(SYNTHETIC_ROOT) / condition / severity
            if not test_dir.exists():
                print(f"Skipping {condition}/{severity} — folder not found "
                      f"(did you run generate_all() in Phase 3?)")
                continue
 
            print(f"Evaluating {condition} / {severity} ...")
            true_labels, pred_labels, confidences = evaluate_directory(
                model, test_dir, idx_to_class, (128,128)
            )
            metrics = compute_metrics(true_labels, pred_labels, class_names)
 
            rows.append({
                "condition": condition,
                "severity": severity,
                "n_images": len(true_labels),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
            })
 
    results_df = pd.DataFrame(rows)
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
 
    print(f"\nSaved {len(rows)} condition/severity results to {RESULTS_PATH}")
    print(results_df)
 
 
if __name__ == "__main__":
    main()
 