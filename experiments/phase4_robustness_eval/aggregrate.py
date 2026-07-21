"""
Phase 4 — Aggregate Results
 
Combines the per-condition/per-severity metrics from evaluate_degraded.py
into degradation curves and a summary table of accuracy loss relative to
the clean baseline.
"""
 
import json
from pathlib import Path
 
import pandas as pd
import matplotlib.pyplot as plt
 
ROBUSTNESS_RESULTS_PATH = "results/tables/phase4_robustness_eval.csv"
CLEAN_RESULTS_PATH = "results/tables/phase2_clean_eval.json"
FIGURES_DIR = "results/figures"
 
# Explicit ordering for the x-axis. "mild"/"moderate"/"severe" don't sort
# into the right order alphabetically (moderate < severe < mild), so this
# has to be spelled out rather than relying on default sorting.
SEVERITY_ORDER = ["mild", "moderate", "severe"]
 
 
def load_clean_baseline():
    with open(CLEAN_RESULTS_PATH) as f:
        return json.load(f)["accuracy"]
 
 
def plot_degradation_curves(df, clean_accuracy, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
 
    # One line per condition (brightness_contrast, blur_gaussian, etc.),
    # each showing how accuracy changes from mild -> moderate -> severe.
    for condition in df["condition"].unique():
        subset = df[df["condition"] == condition].copy()
        # Force severity into the correct order regardless of what order
        # rows happen to sit in the CSV.
        subset["severity"] = pd.Categorical(
            subset["severity"], categories=SEVERITY_ORDER, ordered=True
        )
        subset = subset.sort_values("severity")
        ax.plot(subset["severity"], subset["accuracy"], marker="o", label=condition)
 
    # A flat reference line at the clean accuracy, so every curve's drop
    # is visible against the same starting point.
    ax.axhline(clean_accuracy, color="black", linestyle="--", linewidth=1,
               label=f"Clean baseline ({clean_accuracy:.1%})")
 
    ax.set_xlabel("Severity")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy vs. Degradation Severity")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
 
    output_path = Path(output_dir) / "degradation_curves.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved degradation curve plot to {output_path}")
 
 
def summarize_drops(df, clean_accuracy):
    """Compute and print accuracy drop relative to clean baseline for
    each condition/severity, sorted worst-first."""
    df = df.copy()
    df["accuracy_drop"] = clean_accuracy - df["accuracy"]
    df = df.sort_values("accuracy_drop", ascending=False)
    print("\nAccuracy drop relative to clean baseline (largest first):")
    print(df[["condition", "severity", "accuracy", "accuracy_drop"]].to_string(index=False))
    return df
 
 
def main():
    df = pd.read_csv(ROBUSTNESS_RESULTS_PATH)
    clean_accuracy = load_clean_baseline()
 
    plot_degradation_curves(df, clean_accuracy, FIGURES_DIR)
    summary_df = summarize_drops(df, clean_accuracy)
 
    summary_path = Path(FIGURES_DIR).parent / "tables" / "phase4_summary_with_drops.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary table to {summary_path}")
 
 
if __name__ == "__main__":
    main()

     