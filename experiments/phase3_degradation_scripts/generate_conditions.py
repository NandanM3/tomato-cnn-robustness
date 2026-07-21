"""
generate_conditions.py - 
Reads clean test images from data/processed/test/ and writes degraded
versions to data/synthetic/{condition}/{severity}/{class}/{filename}.
 
Only the test split is ever touched (train and val are never read or
modified by this script). See split_dataset.py for how the split was made.
 
Run this once before Phase 4 evaluation. Re-running is safe (exist_ok=True
on all folder creation) but will overwrite existing degraded images.
 
Usage:
    python generate_conditions.py
"""
 



import os
import cv2
import numpy as np
from pathlib import Path
 
from degradation import (
    DEGRADATION_CONFIG,
    CONDITION_FUNCTIONS,
)

 
TEST_DIR   = Path("data") / "Tomato_Leaf_Data_Split" / "test"
OUTPUT_DIR = Path("data") / "synthetic"
 
 
#IMAGE I/O 
 
def load_image(path):
    img = cv2.imread(str(path))             # str() for older OpenCV compatibility
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
 
def save_image(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
 
 
# MAIN LOOP 
 
def generate_all():
    """
    For every image in data/processed/test/, apply all conditions and
    severity levels, and write outputs to data/synthetic/.
 
    Loop order: class -> image -> condition -> severity
    Works well becuase we need to load each img only once
    Output path structure mirrors the input class structure:
        data/synthetic/{condition}/{severity}/{class}/{original_filename}
 
    The condition and severity folder names are taken directly from
    DEGRADATION_CONFIG keys(they match the folder names exactly so
    Phase 4 evaluation can construct paths using the same dict).
    """
    total_written = 0
 
    # os.listdir returns class folder names: ["blight", "early_blight", "healthy"]
    # sorted() ensures consistent ordering across operating systems 
    # os.listdir order is not guaranteed to be the same on Linux 

    class_names = sorted(os.listdir(TEST_DIR))
 
    for class_name in class_names:
        class_dir = TEST_DIR / class_name

        image_files = [
            f for f in sorted(os.listdir(class_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
 
        print(f"\nClass: {class_name} ({len(image_files)} images)")
 
        for image_filename in image_files:
            image_path = class_dir / image_filename
 
            # All conditions and severities are applied to
            # this single loaded array before we move to the next file
            img = load_image(image_path)
 
            for condition_name, severity_dict in DEGRADATION_CONFIG.items():
 
                fn = CONDITION_FUNCTIONS[condition_name]
 
                for severity_name, params in severity_dict.items():
                    # severity_name: "mild", "moderate", or "severe"
                    # params: e.g. {"alpha": 0.9, "beta": -10}
 
                    # Apply the transform.
                    # **params unpacks the dict into keyword arguments:
                    # fn(img, **{"alpha": 0.9, "beta": -10})
                    # becomes: apply_brightness_contrast(img, alpha=0.9, beta=-10)
                    # This works because the dict keys match the function's
                    # parameter names exactly — by design in degradation_functions.py
                    degraded = fn(img, **params)
 
                    # Construct output path using Path's / operator.
                    # condition_name and severity_name are already the right
                    # strings for folder names — no translation needed.
                    output_path = (
                        OUTPUT_DIR
                        / condition_name    # e.g. "brightness_contrast"
                        / severity_name     # e.g. "mild"
                        / class_name        # e.g. "blight"
                        / image_filename    # e.g. "img001.jpg" — same name as source
                    )
 
                    save_image(degraded, output_path)
                    total_written += 1
 
        # Per-class summary: printed after all conditions/severities for this class
        print(f"  Done: {len(image_files)} images x "
              f"{len(DEGRADATION_CONFIG)} conditions x "
              f"3 severities = "
              f"{len(image_files) * len(DEGRADATION_CONFIG) * 3} files written")
 
    print(f"\nFinished. {total_written} total degraded images written to {OUTPUT_DIR}")
 
 
# SANITY CHECK
 
def sanity_check(class_name=None):
    """
    Run all transforms on a single image and save results to a scratch folder
    for visual inspection before running generate_all() on the full dataset.
 
 
    Usage:
        sanity_check()               # uses first class and first image found
        sanity_check("early_blight") # uses first image from a specific class
 
    Outputs to: data/synthetic/_sanity_check/{condition}/{severity}/
    """
    SCRATCH_DIR = OUTPUT_DIR / "_sanity_check"
 
    # Find one image to test with
    if class_name is None:
        class_name = sorted(os.listdir(TEST_DIR))[0]
 
    class_dir = TEST_DIR / class_name
    test_image_name = sorted(os.listdir(class_dir))[0]
    test_image_path = class_dir / test_image_name
 
    print(f"Sanity check using: {test_image_path}")
    img = load_image(test_image_path)
 
    for condition_name, severity_dict in DEGRADATION_CONFIG.items():
        fn = CONDITION_FUNCTIONS[condition_name]
        for severity_name, params in severity_dict.items():
            degraded = fn(img, **params)
            out_path = SCRATCH_DIR / condition_name / f"{severity_name}_{test_image_name}"
            save_image(degraded, out_path)
            print(f"  Saved: {out_path}")
 
    # saves the clean original for direct comparison
    save_image(img, SCRATCH_DIR / f"_clean_{test_image_name}")
    print(f"\nSanity check complete. Open {SCRATCH_DIR} and inspect visually.")
    print("Run generate_all() only after confirming severity levels look correct.")
 
 
 
 
if __name__ == "__main__":


    """      Step 1: always run sanity_check() first on a fresh setup
             Step 2: once you've visually confirmed the outputs look right,
                 comment out sanity_check() and uncomment generate_all() """
 
    #sanity_check()                            #ADD COMMENT WHEN SANITY CHECK DONE 
    generate_all()                          #REMOVE COMMENT WHEN SANITY CHECK DONE