# ============================================================
# LeafVision — Prediction Visualizer
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ────────────────────────────────────────────────────────────
# CONFIG — update these paths
# ────────────────────────────────────────────────────────────
MODEL_PATH = r'leafvision_efficientnet_v1.keras'
IMAGE_PATH = r'data\synthetic\resolution_reduction\moderate\Tomato___Early_blight\image (30).JPG'  # clean or degraded — works for both

CLASS_NAMES = ['Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___healthy']  # alphabetical
IMG_SIZE = (128, 128)

# ────────────────────────────────────────────────────────────
# LOAD + PREDICT
# ────────────────────────────────────────────────────────────
model = load_model(MODEL_PATH)

img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)
img_array = np.expand_dims(image.img_to_array(img), axis=0)

preds = model.predict(img_array, verbose=0)[0]
pred_class = CLASS_NAMES[np.argmax(preds)]
confidence = float(np.max(preds)) * 100

# ────────────────────────────────────────────────────────────
# VISUALIZE
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor('#f8f9fa')

# Left: image
axes[0].imshow(img)
axes[0].axis('off')
axes[0].set_title("Input Image", fontsize=12, color='#333')

# Right: probability bar chart
is_low_confidence = confidence < 60
bar_color = '#e57373' if is_low_confidence else '#4CAF50'
muted_color = '#ef9a9a' if is_low_confidence else '#90CAF9'
title_color = '#c62828' if is_low_confidence else '#2e7d32'

colors = [bar_color if c == pred_class else muted_color for c in CLASS_NAMES]
bars = axes[1].barh(CLASS_NAMES, preds * 100, color=colors, edgecolor='white', height=0.5)
axes[1].set_xlim(0, 100)
axes[1].set_xlabel("Confidence (%)", fontsize=11)
axes[1].set_title(f"Prediction: {pred_class}\n{confidence:.1f}% confidence",
                  fontsize=13, fontweight='bold', color=title_color)
axes[1].axvline(x=50, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

for bar, prob in zip(bars, preds):
    axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f'{prob*100:.1f}%', va='center', fontsize=10)

axes[1].set_facecolor('#f8f9fa')
plt.tight_layout()

output_name = 'prediction_output.png'
plt.savefig(output_name, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {output_name} | {pred_class} ({confidence:.1f}%)")