import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0  
from tensorflow.keras.applications.efficientnet import preprocess_input

import sklearn.metrics as metrics
import numpy as np
import os

#-----
#Data Preparation
#-----

#path to split data
BASE_DIR = r'C:\Users\Menon\OneDrive\Documents\LeafVision-ML\data\Tomato_Leaf_Data_Split'

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR,'val')
TEST_DIR = os.path.join(BASE_DIR,'test')

#Constant image parameters
IMG_SIZE = (128,128)
BATCH_SIZE = 32

#Normalising pixel values (0-255 -> 0-1)
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,       
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    shuffle = False
)


#----
#Building the CNN model version - 001
#----


#Base Model - EfficientNetB0

base_model = EfficientNetB0(
    weights = 'imagenet',
    include_top = False,
    input_shape = (128,128,3)
)

base_model.trainable = False



model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation= 'relu'),

    layers.Dropout(0.5),

    layers.Dense(
        train_generator.num_classes,
          activation= 'softmax'
        )
])






#----
#Compiling the model
#----

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']

)

#----
#Training the model
#----

history = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs = 10
)

#----
#Evaluating the model on the test set
#----

test_loss, test_accuracy = model.evaluate(test_generator)
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


cm = metrics.confusion_matrix(
    true_classes, predicted_classes
    )

print("Confusion Matrix:")
print(cm)

#----
#Saving the model
#----

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

model.save('leafvision_efficientnet_v1.keras')
print("Model saved as 'leafvision_efficientnet_v1.keras'")
print("Class indices:", train_generator.class_indices)


# ───
# 2. CONFUSION MATRIX ->  saved as confusion_matrix.png
# ───
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=list(test_generator.class_indices.keys()),
    yticklabels=list(test_generator.class_indices.keys())
)

plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix — LeafVision EfficientNet-B0", fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("Saved: confusion_matrix.png")


# ─────
# 3. Classification Report ->  saved as classification_report.png
# ─────
report = classification_report(
    true_classes,
    predicted_classes,
    target_names=list(test_generator.class_indices.keys()),
    output_dict=True
)

df = pd.DataFrame(report).transpose().round(2)

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.axis('off')

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    rowLabels=df.index,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

plt.title("Classification Report — LeafVision EfficientNet-B0", fontsize=13, pad=16)
plt.tight_layout()
plt.savefig('classification_report.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: classification_report.png")



 