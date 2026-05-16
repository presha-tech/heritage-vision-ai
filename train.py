import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# =========================
# SETTINGS
# =========================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

DATASET_PATH = "dataset/train"

# =========================
# DATA PREPROCESSING
# =========================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# =========================
# LOAD RESNET50
# =========================

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# =========================
# BUILD MODEL
# =========================

model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dropout(0.5),

    Dense(256, activation='relu'),

    Dropout(0.3),

    Dense(5, activation='softmax')  # 5 monuments
])

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# CALLBACKS
# =========================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_resnet_model.h5",
    monitor='val_accuracy',
    save_best_only=True
)

# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# =========================
# SAVE FINAL MODEL
# =========================

model.save("heritage_resnet_model.keras")

print("\nTraining Complete!")
print("Model saved as heritage_resnet_model.h5")

# =========================
# CLASS LABELS
# =========================

print("\nClass Indices:")
print(train_generator.class_indices)