import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Configuration
IMG_HEIGHT = 128  # Reduced for speed
IMG_WIDTH = 128   # Reduced for speed
BATCH_SIZE = 4    # Small batch for small dataset
EPOCHS = 10       # Reaching accuracy quickly on tiny data
TRAIN_DIR = 'data/Train/'
TEST_DIR = 'data/Test/'
SMALL_DIR = 'data_small'
MODEL_PATH = 'models/car_logo_model.h5'

def create_small_dataset(source_dir, target_dir, count=20):
    """Creates a tiny subset of the dataset for fast training."""
    if not os.path.exists(source_dir):
        return False
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    for brand in os.listdir(source_dir):
        brand_path = os.path.join(source_dir, brand)
        if os.path.isdir(brand_path):
            target_brand_path = os.path.join(target_dir, brand)
            os.makedirs(target_brand_path)
            
            images = [f for f in os.listdir(brand_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
import os

# --- Configuration ---
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10  # Adjusted for transfer learning efficiency
DATASET_PATH = 'data/Train'

def train_model():
    """Train a car logo classifier using VGG19 transfer learning."""
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path {DATASET_PATH} not found.")
        return

    # Data Augmentation & Preprocessing
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Load pre-trained VGG19 model (exclude top layers)
    base_model = applications.VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False  # Freeze base layers

    # Build final model architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks for better training management
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print("Starting training...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stop]
    )

    # Save the finalized model
    os.makedirs('models', exist_ok=True)
    model.save('models/car_logo_model.h5')
    print("Model saved to models/car_logo_model.h5")

if __name__ == "__main__":
    train_model()
