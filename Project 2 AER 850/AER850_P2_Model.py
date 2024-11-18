# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define image processing parameters
WIDTH = 500
HEIGHT = 500
CHANNELS = 3
INPUT_DIM = (WIDTH, HEIGHT, CHANNELS)

# Define dataset directories
base_dir = "./Data"
training_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "valid")
testing_dir = os.path.join(base_dir, "test")

print("Loading image data...")

# Data augmentation setup for training data
train_data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True
)

# Data generator for validation and test datasets (only rescaling)
val_data_gen = ImageDataGenerator(rescale=1.0 / 255)
test_data_gen = ImageDataGenerator(rescale=1.0 / 255)

# Load the datasets
train_data = train_data_gen.flow_from_directory(
    training_dir,
    target_size=(WIDTH, HEIGHT),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

val_data = val_data_gen.flow_from_directory(
    validation_dir,
    target_size=(WIDTH, HEIGHT),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

test_data = test_data_gen.flow_from_directory(
    testing_dir,
    target_size=(WIDTH, HEIGHT),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

# Display information about the dataset
print("\nDataset Summary:")
print(f"Training samples: {train_data.samples}")
print(f"Validation samples: {val_data.samples}")
print(f"Testing samples: {test_data.samples}")
print("\nClass Indices:", train_data.class_indices)

# Define the Convolutional Neural Network architecture
def create_cnn_model():
    cnn_model = models.Sequential([
        # Block 1: Initial feature extraction
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_DIM),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2: Deeper feature extraction
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Block 3: Complex feature learning
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Flattening and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])

    # Compile the model
    cnn_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return cnn_model

# Build and summarize the CNN model
model = create_cnn_model()
model.summary()

# Train the model
print("Starting model training...")
training_history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Plot the training progress
def visualize_training(history):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(train_acc) + 1)

    plt.figure(figsize=(14, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='orange', linestyle='--', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend(loc='upper left')
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='orange', linestyle='--', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Display the training progress
visualize_training(training_history)

# Evaluate the model on the test data
print("\nEvaluating the model on test data...")
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save the trained model
save_model_path = "./AER850_Project2.h5"
model.save(save_model_path)
print(f"Model successfully saved as {save_model_path}")
