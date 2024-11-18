import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib

# Set the backend explicitly (to ensure plots are displayed)
matplotlib.use('TkAgg')

# Load the trained model
model_path = "./AER850_Project2_Model10.h5"
print("Loading the model...")
model = load_model(model_path, compile=False)

# Class names for the prediction
class_names = ['crack', 'missing-head', 'paint-off']

# Helper function to preprocess a test image
def load_and_preprocess_image(img_path, target_size=(500, 500)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input
    return img_array

# Function to make predictions using the model
def make_prediction(model, img_array, class_names):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_names[predicted_class]
    predicted_probabilities = predictions[0] * 100  # Convert to percentage
    return predicted_label, predicted_probabilities

# Function to display the test image and prediction results
def display_prediction(img_path, true_label, predicted_label, predicted_probabilities, class_names):
    img = image.load_img(img_path, target_size=(500, 500))
    plt.imshow(img)
    plt.axis('off')

    # Display the true label at the top
    plt.title(f"True Label: {true_label.capitalize()}\nPredicted Label: {predicted_label}", fontsize=14, color='blue')

    # Create a text annotation with predicted probabilities at the bottom center
    prob_text = " | ".join([f"{class_name}: {prob:.1f}%" for class_name, prob in zip(class_names, predicted_probabilities)])
    plt.text(
        0.5, -0.15, prob_text,
        color='green', fontsize=12, ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.7),
        transform=plt.gca().transAxes
    )

    # Display the plot and save it as an image file
    plt.tight_layout()
    plt.show(block=True)
    plt.savefig("prediction_output.png")
    print("Plot saved as 'prediction_output.png'")

# Main function to test the model with the specified test images
def test_model():
    test_images = {
        "crack": "./Data/test/crack/test_crack.jpg",
        "missing-head": "./Data/test/missing-head/test_missinghead.jpg",
        "paint-off": "./Data/test/paint-off/test_paintoff.jpg"
    }

    for true_label, img_path in test_images.items():
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        print(f"\nTesting model with image: {img_path}")
        img_array = load_and_preprocess_image(img_path)
        predicted_label, predicted_probabilities = make_prediction(model, img_array, class_names)
        print(f"True Label: {true_label.capitalize()}, Predicted Label: {predicted_label}")
        display_prediction(img_path, true_label, predicted_label, predicted_probabilities, class_names)

# Run the testing function
if __name__ == "__main__":
    test_model()
