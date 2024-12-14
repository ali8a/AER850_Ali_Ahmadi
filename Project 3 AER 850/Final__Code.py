# Install necessary libraries
#!pip install ultralytics

# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
from ultralytics import YOLO
import os
import glob

'''This code waas written on Google Collab'''
''' In Order to run the code you need to change the paths'''

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
base_path = "/content/drive/My Drive/Project 3 Data"
data_path = os.path.join(base_path, "data")
image_path = os.path.join(data_path, "motherboard_image.JPEG")
yml_file_path = os.path.join(data_path, "data.yaml")
test_path = os.path.join(data_path, "test")
valid_path = os.path.join(data_path, "valid")
train_path = os.path.join(data_path, "train")
evaluation_path = os.path.join(data_path, "evaluation")

# PART 1: Image Preprocessing
# Load the image
img_real = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Step 1: Preprocessing with Gaussian Blur and Adaptive Thresholding
img_blur = cv2.GaussianBlur(img_real, (47, 47), 4)  # Smooth the image
img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
img_thresh = cv2.adaptiveThreshold(
    img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 6
)  # Adaptive thresholding

# Step 2: Edge Detection and Contours
edges = cv2.Canny(img_thresh, 50, 300)  # Edge detection
edges_dilated = cv2.dilate(edges, None, iterations=10)  # Dilation to close gaps
contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 3: Mask the Largest Contour
contour_img = np.zeros_like(img_real)  # Blank image for the mask
largest_contour = max(contours, key=cv2.contourArea)  # Find the largest contour
cv2.drawContours(contour_img, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)  # Draw filled contour
masked_img = cv2.bitwise_and(img_real, contour_img)  # Apply the mask

# Display the results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB))
plt.axis("off")

# Thresholded Image
plt.subplot(2, 2, 2)
plt.title("Thresholded Image")
plt.imshow(img_thresh, cmap='gray')
plt.axis("off")

# Edges Dilated
plt.subplot(2, 2, 3)
plt.title("Edges Dilated")
plt.imshow(edges_dilated, cmap='gray')
plt.axis("off")

# Masked Image
plt.subplot(2, 2, 4)
plt.title("Masked Image (Largest Contour)")
plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()

# PART 2: Train YOLOv8 Model
model = YOLO("yolov8n.pt")  # Using a pretrained model

model.train(
    data=yml_file_path,       # Path to  dataset configuration file (data.yaml)
    epochs=30,                # Reduce epochs as needed
    batch=2,                  # Set batch size
    imgsz=928,                # Set image size
    project="/content/drive/My Drive/yolo_project/experiment13",  # Save training logs to Google Drive
    name="experiment13",       # Experiment name
)

# PART 3: Evaluate YOLOv8 Model
# Load the trained model
trained_model = YOLO("/content/drive/My Drive/yolo_project/experiment13/weights/best.pt")  # Update with path to trained weights

# Run predictions on evaluation images with adjusted NMS threshold
results = trained_model.predict(
    source=evaluation_path,  # Path to evaluation images
    save=True,               # Save the predictions
    imgsz=928,               # Image size
    conf=0.3,               # Confidence threshold (adjust if necessary)
    iou=0.3                 # Reduced IOU threshold to minimize overlapping boxes
)

# Display a few predictions
predicted_images = glob.glob("/content/runs/predict/*/*.jpg")  
