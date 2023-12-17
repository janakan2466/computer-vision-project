#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from google.colab import drive
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate

# # Install kaggle and upload JSON file
# !pip install -q kaggle
# from google.colab import files
# files.upload()

# # Unzip dataset
# !mkdir ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d grassknoted/asl-alphabet
# !unzip asl-alphabet.zip

# # Mount Google Drive
# drive.mount("/content/drive")

# Define constants
MAX_IMAGES_PER_LETTER = 100
DESIRED_SIZE = (320, 320)
TEST_SIZE = 0.3

# Set up a dictionary to count images per letter
image_count_per_letter = defaultdict(int)

# Set the start directory
start_directory = "./asl_alphabet_train"

# List to store the names of subdirectories
subdirectories = []

# Subdirectories to skip
skip_dirs = {"del", "nothing", "space"}


# Function to process and plot the final image
def process_and_plot_final_image(path):
    # Load the image
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Resize the image
    resized_image = cv2.resize(image, DESIRED_SIZE)

    # Convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Canny edge detection
    final_image = cv2.Canny(thresholded, 50, 150)

    # Normalize the final image
    normalized_image = final_image / 255.0

    # Display the final image
    plt.imshow(normalized_image, cmap="gray")
    plt.title("Final Processed Image")
    plt.xticks([]), plt.yticks([])  # Hide tick marks
    plt.show()


# Function to validate and plot images
def validate_26_images(predictions_array, true_label_array, img_array):
    # Array for pretty printing and figure size
    class_names = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]
    plt.figure(figsize=(15, 15))

    for i in range(1, 27):
        # Assign variables
        prediction = predictions_array[i]
        true_label = true_label_array[i]
        img = img_array[i]

        # Plot in a good way
        plt.subplot(7, 4, i)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(prediction)

        # Change color of title based on good prediction or not
        color = "blue" if predicted_label == true_label else "red"

        plt.xlabel(
            "Predicted: {} {:2.0f}% (True: {})".format(
                class_names[predicted_label],
                100 * np.max(prediction),
                class_names[true_label],
            ),
            color=color,
        )
    plt.show()


# Load image paths and preprocess images
imagepaths = []
for root, dirs, files in os.walk(start_directory, topdown=False):
    for dir in dirs:
        if dir in skip_dirs:
            continue
        subdirectory = os.path.join(root, dir)
        subdirectories.append(subdirectory)
    for name in files:
        if name.lower().endswith(".jpg"):
            letter = os.path.basename(root)
            if letter in skip_dirs:
                continue
            if image_count_per_letter[letter] < MAX_IMAGES_PER_LETTER:
                path = os.path.join(root, name)
                imagepaths.append(path)
                image_count_per_letter[letter] += 1
                if image_count_per_letter[letter] == MAX_IMAGES_PER_LETTER:
                    continue

# Print information about collected image paths
print("Number of image paths:", len(imagepaths))
print(f"Collected image paths, with up to {MAX_IMAGES_PER_LETTER} images per letter.")

# Print subdirectories
print("Subdirectories found:")
for subdir in subdirectories:
    print(subdir)

# Process and plot a sample image
process_and_plot_final_image(imagepaths[2])

# Mount Google Drive
drive.mount("/content/drive")

# Apply computer vision algorithms to all pictures in the dataset
X = []  # Image data
y = []  # Labels
skipped_images = 0

# Loop through image paths to load images and labels into arrays
for path in imagepaths:
    # Load the image
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Resize the image to the desired size
    resized_image = cv2.resize(image, DESIRED_SIZE)

    # Convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Canny edge detection
    final_image = cv2.Canny(thresholded, 50, 150)

    # Add the processed image to the dataset
    X.append(final_image)

    # Process the label in the image path
    filename = os.path.basename(path)
    label_part = filename.split(".")[0]

    label_mapping = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "J": 9,
        "K": 10,
        "L": 11,
        "M": 12,
        "N": 13,
        "O": 14,
        "P": 15,
        "Q": 16,
        "R": 17,
        "S": 18,
        "T": 19,
        "U": 20,
        "V": 21,
        "W": 22,
        "X": 23,
        "Y": 24,
        "Z": 25,
    }

    if len(label_part) > 1 and label_part[1:].isdigit():
        label_char = label_part[0]
        label = label_mapping.get(label_char, -1)
    else:
        label = label_mapping.get(label_part, -1)

    y.append(label)

# Convert X and y into np.array to speed up future processing
X = np.array(X, dtype="uint8")
X = X / 255.0  # Normalize pixel values to be between 0 and 1
X = X.reshape(
    len(imagepaths) - skipped_images, DESIRED_SIZE[0], DESIRED_SIZE[1], 1
)  # Adjust for 1 channel

y = np.array(y)

print("Images loaded: ", len(X))
print("Labels loaded: ", len(y))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)


# Define a learning rate schedule
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 30:
        lr *= 0.5
    if epoch > 60:
        lr *= 0.5
    return lr


# Create a LearningRateScheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Create and compile the model with corrected input shape
model = Sequential(
    [
        Conv2D(32, (5, 5), activation="relu", input_shape=(320, 320, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(26, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Filter out samples with '-1' labels
valid_train_samples = y_train != -1
valid_test_samples = y_test != -1

X_train_filtered = X_train[valid_train_samples]
y_train_filtered = y_train[valid_train_samples]

X_test_filtered = X_test[valid_test_samples]
y_test_filtered = y_test[valid_test_samples]

# Train the model with the filtered datasets and the learning rate scheduler
model.fit(
    X_train_filtered,
    y_train_filtered,
    epochs=20,
    batch_size=64,
    verbose=2,
    validation_data=(X_test_filtered, y_test_filtered),
    callbacks=[lr_scheduler],
)

# Save the entire model to an HDF5 file
model.save("project_ASL_recognition.keras")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert predictions from probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate a classification report
report = classification_report(y_test, y_pred_classes, target_names=class_names)

print("Classification Report:\n", report)

# Plot validation images
validate_26_images(y_pred, y_test, X_test)

# Assuming y_test contains the true labels and y_pred contains the predicted labels
confusion = confusion_matrix(y_test, y_pred)

# Create a DataFrame for the confusion matrix with labeled columns and indices
confusion_df = pd.DataFrame(
    confusion,
    columns=[f"Predicted {class_name}" for class_name in class_names],
    index=[f"Actual {class_name}" for class_name in class_names],
)

# Display the confusion matrix as a formatted table
print(tabulate(confusion_df, headers="keys", tablefmt="fancy_grid"))
