import cv2
import numpy as np

# Step 1: Read User Input Image
image = cv2.imread('assets/hand_sign.jpg')

# Step 2: Resize Image to appropriate Pixels
desired_size = (600, 600)
resized_image = cv2.resize(image, desired_size)

# Display resized image
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)

# Step 3: Convert to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Display grayscale image
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(0)

# Step 4: Noise removal (Gaussian blur)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Display blurred image
cv2.imshow('Blurred Image', blurred)
cv2.waitKey(0)

# Step 5: Otsu's thresholding and inverted binary thresholding
_, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Display thresholded image
cv2.imshow('Thresholded Image', thresholded)
cv2.waitKey(0)

# Step 6: Canny edge detection
edges = cv2.Canny(thresholded, 50, 150)  # You can adjust the threshold values if needed

# Display edges (Canny) image
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)

# Step 7: Contour extraction and ROI extraction
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Find the bounding box of the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Draw the bounding box on the original image
image_with_bbox = cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with bounding box
cv2.imshow('Image with Bounding Box', image_with_bbox)
cv2.waitKey(0)

# Extract the ROI from the original image based on the bounding box coordinates
roi = resized_image[y:y + h, x:x + w]

# Display the extracted ROI
cv2.imshow('ROI', roi)
cv2.waitKey(0)

cv2.destroyAllWindows()