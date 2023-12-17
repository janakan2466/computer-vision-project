# A Computer Vision Project: Hand Gesture Recognition Systems

## Abstract
In the realm of human-computer interaction, the quest for more accessible communication methods has driven the development of Hand Gesture Recognition Systems. This project aims to contribute to this advancement by creating a detection system capable of recognizing hand forms through a combination of carefully crafted preprocessing algorithms and the utilization of Convolutional Neural Networks (CNNs) for image classification.

## Method
The experimental methodology employed in this project focuses on the American Sign Language (ASL) Alphabet dataset. The following steps outline our approach:

1. **Import and Resize Images:** Standardize images by resizing.
2. **Convert to Grayscale:** Extract feature intensity values through grayscale conversion.
3. **Gaussian Noise Removal:** Enhance reliability by smoothing edges through noise reduction.
4. **Otsu’s Thresholding and Inverted Binary Thresholding:** Analyze intensity distribution for effective image segmentation.
5. **Canny Edge Detection:** Identify hand boundaries in the frame.
6. **CNN for Gesture Classification:** Pair gestures using a Convolutional Neural Network.
7. **Evaluate Metrics:** Assess model performance using metrics such as Accuracy, Precision, F1 Score, ROC Curve, and Confusion Matrix.

## Example
To illustrate our methodology, consider the ASL Alphabet dataset, where each gesture represents a distinct sign in American Sign Language. The algorithm processes these gestures, identifies hand forms, and utilizes CNN to classify and pair them accurately.

## Dataset & References
- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data) - Image dataset for alphabets in the American Sign Language.

