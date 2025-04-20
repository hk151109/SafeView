# SafeView: NSFW Content Detection Model & Chrome Extension

SafeView is an advanced NSFW content detection solution consisting of two components:

- **NSFW Content Detection Model**: A deep learning model built using MobileNetV3-Small, fine-tuned to detect and classify NSFW content into five categories: Drawing, Neutral, Hentai, Porn, and Sexy.

- **Chrome Extension**: A comprehensive extension designed to integrate the SafeView model into web browsing, providing real-time NSFW content detection for images, videos, and user-uploaded content.

Together, these components create a robust content moderation solution suitable for real-time use on any web platform.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Chrome Extension Installation](#chrome-extension-installation)
- [Model Setup and Usage](#model-setup-and-usage)
- [Chrome Extension Usage](#chrome-extension-usage)
- [Model Details](#model-details)

## Project Overview
SafeView offers an all-in-one solution for NSFW content detection across both backend (model) and frontend (browser extension) applications. It is designed to ensure safe and user-friendly experiences by detecting inappropriate content in real time.

**Components:**
- **NSFW Detection Model**: A fine-tuned deep learning model built on MobileNetV3-Small to classify images into five categories: Drawing, Neutral, Hentai, Porn, and Sexy.
- **Chrome Extension**: A browser extension that integrates the SafeView model, enabling real-time NSFW detection for web content.

## Features
- **Real-Time NSFW Detection**: Both the model and Chrome extension are optimized for quick, low-latency content classification.
- **Multi-Class Classification**: Classifies images and videos into five distinct categories: Drawing, Neutral, Hentai, Porn, and Sexy.
- **User-Friendly Chrome Extension**: Provides a seamless browsing experience by alerting users when NSFW content is detected.
- **Extensive Model Optimizations**: The model has been fine-tuned and optimized for high accuracy with a reduced false-positive rate.
- **Scalable for Multiple Platforms**: While currently deployed for web use, the model can be adapted to other platforms, including mobile applications.

## Chrome Extension Installation
To install and use the SafeView Chrome Extension:

1. **Download the Extension Files**: Clone the repository or download the extension files from the provided release folder.

2. **Load the Extension in Chrome**:
   - Open Chrome and navigate to `chrome://extensions/`.
   - Enable Developer mode (toggle in the top-right corner).
   - Click `Load unpacked` and select the directory containing the extension files.

3. **Access the Extension**:
   - Once installed, the SafeView icon will appear in your browser's toolbar.
   - Click the icon to enable or disable NSFW content detection while browsing.
   - The extension will automatically detect NSFW images and blur if any content falls under the NSFW categories.

## Model Setup and Usage
SafeView's model is designed for server-side integration, allowing you to deploy it as a backend service for any web application that requires NSFW detection.

### Input
- Image (JPEG, PNG, or any standard image format)

### Output
- Probability distribution across five categories: Drawing, Neutral, Hentai, Porn, and Sexy.

### Installation
1. Install Python 3.7+ and the necessary dependencies by setting up a virtual environment.
2. Once installed, load the model and use it to perform content classification.

```python
import tensorflow as tf
import numpy as np
import cv2

# Load model and preprocess image
model = tf.keras.models.load_model('path_to_saved_model')
image = cv2.imread('path_to_image.jpg')

# Perform classification
predictions = model.predict(image)
```

## Chrome Extension Usage
The Chrome extension integrates directly with the SafeView NSFW detection model. Here's how it works:

- **Real-Time Detection**: The extension detects images and videos on the current webpage in real-time. Whenever NSFW content is detected, it sends a notification to the user.

The extension is lightweight, and once installed, it works seamlessly with no additional setup required.

## Model Details
SafeView's model is built on MobileNetV3-Small and fine-tuned for detecting NSFW content. The model classifies images into the following categories:

- Drawing
- Neutral
- Hentai
- Porn
- Sexy

### Model Enhancements
- **Fine-tuning for Accuracy**: The model has been optimized using the LSPD dataset, improving its ability to detect NSFW content with minimal false positives.
- **Optimized for Speed**: The model is optimized to work in real-time environments, ensuring quick response times for both the backend service and Chrome extension.
