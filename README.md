# CNN-Based Vehicle Classification under Memory Constraints

This project implements a Convolutional Neural Network (CNN) based image classifier that categorizes real world images into five classes.

---

## Class Index Mapping

The classifier returns an integer class index corresponding to the predicted category.

0 → Bus
1 → Truck
2 → Car
3 → Bike
4 → None

The model satisfies a **memory constraint of less than 5 MB**.

---

## Models Implemented

Two models were implemented:

1. **SmallCNN** – lightweight CNN baseline
2. **MobileNetV2** – transfer learning model (MobileNetV2 pretrained on ImageNet)

The final model uses **MobileNetV2**.

The model weights are stored in:

```
student_model.pth
```

---

## VehicleClassifier

The `VehicleClassifier` class performs inference on a single image and returns the predicted class index (0–4).

The class expects the **path to an image file** as input.

---

## Example

```python
from vehicle_classifier import VehicleClassifier

classifier = VehicleClassifier("student_model.pth")

prediction = classifier.predict("image.jpg")

print(prediction)
```

---

## Inference Pipeline

During inference, the following preprocessing steps are applied automatically:

1. The input image is loaded and converted to **RGB format**
2. The image is **resized to 224 × 224**
3. The image is **converted to a tensor**
4. **ImageNet normalization** is applied
5. The processed image is passed through the trained **MobileNetV2 model**

The predicted **class index (0–4)** is returned.

---

## Required Python Libraries

The following libraries are required to run inference:

```
torch
torchvision
Pillow
numpy
```

Install them using:

```bash
pip install torch torchvision pillow numpy
```

---

## Model Size

```
Model Size: 4.42 MB
```
