# CNN-Based Vehicle Classification under Memory Constraints

This project implements a lightweight CNN-based image classifier that categorizes street images into five classes.

---

## Class Index Mapping

The classifier outputs an integer class index corresponding to the predicted class.

0 → Bus
1 → Truck
2 → Car
3 → Bike
4 → None

---

## Project Pipeline

The complete pipeline of the project is:

Dataset Preparation → Model Training → Model Evaluation → Visualization → Inference

---

## Models Implemented

Two models were developed:

### 1. SmallCNN

A lightweight convolutional neural network used as a baseline model.

### 2. MobileNetV2 (Transfer Learning)

A transfer learning model using MobileNetV2 with width multiplier = 0.5.

The final model uses MobileNetV2.

---

## Dataset Preparation

Dataset images are downloaded from ImageNet using specific WordNet IDs.

Run:

```bash
python getdataset.py
python select_none_images
```

To:

* Download ImageNet classes and None class images
* Extract images
* Organize them into dataset folders

---

## Training

Two models are trained separately.

Train SmallCNN

```bash
python train_smallcnn.py
```

Train MobileNetV2

```bash
python train_mobilenet.py
```

Saved models:

```
smallcnn_model.pth
mobilenet_model.pth
```

---

## Evaluation

Models are evaluated on a test dataset.

Run:

```bash
python evaluate_model.py
```

Results are saved as:

```
smallcnn_test_results.json
mobilenet_test_results.json
```

---

## Visualization

Training curves and evaluation results are visualized using:

```bash
python visualizations.py
```

---

## Inference (VehicleClassifier)

The VehicleClassifier class loads the trained model and performs inference on a single image (for testing purposes).

Example:

```python
from vehicle_classifier import VehicleClassifier

classifier = VehicleClassifier("student_model.pth")

prediction = classifier.predict("image.jpg")

print(prediction)
```

### Inference steps

* Load image
* Resize to 224 × 224
* Convert to tensor
* Apply normalization
* Run MobileNetV2 model
* Return predicted class index

---

## Required Python Libraries

The project requires the following libraries:

```
torch
torchvision
Pillow
numpy
matplotlib
seaborn
scikit-learn
```

Install them using:

```bash
pip install torch torchvision pillow numpy matplotlib seaborn scikit-learn
```

---

## Running the Full Pipeline

Run the entire pipeline using:

```bash
python run_all.py
```

This will:

* Train SmallCNN
* Train MobileNetV2
* Evaluate models for the test data
* Generate visualizations for training, validation and test data used

---

## Model Size

Final submitted model:

```
student_model.pth
```

Model size:

```
4.42 MB
```

The model satisfies the assignment requirement of less than 5 MB.

---

## Project Structure

```
project/

dataset/
test/

train_smallcnn.py
train_mobilenet.py
evaluate_model.py
visualizations.py
vehicle_classifier.py
run_all.py
getdataset.py

smallcnn_model.pth
mobilenet_model.pth
student_model.pth

plots/
report.pdf
README.md
```
