import torch
import numpy as np

from collections import Counter

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from vehicle_classifier import VehicleClassifier
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

import subprocess
import time
import os
import psutil


# ============================
# MEMORY FUNCTION
# ============================

def print_memory(stage):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**2)
    print(f"[MEMORY] {stage}: {mem:.2f} MB")

TEST_PATH = "test"

# preprocessing
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


dataset = ImageFolder(TEST_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

print("\n==============================")
print("TEST DATASET INFO")
print("==============================")

print("Class mapping:", dataset.class_to_idx)
print("Total test samples:", len(dataset))

print_memory("After loading dataset")


# SMALL CNN EVALUATION

cnn_model = VehicleClassifier("smallcnn_model.pth")

y_true = []
y_pred = []

for i,(img,label) in enumerate(loader):

    img_path = dataset.samples[i][0]

    pred = cnn_model.predict(img_path)

    y_true.append(label.item())
    y_pred.append(pred)

print("\n   SMALL CNN RESULTS   ")

acc = accuracy_score(y_true,y_pred)

print("Accuracy:",acc)
print("Test Error:",1-acc)

print("\nConfusion Matrix")
cm=confusion_matrix(y_true,y_pred)
print(cm)

print("\nClassification Report")
print(classification_report(y_true,y_pred))

print("\nPer-Class Accuracy")

for i in range(cm.shape[0]):

    acc = cm[i,i] / cm[i].sum()

    print(f"Class {i} accuracy: {acc:.3f}")


# ==============================
# PREDICTION DISTRIBUTION
# ==============================

print("\nPrediction Distribution")

pred_counts = Counter(y_pred)

for k,v in pred_counts.items():
    print(f"Class {k}: {v} predictions")


print_memory("After SmallCNN evaluation")
# MOBILENET EVALUATION

# device = torch.device("cpu")

print("\n\n==============================")
print("EVALUATING MOBILENET")
print("==============================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel,5)

mobilenet.load_state_dict(torch.load("mobilenet_model.pth",map_location=device))

mobilenet.eval()


mobilenet_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

y_true_mn = []
y_pred_mn = []

for i,(img,label) in enumerate(loader):

    img_path = dataset.samples[i][0]

    image = Image.open(img_path).convert("RGB")
    tensor = mobilenet_transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = mobilenet(tensor)
        _,pred = torch.max(outputs,1)

    y_true_mn.append(label.item())
    y_pred_mn.append(pred.item())

print("\n    MOBILENET RESULTS    ")

acc = accuracy_score(y_true_mn,y_pred_mn)

print("Accuracy:",acc)
print("Test Error:",1-acc)

print("\nConfusion Matrix")
cm_mn=confusion_matrix(y_true_mn,y_pred_mn)
print(cm_mn)

print("\nClassification Report")
print(classification_report(y_true_mn,y_pred_mn))


print("\nPer-Class Accuracy")

for i in range(cm_mn.shape[0]):

    acc = cm_mn[i,i] / cm_mn[i].sum()

    print(f"Class {i} accuracy: {acc:.3f}")


print("\nPrediction Distribution")

pred_counts = Counter(y_pred_mn)

for k,v in pred_counts.items():
    print(f"Class {k}: {v} predictions")


print_memory("After MobileNet evaluation") 