import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models # Used to load pretrained models.
from torch.utils.data import DataLoader, random_split
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

BATCH_SIZE = 32
EPOCHS = 15
IMG_SIZE = 224
LR = 0.0003
print("Initial Learning Rate:", LR)

DATASET_PATH = "dataset"
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        path = os.path.join(root, file)
        try:
            img = Image.open(path)
            img.verify()
        except:
            print("Removing corrupt:", path)
            os.remove(path)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2,contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])



val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

from torch.utils.data import WeightedRandomSampler

# Load Dataset

dataset = ImageFolder(DATASET_PATH)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# validation should not use augmentation
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform


print("Training samples:", train_size)
print("Validation samples:", val_size)


# Compute Class Weights

targets = dataset.targets
class_counts = torch.bincount(torch.tensor(targets))

print("Class counts:", class_counts)

class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(DEVICE)

print("Class weights:", class_weights)


# -------- Weighted Sampler (NEW PART) --------

train_targets = [targets[i] for i in train_dataset.indices]

train_class_counts = torch.bincount(torch.tensor(train_targets))
train_class_weights = 1.0 / train_class_counts.float()

sample_weights = [train_class_weights[t] for t in train_targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Initialize Model

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

for param in model.features[:-3].parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.last_channel, 5)
model = model.to(DEVICE)

params = sum(p.numel() for p in model.parameters())
print("Total MobileNet parameters:", params)


criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_val_acc = 0
best_epoch=1
best_val_preds = None
best_val_labels = None

for epoch in range(EPOCHS):

    model.train()

    correct = 0
    total = 0
    train_loss = 0

    for images,labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        _,predicted = torch.max(outputs,1)

        

        total += labels.size(0)
        correct += (predicted==labels).sum().item()

    train_acc = 100*correct/total

    # validation

    model.eval()
    val_preds = []
    val_labels = []

    correct = 0
    total = 0

    with torch.no_grad():

        for images,labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _,predicted = torch.max(outputs,1)
            

            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = 100 * correct / total
    
    train_loss = train_loss
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("Current LR:", optimizer.param_groups[0]['lr'])
    print(f"Train Loss: {train_loss:.3f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print("------------------------------------------------")


    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_epoch = epoch + 1
        best_val_preds = val_preds
        best_val_labels = val_labels
        torch.save(model.state_dict(), "mobilenet_model.pth")

        print("Best model saved!")
    scheduler.step()

print("Training complete")
print("\n====================================")
print("BEST MODEL VALIDATION RESULTS")
print("====================================")

print("Best Epoch:", best_epoch)
print("Best Validation Accuracy:", round(best_val_acc,2),"%")

cm = confusion_matrix(best_val_labels, best_val_preds)

print("\nConfusion Matrix")
print("(Rows = Actual, Columns = Predicted)")
print(cm)

print("\nClassification Report")
print(classification_report(best_val_labels, best_val_preds))

print("\nPer-Class Accuracy")
for i in range(cm.shape[0]):

    acc = cm[i,i] / cm[i].sum()

    print(f"Class {i} accuracy: {acc:.3f}")

print("\nPrediction Distribution")

pred_counts = Counter(best_val_preds)

for k,v in pred_counts.items():

    print(f"Class {k}: {v} predictions")

model_size = os.path.getsize("mobilenet_model.pth")/(1024*1024)

print("\nModel Size:", round(model_size,2),"MB")

print_memory("After training finished")

