import torch
import torch.nn as nn # layers (Conv2d, Linear) loss functions activation functions
import torch.optim as optim  # Imports optimization algorithms Adam SGD RMSprop
import os
import json

from sklearn.metrics import confusion_matrix, classification_report

from collections import Counter
from sklearn.model_selection import train_test_split


from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision.datasets import ImageFolder
from torchvision import transforms  # transforms are used for image preprocessing and augmentation.
from torch.utils.data import DataLoader, random_split

from torch.utils.data import Subset

# import the given model
from vehicle_classifier import SmallCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE=torch.device("cpu")

DATASET_PATH = "dataset"

BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
IMG_SIZE = 32
print("Initial Learning Rate:", LR)

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        path = os.path.join(root, file) #Create full path like: dataset/vehicle/img1.jpg
        try:
            img = Image.open(path)
            img.verify()
        except:
            print("Removing corrupt:", path)
            os.remove(path)

train_transform = transforms.Compose([
    # transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomResizedCrop(
    IMG_SIZE,
    scale=(0.6,1.0)
),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2
),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


# Load Dataset

dataset = ImageFolder(DATASET_PATH)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size


indices = list(range(len(dataset)))
targets = dataset.targets

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=targets,
    random_state=42
)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
# validation should not use augmentation

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# Compute Class Weights

train_targets = [dataset.targets[i] for i in train_idx]

class_counts = torch.bincount(torch.tensor(train_targets))

print("Class counts:", class_counts)

class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(DEVICE)


val_targets = [targets[i] for i in val_dataset.indices]
print("Val class counts:", torch.bincount(torch.tensor(val_targets)))


# -------- Weighted Sampler (NEW PART) --------

# train_targets = [targets[i] for i in train_dataset.indices]

# train_class_counts = torch.bincount(torch.tensor(train_targets))
# train_class_weights = 1.0 / train_class_counts.float()

# sample_weights = [train_class_weights[t] for t in train_targets]

# sampler = WeightedRandomSampler(
#     weights=sample_weights,
#     num_samples=len(sample_weights),
#     replacement=True
# )


# DataLoaders

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     sampler=sampler
# )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Initialize Model

model = SmallCNN(num_classes=5).to(DEVICE)
# Print model parameters
params = sum(p.numel() for p in model.parameters())
print("Total SmallCNN parameters:", params)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_val_acc = 0
best_epoch=1
best_val_preds = None
best_val_labels = None

# Training Loop
train_acc_list = []
val_acc_list = []

train_loss_list = []
val_loss_list = []


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

# validation
    model.eval()
    val_preds = []
    val_labels = []

    val_loss= 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = 100 * correct / total
    train_loss = train_loss/len(train_loader)
    val_loss=val_loss/len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("Current LR:", optimizer.param_groups[0]['lr'])
    print(f"Train Loss: {train_loss:.3f}")
    print(f"Val Loss: {val_loss:.3f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    print("------------------------------------------------")

    
    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_epoch = epoch + 1
        best_val_preds = val_preds
        best_val_labels = val_labels
        torch.save(model.state_dict(), "smallcnn_model.pth")

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


model_size = os.path.getsize("smallcnn_model.pth")/(1024*1024)
print("\nModel Size:", round(model_size,2),"MB")

metrics = {
    "train_acc": train_acc_list,
    "val_acc": val_acc_list,
    "train_loss": train_loss_list,
    "val_loss": val_loss_list
}

with open("smallcnn_metrics.json","w") as f:
    json.dump(metrics,f)
