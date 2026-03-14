import torch
import torch.nn as nn # layers (Conv2d, Linear) loss functions activation functions
import torch.optim as optim  # Imports optimization algorithms Adam SGD RMSprop

from torchvision.datasets import ImageFolder
from torchvision import transforms  # transforms are used for image preprocessing and augmentation.
from torch.utils.data import DataLoader, random_split

# import the given model
from vehicle_classifier import SmallCNN


BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
IMG_SIZE = 32


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE=torch.device("cpu")
DATASET_PATH = "dataset"

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1,0.1)
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

dataset = ImageFolder(DATASET_PATH, transform=train_transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size


train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# validation should not use augmentation
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

# Initialize Model

model = SmallCNN(num_classes=5).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0


# Training Loop

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

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total


    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.3f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print("------------------------------------------------")


    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(model.state_dict(), "student_model.pth")

        print("Best model saved!")

print("Training complete")

from vehicle_classifier import VehicleClassifier, CLASS_IDX

classifier = VehicleClassifier("student_model.pth")

image_path = "test.jpg"

pred = classifier.predict(image_path)

print("Predicted index:", pred)
print("Predicted label:", CLASS_IDX[pred])
