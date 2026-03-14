import os
from torchvision.datasets import ImageFolder
from torchvision import transforms

dataset_path = "dataset"

# count images in each folder
for folder in os.listdir(dataset_path):
    path = os.path.join(dataset_path, folder)
    
    if os.path.isdir(path):
        count = len(os.listdir(path))
        print(f"{folder}: {count} images")


# create dataset object
dataset = ImageFolder(dataset_path, transform=transforms.ToTensor())

# print class index mapping
print("Class to index mapping:", dataset.class_to_idx)