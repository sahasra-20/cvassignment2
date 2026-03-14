import os
import random
import shutil

import os

folder = "noneimages/non-vehicles"

images = [f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg"))]

print("Total images:", len(images))

src = "noneimages/non-vehicles"
dst = "dataset/none"

os.makedirs(dst, exist_ok=True)

images = [f for f in os.listdir(src) if f.endswith((".png",".jpg",".jpeg",".JPEG"))]

selected = random.sample(images, 2500)

for img in selected:
    shutil.copy(os.path.join(src, img), os.path.join(dst, img))

print("2500 random images copied to dataset/none")