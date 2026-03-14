import os
import shutil

src = "imagenet_subset"
dst = "dataset"

mapping = {
    "Bus": ["n02892201"],
    "Truck": ["n04467665"],
    "Car": ["n02958343","n03594945","n03769881","n04037443","n04285008"],
    "Bike": ["n03790512","n02835271"]
}

for cls in mapping:
    os.makedirs(os.path.join(dst, cls), exist_ok=True)

for file in os.listdir(src):

    if file.endswith(".JPEG"):

        synset = file.split("_")[0]

        for cls in mapping:
            if synset in mapping[cls]:

                src_path = os.path.join(src, file)
                dst_path = os.path.join(dst, cls, file)

                shutil.copy(src_path, dst_path)

print("Dataset organized successfully!")