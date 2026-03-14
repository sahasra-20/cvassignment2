import os
import requests
import tarfile
import shutil

# -------------------------
# CLASS MAPPING
# -------------------------
mapping = {
    "0_bus": ["n04146614","n04487081"], 
    "1_truck": ["n04467665","n04467332","n04461696"],
    "2_car": ["n02958343","n03594945","n03769881","n04037443","n04285008","n03100240"],
    "3_bike": ["n03790512","n02835271"]
}

BASE_URL = "https://image-net.org/data/winter21_whole"

tar_folder = "tar_files"
extract_folder = "extracted"
dataset_folder = "dataset"

os.makedirs(tar_folder, exist_ok=True)
os.makedirs(extract_folder, exist_ok=True)
os.makedirs(dataset_folder, exist_ok=True)


# -------------------------
# DOWNLOAD FUNCTION
# -------------------------
def download_tar(wnid):

    url = f"{BASE_URL}/{wnid}.tar"
    tar_path = os.path.join(tar_folder, f"{wnid}.tar")

    print(f"Downloading {wnid}...")

    r = requests.get(url, stream=True)

    if r.status_code != 200:
        print(f"Failed to download {wnid}")
        return False

    with open(tar_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return True


# -------------------------
# TAR VALIDATION
# -------------------------
def is_tar_valid(path):

    try:
        with tarfile.open(path) as tar:
            tar.getmembers()
        return True
    except:
        return False


# -------------------------
# STEP 1 DOWNLOAD
# -------------------------
print("\nChecking / Downloading TAR files\n")

for class_name, wnids in mapping.items():

    for wnid in wnids:

        tar_path = os.path.join(tar_folder, f"{wnid}.tar")

        # if tar exists, validate
        if os.path.exists(tar_path):

            if is_tar_valid(tar_path):
                print(f"{wnid}.tar OK")
                continue
            else:
                print(f"{wnid}.tar corrupted, re-downloading")
                os.remove(tar_path)

        download_tar(wnid)

print("\nDownload step complete\n")


# -------------------------
# STEP 2 EXTRACT
# -------------------------
print("Extracting TAR files\n")

for file in os.listdir(tar_folder):

    if not file.endswith(".tar"):
        continue

    wnid = file.replace(".tar","")

    tar_path = os.path.join(tar_folder, file)
    extract_path = os.path.join(extract_folder, wnid)

    if os.path.exists(extract_path):
        print(f"{wnid} already extracted")
        continue

    os.makedirs(extract_path, exist_ok=True)

    print(f"Extracting {file}")

    with tarfile.open(tar_path) as tar:
        tar.extractall(extract_path)

print("\nExtraction complete\n")


# -------------------------
# STEP 3 ORGANIZE
# -------------------------
print("Organizing dataset\n")

for class_name, wnids in mapping.items():

    class_dir = os.path.join(dataset_folder, class_name)
    os.makedirs(class_dir, exist_ok=True)

    for wnid in wnids:

        src_dir = os.path.join(extract_folder, wnid)

        if not os.path.exists(src_dir):
            continue

        for file in os.listdir(src_dir):

            if file.lower().endswith((".jpg",".jpeg",".png")):

                src = os.path.join(src_dir, file)

                # avoid overwrite
                dst = os.path.join(class_dir, f"{wnid}_{file}")

                try:
                    shutil.move(src, dst)
                except:
                    pass


print("\nDataset ready!\n")

print("Final structure:")
print("dataset/0_bus")
print("dataset/1_truck")
print("dataset/2_car")
print("dataset/3_bike")