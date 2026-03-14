import subprocess
import time
import os
import psutil

def print_memory(stage):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**2)
    print(f"[MEMORY] {stage}: {mem:.2f} MB")

def run_script(script):

    print("\n===================================")
    print(f"Running: {script}")
    print("===================================")

    start = time.time()

    subprocess.run(["python", script])

    end = time.time()

    print(f"\nExecution time: {round(end-start,2)} seconds")

    print_memory(f"After running {script}")

print_memory("Start")


# ----------------------------
# TRAIN SMALL CNN
# ----------------------------

run_script("train_smallcnn.py")


# ----------------------------
# TRAIN MOBILENET
# ----------------------------

run_script("train_mobilenet.py")


# ----------------------------
# RUN EVALUATION
# ----------------------------

run_script("evaluate_model.py")

print("\n==============================")
print("MODEL SIZE CHECK")
print("==============================")

if os.path.exists("smallcnn_model.pth"):

    size = os.path.getsize("smallcnn_model.pth")/(1024*1024)

    print("SmallCNN model size:",round(size,2),"MB")

if os.path.exists("mobilenet_model.pth"):

    size = os.path.getsize("mobilenet_model.pth")/(1024*1024)

    print("MobileNet model size:",round(size,2),"MB")

print_memory("End of pipeline")