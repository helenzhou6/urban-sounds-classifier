from datasets import load_dataset
import numpy as np
import pickle
import os

NUM_SAMPLES = 100

train_us_dataset = load_dataset("danavery/urbansound8K", split="train")

samples = []
for sample_row in train_us_dataset.select(range(NUM_SAMPLES)):
    data = {
        "audio": np.array(sample_row["audio"]["array"]),
        "target_label": sample_row["class"]
    }
    samples.append(data)

if not os.path.exists("data"):
    os.makedirs("data")

with open("data/urbansound_samples.pkl.gz", "wb") as file:
    pickle.dump(samples, file)

