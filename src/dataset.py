from datasets import load_dataset

train_us_dataset = load_dataset("danavery/urbansound8K", split="train")

print(train_us_dataset.column_names)
for i in range(3):
    print(train_us_dataset[i])
