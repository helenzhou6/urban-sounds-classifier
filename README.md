# urban-sounds-classifier
Takes urban sounds and classifies them

## Pre-requisities 
- Python 3.10
- uv - install via https://github.com/astral-sh/uv
- wandb log in details and be added to the project - https://wandb.ai/site or in remote terminal 

## The dataset
See https://huggingface.co/datasets/danavery/urbansound8K which is based off https://urbansounddataset.weebly.com/urbansound8k.html
- Only has training dataset, 8732 rows
- urban sounds from 10 classes (dataset.class): air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music
    - Also has classID that is 0 to 9 and maps to a class
- Note: The docs mention not to shuffle, and also to "perform 10-fold cross validation using the provided folds and report the average score."

## Running - create dataset
1. Run `uv run src/dataset.py` or create the dataset
- This will download the Urban Sounds 8k dataset, create a log mel spectogram and generate a new dataset that includes that (minus the audio, to make it smaller size).
- This will upload to Hugging Face. You'll need to create a hugging face profile, create a new dataset, and then generate an Access Token (API key) with write permissions, and put `HF_TOKEN` in `.env`.
- It has already been run, and uploaded to: https://huggingface.co/datasets/EthanGLEdwards/urbansounds_melspectrograms so noone else needs to use it
2. After we've created the dataset, the mel_spectogram is a numpy array of 1501 (time frames) by dimension of 128 (the frequency bins)
- We will then normalise the values (from -80 etc) to between 0 and 1, and also added padding to the maximum time frame of 1501.

## Running .ipynb
Install Jupyter VSCode plugin, and open the .ipynb and select Kernel to be the current .venv virtual environment. You can run each cell with control enter