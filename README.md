# urban-sounds-classifier
Takes urban sounds (8k dataset) and classifies them into 10 categories (dog barking etc)
Has two models: CNN and Transformer Encoder from scratch

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
2. After we've created the dataset, the mel_spectogram is a numpy array of 1501 (time frames) by dimension of 128 (the frequency bins we've specified)
- We will then normalise the values (from -80 etc) to between 0 and 1, and also added padding to the maximum time frame of 1501. This will be done before it goes into the model

## CNN: AudioCNN1D model
1. **Convolutional layers**: First 1d convolution + maxpool + relu x3
- do 1D convolutions (see the slides) x3. It bakes in positionality and multiply the values by a scalar. 
- Max pooling x3 on the time frames (from 1501 -> 750 -> 375 -> 187)
- Relu for normalisation
- Above is done three times for feature extraction (of simpler, then more complex etc after each layer)
2. Then flatten - crucial step when transitioning from convolutional layers to fully connected (dense) layers in a CNN.
3. **Fully connected layers**
- Linear layer to project to lower dimension
4. **Classification layer**
- Linear layer to project to number of classes (number of categories we are classifying to)

## Transformer (encoder only) classifying model
1. Run 02...py file to preprocess, save it and upload to hugging face
2. Run 04... (03.. is for the test) that will run the Transformer Encoder
- NB we've having memory issues


## Running .ipynb
Install Jupyter VSCode plugin, and open the .ipynb and select Kernel to be the current .venv virtual environment. You can run each cell with control enter