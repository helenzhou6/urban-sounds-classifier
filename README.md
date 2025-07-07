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