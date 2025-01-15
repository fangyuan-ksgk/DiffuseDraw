# Augment Dataset 
from utils import augment_dataset
from datasets import load_dataset

dataset = load_dataset("Ksgk-fy/kanji-dataset")


expanded_dataset = dataset.map(augment_dataset, num_proc=10)

expanded_dataset.push_to_hub("Ksgk-fy/augmented-kanji-dataset")