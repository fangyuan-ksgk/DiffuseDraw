# from src.utils import prepare_kanji_dict

# prepare_kanji_dict(regenerate=True)


from datasets import load_dataset
from tqdm import tqdm
import os 
os.makedirs("data/kanji_expand_image", exist_ok=True)

# load dataset
dataset = load_dataset("Ksgk-fy/expanded-kanji-dataset")
# save images
for idx, img in enumerate(tqdm(dataset['train']['image'])):
    img.save(f"data/kanji_expand_image/kanji_{idx}.png")
    
# Pipeline of Training (Conceptual)
# - 1. DreamBooth Concept Embedding 
# - 2. Text2Image Caption-Image Matching