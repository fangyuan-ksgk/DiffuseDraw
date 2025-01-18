# from src.utils import prepare_kanji_dict

# prepare_kanji_dict(regenerate=True)


from datasets import load_dataset
from tqdm import tqdm
# load dataset
dataset = load_dataset("Ksgk-fy/expanded-kanji-dataset")
# save images
for idx, img in enumerate(tqdm(dataset['train']['image'])):
    img.save(f"data/kanji_expand_image/kanji_{idx}.png")