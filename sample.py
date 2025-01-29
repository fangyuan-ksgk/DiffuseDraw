from tqdm import tqdm as tqdm 
from diffusers import DiffusionPipeline
from src.utils import add_caption

# Load current-best model checkpoint (Inception model with 4:1 concept learning v.s. instance learning ratio)
pipe = DiffusionPipeline.from_pretrained("Ksgk-fy/kanji_inception_finetune_v4")
pipe.to("cuda")

caption_str_list = [
    "Bitcoin, Cryptocurrency", "Amazon", "Youtube", "Elon Musk", "DeepMind", "Jeff Hinton", "Blizzard", "Dragon Ball", "Naruto", "Worm", "AI", "Old man", "God", "water", "cliff", "drama", "Adele", "music"
]
caption_str = caption_str_list[2]
demo_path = "demo/"
for caption_str in tqdm(caption_str_list):
    
    prompt = f"a Kanji meaning {caption_str}"
    prompts = [prompt] * 8
    
    # 8 images per prompt, then pick out ones which makes more sense 
    images = pipe(prompts, guidance_scale=7.5, num_inference_steps=200).images
    for i, image in enumerate(images): 
        file_name = demo_path + caption_str.split(",")[0]+f"_{i}.png"
        add_caption(image, caption_str).save(file_name)