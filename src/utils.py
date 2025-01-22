import xml.etree.ElementTree as ET
import re, os, glob, io, re, random
from PIL import Image
from cairosvg import svg2png
from tqdm import tqdm 
import matplotlib.pyplot as plt
from matplotlib import font_manager
from datasets import Dataset, DatasetDict
from dataclasses import dataclass
from torchvision import transforms
from itertools import combinations
import torch
import numpy as np 
import pandas as pd 
from nltk.corpus import words, names
import nltk


def isKanji(v):
	return (v >= 0x4E00 and v <= 0x9FC3) or (v >= 0x3400 and v <= 0x4DBF) or (v >= 0xF900 and v <= 0xFAD9) or (v >= 0x2E80 and v <= 0x2EFF) or (v >= 0x20000 and v <= 0x2A6DF)

def is_valid_kanji_code(code):
    try:
        # Convert hex string code to integer
        kanji_value = int(code, 16)
        return isKanji(kanji_value)
    except ValueError:
        return False
    
def code_to_kanji(code):
    return chr(int(code, 16))

def kanji_to_code(character):
    return f"{ord(character):05x}"

pathre = re.compile(r'<path .*d="([^"]*)".*/>')

def createPathsSVG(f, out_dir: str = "kanji_paths"): # create a new SVG file with black strokes only
	s = open(f, "r", encoding="utf-8").read()
	paths = pathre.findall(s)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	out_path = os.path.join(out_dir, f"{os.path.basename(f)[:-4]}-paths.svg")
	out = open(out_path, "w", encoding="utf-8")
	out.write("""<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.0//EN" "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd" []>
<svg xmlns="http://www.w3.org/2000/svg" width="109" height="109" viewBox="0 0 109 109" style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">\n""")
	i = 1
	for path in paths:
		out.write('<!--%2d--><path d="%s"/>\n' % (i, path))
		i += 1
	out.write("</svg>")
	out.close()
	return out_path

def parse_kanjidic2(xml_path):
    """
    Parse KANJIDIC2 XML file and return a dictionary mapping kanji codes to English meanings.
    
    Args:
        xml_path (str): Path to kanjidic2.xml file
    
    Returns:
        dict: {kanji_code: english_meaning} where kanji_code is in "0xxxx" format
    """
    kanji_dict = {}
    
    # Parse XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Iterate through each character entry
    for character in root.findall('character'):
        # Get hex code from codepoint
        cp_elem = character.find('.//cp_value[@cp_type="ucs"]')
        if cp_elem is None:
            continue
            
        # Format code to 5-digit hex with leading zeros
        kanji_code = f"{int(cp_elem.text, 16):05x}"  # Convert to int and back to 5-digit hex
        
        # Get English meanings
        meanings = []
        rm_group = character.find('.//reading_meaning')
        if rm_group is not None:
            rmg = rm_group.find('rmgroup')
            if rmg is not None:
                meanings = [m.text for m in rmg.findall('meaning') 
                          if m.get("xml:lang") is None]
        
        if meanings and is_valid_kanji_code(kanji_code):
            kanji_dict[code_to_kanji(kanji_code)] = '; '.join(meanings)
    
    return kanji_dict

def convert_svg_to_image(svg_path: str, out_dir: str = "data/kanji_image"):
    out_dir = "data/kanji_image"
    os.makedirs(out_dir, exist_ok=True)

    # Read SVG and convert to PNG in memory
    png_data = svg2png(url=svg_path, output_width=128, output_height=128)

    # Convert PNG bytes to PIL Image
    image = Image.open(io.BytesIO(png_data))
    
    # Create a white background image
    white_bg = Image.new('RGBA', image.size, 'WHITE')
    
    # Paste the kanji image onto the white background
    white_bg.paste(image, (0, 0), image)
    
    image_path = os.path.join(out_dir, f"{os.path.basename(svg_path).replace('-paths.svg', '-128.png')}")
    white_bg.save(image_path)
    return image_path

def get_kanji_graphs(kanji_svg_folder: str = "data/kanji", kanji_graph_folder: str = "data/kanji_paths"):
    kanji_files = [p for p in glob.glob(f"{kanji_svg_folder}/*.svg") if not "-" in p]
    kanji_codes = [p.split("/")[-1].split(".")[0] for p in kanji_files]

    kanji_graphs = {}
    for file, code in tqdm(zip(kanji_files, kanji_codes), total=len(kanji_files), desc="Processing kanji graphs"):
        if not is_valid_kanji_code(code):
            continue
        svg_path = createPathsSVG(file, kanji_graph_folder)
        image_path = convert_svg_to_image(svg_path, kanji_graph_folder)
        os.remove(svg_path)
        kanji_graphs[code_to_kanji(code)] = image_path
        
    return kanji_graphs

import json 

# combine kanji_dict and kanji_meanings
def prepare_kanji_dict(kanji_svg_folder: str = "data/kanji", kanji_graph_folder: str = "data/kanji_paths", kanjidic2_path: str = "data/kanjidic2.xml",
                       out_dir: str = "data", regenerate: bool = False):
    
    if os.path.exists(os.path.join(out_dir, "kanji_dict.json")) and not regenerate:
        return json.load(open(os.path.join(out_dir, "kanji_dict.json"), "r", encoding="utf-8"))
    
    kanji_graphs = get_kanji_graphs(kanji_svg_folder, kanji_graph_folder)
    kanji_meanings = parse_kanjidic2(kanjidic2_path)
    
    kanji_dict = {} 
    for kanji, path in kanji_graphs.items():
        if kanji in kanji_meanings:
            kanji_dict[kanji] = {"path": path, "meanings": kanji_meanings[kanji]}
        else:
            kanji_dict[kanji] = {"path": path, "meanings": ""}

    with open(os.path.join(out_dir, "kanji_dict.json"), "w", encoding="utf-8") as f:
        json.dump(kanji_dict, f, ensure_ascii=False, indent=4)
        
    return kanji_dict

def evaluate_kanji_pipeline(pipeline, dataset, n_rows=2, n_cols=4, seed=33, out_dir: str = "runs", out_name: str = "kanji_eval.png"): 
    # Move pipeline to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    random.seed(seed)
    prompts = []
    kanji_list = random.sample(list(dataset['test']['text']), n_rows*n_cols)
    for s in kanji_list:
        if isinstance(s, str):
            s = s.split(";") 
        prompts.append(random.choice(s))
    
    # Generate images
    with torch.autocast(device):
        images = pipeline(prompts, num_inference_steps=50).images
    
    # Create a figure with n_rows and n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, img in enumerate(images):
        caption = prompts[idx]
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].text(0.5, -0.3, caption, 
                    horizontalalignment='center', 
                    transform=axes[idx].transAxes,
                    fontsize=20)
    
    plt.tight_layout(h_pad=2)  # Increase vertical padding between subplots
    
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Save figure before showing it
    plt.savefig(os.path.join(out_dir, out_name))
    plt.close()

    return images
    
    
def vis_kanji_data(kanji_dict, n_rows=2, n_cols=4):
    # Add a Japanese font
    plt.rcParams['font.family'] = ['Hiragino Sans GB', 'Arial Unicode MS', 'sans-serif']

    # Create a figure with 2 rows and 6 columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))
    axes = axes.flatten()

    # Get 12 kanji characters
    kanji_samples = list(kanji_dict.keys())[:n_rows * n_cols]
    random.shuffle(kanji_samples) # shuffle the order

    # Plot each kanji and its meanings
    for idx, kanji in enumerate(kanji_samples):
        image_path = kanji_dict[kanji]['path']
        meanings = kanji_dict[kanji]['meanings'].split(';')[:2]  # Take only first 2 meanings
        
        # Load and display image
        img = Image.open(image_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Add kanji and meanings below the image
        meanings_text = '\n'.join(m.strip() for m in meanings)  # Strip whitespace and join with newline
        # Update text rendering with font properties
        axes[idx].text(0.5, -0.15, meanings_text, 
                    horizontalalignment='center', 
                    transform=axes[idx].transAxes,
                    fontsize=20)

    plt.tight_layout(h_pad=2)  # Increase vertical padding between subplots
    plt.show()
    
    

def setup_english_words():
    try:
        # Download required NLTK data if not already present
        nltk.download('words', quiet=True)
        nltk.download('names', quiet=True)
    except:
        pass
    
    # Create a set of English words and names for faster lookup
    english_words = set(words.words())
    english_names = set(names.words())
    return english_words.union(english_names)

def is_english(text):
    """
    Check if a given text is English using multiple heuristics.
    
    Args:
        text (str): Input text to check
        
    Returns:
        bool: True if the text is likely English, False otherwise
    """
    # Initialize English word set if not already done
    if not hasattr(is_english, 'english_words'):
        is_english.english_words = setup_english_words()
    
    # Clean and normalize the text
    text = text.strip().lower()
    
    # Skip empty strings
    if not text:
        return False
    
    # Split into words and remove punctuation
    words_in_text = re.findall(r'\b\w+\b', text)
    
    if not words_in_text:
        return False
    
    # Count how many words are in our English dictionary
    english_word_count = sum(1 for word in words_in_text 
                           if word in is_english.english_words)
    
    # Calculate the ratio of English words
    english_ratio = english_word_count / len(words_in_text)
    
    # Common non-English characters (including accents and special characters)
    non_english_pattern = r'[àáâãäçèéêëìíîïñòóôõöùúûüýÿ]'
    has_non_english_chars = bool(re.search(non_english_pattern, text))
    
    # Return True if:
    # 1. No non-English characters are present AND
    # 2. At least 60% of words are recognized English words
    return (not has_non_english_chars) and (english_ratio >= 0.6)
    

def get_all_combinations(list_of_captions):
    text_list = []
    for i in range(1, len(list_of_captions) + 1):
        for combo in combinations(list_of_captions, i):
            text_list.append(", ".join(combo))
    return text_list

    
def _create_inception_dataset(kanji_dict):
    dataset_dict = {"text": [], "image": []}

    for kanji, data in tqdm(kanji_dict.items(), total=len(kanji_dict), desc="Processing Kanji Dataset"):
        image = Image.open(data['path'])
        list_of_captions = data['meanings'].split("; ")

        en_captions = [caption for caption in list_of_captions if is_english(caption)] # filter out non-english captions

        all_combinations = get_all_combinations(en_captions) # exhaust all combinations (variable # of element)

        special_token = "Kanji"
        CAPTION_TEMPLATE = "a {special_token} meaning {caption}"
        transform_text = lambda caption: CAPTION_TEMPLATE.format(special_token=special_token, caption=caption)

        for text in all_combinations:
            dataset_dict["text"].append(transform_text(text))
            dataset_dict["image"].append(image)
                    
    dataset = DatasetDict({'train': Dataset.from_dict(dataset_dict)})
    return dataset
    


def create_inception_dataset(push_to_hub: bool = False, hub_model_id: str = "Ksgk-fy/inception-kanji-dataset"):
    kanji_dict = prepare_kanji_dict()
    dataset = _create_inception_dataset(kanji_dict)
    if push_to_hub:
        dataset.push_to_hub(hub_model_id)
    return dataset




@dataclass
class TrainingConfig:
    image_size: int
    train_batch_size: int
    eval_batch_size: int
    num_epochs: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_warmup_steps: int
    save_image_epochs: int
    save_model_epochs: int
    mixed_precision: str
    output_dir: str
    push_to_hub: bool
    hub_model_id: str
    hub_private_repo: bool
    overwrite_output_dir: bool
    seed: int

    @classmethod
    def from_json(cls, config_path: str):
        config_dict = json.load(open(config_path, "r", encoding="utf-8"))
        return cls(**config_dict)


def get_transform(config, tokenizer, text_encoder):
    def transform(examples):
        # Image transforms
        preprocess = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        
        # Text transforms using CLIP tokenizer
        text_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get CLIP embeddings
        with torch.no_grad():
            text_embeddings = text_encoder(text_inputs.input_ids)[0]
        
        return {
            "images": images,
            "text_embeddings": text_embeddings
        }
    return transform

LUMINOSITY_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])


def rgb_to_gray(image: Image.Image) -> Image.Image:
    """Convert RGB PIL Image to grayscale using luminosity weights.
    
    Args:
        image (PIL.Image): Input RGB image
        
    Returns:
        PIL.Image: Grayscale image
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Apply luminosity weights and sum across color channels
    gray_array = np.dot(img_array, LUMINOSITY_WEIGHTS)
    
    # Convert back to PIL Image
    return Image.fromarray(np.uint8(gray_array))



def save_loss_curve(metrics_file: str):
    df = pd.read_csv(metrics_file)
    plt.plot(df['epoch'], df['avg_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Curve')
    # Convert Path to string before using replace
    output_path = str(metrics_file).replace("metrics.csv", "loss_curve.png")
    plt.savefig(output_path)
    plt.close()
    

def expand_dataset(examples):
    new_examples = {
        "image": [],
        "text": []
    }
    
    images = examples["image"]
    texts = examples["text"]    
    
    for image, text in zip(images, texts):
        if not isinstance(text, list):
            text = [text]
        for text_variant in text:
            new_examples["image"].append(image)
            new_examples["text"].append(str(text_variant))
                
    return new_examples

# Define augmentation transforms with different parameters
augmentations = [
    # Original image (no augmentation)
    transforms.Compose([
        transforms.Lambda(lambda x: x)
    ]),
    # Random rotations with different angles
    transforms.Compose([
        transforms.RandomRotation(degrees=(-5, 5))
    ]),
    transforms.Compose([
        transforms.RandomRotation(degrees=(-10, 10))
    ]),
    # Random perspective with different distortion scales
    transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.15, p=1.0)
    ]),
    transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.25, p=1.0)
    ]),
    # Elastic distortions with different parameters
    transforms.Compose([
        transforms.ElasticTransform(alpha=40.0, sigma=4.0)
    ]),
    transforms.Compose([
        transforms.ElasticTransform(alpha=60.0, sigma=6.0)
    ]),
    # Additional perspective transformations (replacing brightness/contrast)
    transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.18, p=1.0)
    ]),
    transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.22, p=1.0)
    ]),
    # Combined transformations
    transforms.Compose([
        transforms.RandomRotation(degrees=(-3, 3)),
        transforms.RandomPerspective(distortion_scale=0.1, p=1.0)
    ])
]
        

def augment_dataset(examples, aug_img: bool = True):
    new_examples = {
        "image": [],
        "text": []
    }
    
    images = examples["image"]
    texts = examples["text"]    
    
    for image, text in zip(images, texts):
        if not isinstance(text, list):
            text = [text]
        # Apply each augmentation for each text variant
        for text_variant in text:
            if aug_img:
                for aug in augmentations:
                    try:
                        # Apply augmentation
                        aug_image = aug(image)
                        new_examples["image"].append(aug_image)
                        new_examples["text"].append(str(text_variant))
                    except Exception as e:
                        print(f"Augmentation failed: {e}")
                        continue
            else:
                aug_image = augmentations[0](image)
                new_examples["image"].append(aug_image)
                new_examples["text"].append(str(text_variant))
    
    return new_examples


def clean_dataset(examples):
    new_examples = {
        "image": [],
        "text": []
    }
    for image, text in zip(examples["image"], examples["text"]):
        if isinstance(text, list):
            text = text[0]
        new_examples["image"].append(image)
        new_examples["text"].append(text)
    return new_examples


def visualize_tensor(pixel_values, is_grayscale=True):
    # Clone tensor to avoid modifying the original
    image = pixel_values.clone()
    
    # Move to CPU if needed
    if image.device != 'cpu':
        image = image.cpu()
        
    # Denormalize from [-1,1] back to [0,1]
    image = (image * 0.5 + 0.5).clamp(0, 1)
    
    # Remove batch dimension if present
    if len(image.shape) == 4:
        image = image[0]
    
    # Convert to PIL Image
    image = transforms.ToPILImage()(image)

    return image