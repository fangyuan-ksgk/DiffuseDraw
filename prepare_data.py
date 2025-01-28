from src.utils import create_inception_dataset

DATASET_NAME = "Ksgk-fy/inception-kanji-dataset"

# 1. Non-english caption is involved ---> remove 
# 2. Resize towards 512x512, to be compatible with the model 

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push dataset to hub")
    parser.add_argument("--max_n", type=int, default=5, help="Maximum number of samples per class")
    parser.add_argument("--hub_model_id", type=str, default=DATASET_NAME, help="Hub model ID")
    args = parser.parse_args()
    create_inception_dataset(push_to_hub=args.push_to_hub, hub_model_id=args.hub_model_id, max_n=args.max_n)