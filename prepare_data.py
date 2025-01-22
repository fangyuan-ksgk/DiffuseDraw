from src.utils import create_inception_dataset

DATASET_NAME = "Ksgk-fy/inception-kanji-dataset"

create_inception_dataset(push_to_hub=True, hub_model_id=DATASET_NAME)