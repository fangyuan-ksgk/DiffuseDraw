from src.utils import augment_dataset, clean_dataset
from datasets import load_dataset, Dataset, DatasetDict

print(":: Loading Original Kanji Dataset")
dataset_name = "Ksgk-fy/kanji-dataset"
dataset = load_dataset(dataset_name)


print(":: Start Augmenting Dataset")
# toy_dataset = Dataset.from_dict(dataset['train'][:2])
# trainset = toy_dataset.map(augment_dataset, batched=True, batch_size=1,remove_columns=toy_dataset.column_names, num_proc=10)
trainset = dataset['train'].map(augment_dataset, batched=True, batch_size=1,remove_columns=dataset['train'].column_names, num_proc=10)
testset = dataset['test'].map(clean_dataset, batched=True, batch_size=1,remove_columns=dataset['test'].column_names, num_proc=10)


print(":: Storing Dataset and upload to HF")
new_dataset = DatasetDict({
    'train': trainset,
    'test': testset
})

# Upload the dataset to the Hub
new_dataset_name = "Ksgk-fy/augmented-kanji-dataset"  # Change this to your desired name

new_dataset.push_to_hub(
    new_dataset_name,
    private=False,  # Set to True if you want it private
)