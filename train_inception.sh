export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="Ksgk-fy/expanded-kanji-dataset"
export OUTPUT_DIR="checkpoint/kanji_inception_finetune"

# copy script
cp -f src/train_inception.py external/diffusers/examples/dreambooth/train_inception.py

# Test with 'Kanji' instead of 'sks' token for dreambooth, so that I could simply combine at script level to train the model instead? 
# train dreambooth model
accelerate launch external/diffusers/examples/dreambooth/train_inception.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --dataset_name=$DATASET_NAME \
  --output_dir=$OUTPUT_DIR \
  --concept_prompt="an image of Kanji character" \
  --concept_learning_ratio=0.7 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=8000 \
  --mixed_precision="no" \
  --report_to="wandb" \
  --logging_dir="log" \
  --validation_prompt="an image of Kanji character meaning dog" \
  --num_validation_images=4 \
  --validation_steps=1000 \
  --train_text_encoder \
  --push_to_hub