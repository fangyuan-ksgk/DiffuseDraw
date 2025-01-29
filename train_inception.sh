export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATASET_NAME="Ksgk-fy/inception-kanji-dataset-512"
export OUTPUT_DIR="checkpoint/kanji_inception_finetune_v4"

# copy script
cp -f src/train_inception.py external/diffusers/examples/dreambooth/train_inception.py

# Test with 'Kanji' instead of 'sks' token for dreambooth, so that I could simply combine at script level to train the model instead? 
# train dreambooth model
accelerate launch external/diffusers/examples/dreambooth/train_inception.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --dataset_name=$DATASET_NAME \
  --output_dir=$OUTPUT_DIR \
  --concept_prompt="an image of Kanji character" \
  --concept_learning_ratio=0.8 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=60000 \
  --mixed_precision="no" \
  --report_to="wandb" \
  --logging_dir="log" \
  --validation_prompt="an image of Kanji character meaning fish" \
  --num_validation_images=4 \
  --validation_steps=1000 \
  --train_text_encoder \
  --checkpoints_total_limit=10 \
  --push_to_hub