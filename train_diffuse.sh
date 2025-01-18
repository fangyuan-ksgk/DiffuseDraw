export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export DATASET_NAME="Ksgk-fy/kanji-dataset"
export DATASET_NAME="Ksgk-fy/expanded-kanji-dataset"
export OUTPUT_DIR="checkpoint/kanji_finetune_hf"

accelerate launch --mixed_precision="fp16"  external/diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=5e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --report_to="wandb" \
  --validation_prompts "alien" "evolution" "shark" \
  --validation_epochs 5 \
  --push_to_hub