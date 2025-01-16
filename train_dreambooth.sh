export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="data/kanji_image"
export OUTPUT_DIR="checkpoint/kanji_db_2"

# Validation image stored in WandB 
# -- Big Batch Size leads to "predicting multiple intertwined characters" side-effect | Set BatchSize = 1
# -- 


accelerate launch external/diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="an image of sks character" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --mixed_precision="no" \
  --report_to="wandb" \
  --logging_dir="log" \
  --validation_prompt="an image of sks character meaning 'dog'" \
  --num_validation_images=4 \
  --validation_steps=100 \