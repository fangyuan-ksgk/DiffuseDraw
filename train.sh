# Naive Diffusion UNet training 
# python src/minimal_train_naive.py \
#     --pretrained_model_name_or_path="Ksgk-fy/stable-diffusion-v1-5-smaller-unet-random" \
#     --dataset_name="Ksgk-fy/kanji-dataset" \
#     --output_dir="./runs/kanji-simple-unet" \
#     --train_batch_size=48 \
#     --num_train_epochs=100

# DS Accelerated Training | Gray scale projection
accelerate launch --config_file ds_config.json src/minimal_train.py \
    --pretrained_model_name_or_path="Ksgk-fy/stable-diffusion-v1-5-smaller-unet-random" \
    --dataset_name="Ksgk-fy/kanji-dataset" \
    --output_dir="./runs/kanji-simple-unet" \
    --resolution=128 \
    --train_batch_size=48 \
    --gradient_accumulation_steps=2 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=100 \
    --num_train_epochs=200