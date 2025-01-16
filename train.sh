# Naive Diffusion UNet training 
# python src/minimal_train_naive.py \
#     --pretrained_model_name_or_path="Ksgk-fy/stable-diffusion-v1-5-smaller-unet-random" \
#     --dataset_name="Ksgk-fy/augmented-kanji-dataset" \
#     --output_dir="./runs/kanji-simple-unet-augment-small-lr" \
#     --resolution=128 \
#     --train_batch_size=1 \
#     --learning_rate=5e-6 \
#     --lr_scheduler="cosine" \
#     --lr_warmup_steps=1000 \
#     --num_train_epochs=200


# SD pre-trained 
python src/minimal_train_naive.py \
    --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --dataset_name="Ksgk-fy/kanji-dataset" \
    --output_dir="./runs/kanji-sd-base-bs1" \
    --resolution=128 \
    --train_batch_size=1 \
    --learning_rate=5e-6 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=1000 \
    --num_train_epochs=100 \