python minimal_train.py \
    --pretrained_model_name_or_path="Ksgk-fy/stable-diffusion-v1-5-smaller-unet-random" \
    --dataset_name="Ksgk-fy/kanji-dataset" \
    --output_dir="./kanji-simple-unet" \
    --train_batch_size=48 \
    --num_train_epochs=100