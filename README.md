# DiffuseDraw
Train diffusion model on drawings

1. Set up environment 
```bash
bash set.sh
```

2. Prepare data 
```bash
python prepare_data.py --push_to_hub --max_n 2 --hub_model_id Ksgk-fy/inception-kanji-dataset-512
```

3. Train model 
```bash
bash train_inception.sh
```

4. Test model 
```bash
python test.py
```