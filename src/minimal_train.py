import argparse
import math
import os
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from utils import evaluate_kanji_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Simple training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the Dataset (from the HuggingFace hub)",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Ksgk-fy/stable-diffusion-v1-5-smaller-unet-kanji",
        help="The model id for pushing to hub",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for training."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "constant"]',
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def main():
    args = parse_args()
    
    # Create unique run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize lists to store metrics
    training_stats = {
        'epoch': [],
        'avg_loss': [],
    }
    
    # Initialize accelerator - removed tensorboard logging
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",  # Use fp16 for faster training
    )

    # Log the device being used
    accelerator.print(f"Accelerator using device: {accelerator.device}")

    # Load the tokenizer, models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder",
    ).to(accelerator.device)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="vae",
    ).to(accelerator.device)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet",
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    # Load dataset
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name)
    else:
        dataset = load_dataset("imagefolder", data_files={"train": os.path.join(args.train_data_dir, "**")})

    # Preprocessing
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        
        # Ensure text input is properly formatted
        text = examples["text"]
        if isinstance(text, list):
            text = [str(t) for t in text]  # Convert all items to strings
        else:
            text = [str(text)]  # Convert single item to list of strings
        
        examples["input_ids"] = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        return examples

    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # Create Dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
    )

    # Prepare everything with accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    # Calculate number of training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Create the learning rate scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Training loop
    for epoch in range(args.num_train_epochs):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                total_loss += loss.detach().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{total_loss/(step+1):.4f}"})
        
        # Print epoch summary
        avg_loss = total_loss / len(train_dataloader)
        accelerator.print(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
        progress_bar.close()

        # Save training stats
        training_stats['epoch'].append(epoch)
        training_stats['avg_loss'].append(avg_loss)
        
        # Save metrics to CSV
        metrics_file = run_dir / 'training_metrics.csv'
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'avg_loss'])
            writer.writeheader()
            for i in range(len(training_stats['epoch'])):
                writer.writerow({
                    'epoch': training_stats['epoch'][i],
                    'avg_loss': training_stats['avg_loss'][i]
                })

        # Save checkpoint
        if epoch % 10 == 0:
            # Save model checkpoint
            # checkpoint_dir = run_dir / f"checkpoint-epoch-{epoch}"
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=text_encoder,
                vae=vae,
            )
            # pipeline.save_pretrained(checkpoint_dir)
            
            # Generate and save evaluation images
            evaluate_kanji_pipeline(
                pipeline, 
                dataset, 
                n_rows=2, 
                n_cols=4, 
                seed=33, 
                out_dir=str(run_dir), 
                out_name=f"kanji_eval_{epoch}.png"
            )
            
    pipeline.push_to_hub(args.model_id + f"_{epoch}")

if __name__ == "__main__":
    main() 