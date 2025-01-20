import argparse
import math
import os
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime
from contextlib import nullcontext
import numpy as np 
import wandb

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
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
from utils import evaluate_kanji_pipeline, LUMINOSITY_WEIGHTS, save_loss_curve

from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers import DiffusionPipeline


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)


def save_model_card(
    args,
    repo_id: str,
    images: list = None,
    repo_folder: str = None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_description += wandb_info

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))
    
    
def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


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
        default="sd-model-finetuned",
        help="The model id for pushing to hub",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
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
        "--lr_warmup_steps", type=int, default=200, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder.",
    )
    args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def main():
    args = parse_args()
    
    # Initialize wandb right after parsing args
    wandb.init(
        project=args.model_id,
        config=vars(args),
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Create unique run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize lists to store metrics
    training_stats = {
        'epoch': [],
        'avg_loss': [],
        'val_loss': [],  # Add validation loss tracking
        'best_val_loss': float('inf'),
        'patience_counter': 0,
    }
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    
    
    
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

    # Add gradient clipping
    clip_grad_norm = 1.0

    # Load dataset | More test-to-image pairs (sample more from caption list) | More noisy images
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name)
    else:
        dataset = load_dataset("imagefolder", data_files={"train": os.path.join(args.train_data_dir, "**")})

    # Preprocessing
    def preprocess_train(examples):
        
        # Use original RGB transforms
        train_transforms = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Three channel normalization for RGB
        ])
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
    val_dataset = dataset['test'].with_transform(preprocess_train)

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
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
    )

    # Prepare everything with accelerator
    unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader
    )

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

    # Add validation loop after each epoch
    patience = 5  # Number of epochs to wait before early stopping
    
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
                
                # Add gradient clipping before optimizer step
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), clip_grad_norm)
                
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

        # Log training metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "learning_rate": lr_scheduler.get_last_lr()[0]
        })

        # Add validation loop after each epoch
        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader: 
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
                val_loss += loss.detach().item()
        
        val_loss /= len(val_dataloader)
        training_stats['val_loss'].append(val_loss)
        
        # Early stopping check
        if val_loss < training_stats['best_val_loss']:
            training_stats['best_val_loss'] = val_loss
            training_stats['patience_counter'] = 0
        else:
            training_stats['patience_counter'] += 1
            if training_stats['patience_counter'] >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # After validation, log validation metrics
        wandb.log({
            "val_loss": val_loss,
            "best_val_loss": training_stats['best_val_loss']
        })
        
        # Save Intermediate Model Output
        if epoch % 10 == 0:
            # Save model checkpoint
            pipeline = StableDiffusionPipeline(
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=accelerator.unwrap_model(vae),
                tokenizer=tokenizer,  # tokenizer doesn't need unwrapping as it's not a torch module
                scheduler=noise_scheduler,  # scheduler doesn't need unwrapping as it's not a torch module
                safety_checker=None,
                feature_extractor=None,
            )
            
            # Generate and save evaluation images
            images = evaluate_kanji_pipeline(
                pipeline, 
                dataset, 
                n_rows=2, 
                n_cols=4, 
                seed=33, 
                out_dir=str(run_dir), 
                out_name=f"kanji_eval_{epoch}.png"
            )
            
            # Log the evaluation image to wandb using the generated images directly
            wandb.log({
                "kanji_samples": wandb.Image(images)
            })
        
    # pipeline.push_to_hub("Ksgk-fy/" + args.model_id + f"_{epoch}")
    os.makedirs('checkpoint/kanji_finetune', exist_ok=True)
    pipeline.save_pretrained('checkpoint/kanji_finetune', args.model_id + f"_{epoch}")
    save_loss_curve(metrics_file)
    
    # At the end of training
    wandb.finish()
    
if __name__ == "__main__":
    main() 