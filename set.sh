apt-get update
apt-get install libcairo2-dev
pip install Pillow wandb matplotlib cairosvg ftfy tensorboard Jinja2 peft sentencepiece protobuf
pip install --upgrade diffusers accelerate transformers xformers torchvision datasets deepspeed

# Set Up for DreamBooth Training
mkdir external
mkdir checkpoint
cd external
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
pip install wandb
pip install backports.tarfile
cd examples/dreambooth
pip install -r requirements.txt

# accelerate config
# wandb login