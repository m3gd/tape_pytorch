# TAPE & UNet Diffusion Models

This repository contains implementations of both TAPE (Transformer And Patch Embeddings) and UNet models for diffusion-based image generation.

## Getting Started
configure the uv venv environment
```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install torch torchvision
uv sync
uv lock
```
we can use accelerate to launch the training with multi-gpu setup
```bash
accelerate config # run once
accelerate launch python train_tape_diffusion.py model=tape
```

## Training Models

### TAPE Model

To train a TAPE model:

```bash
python train_tape_diffusion.py model=tape
```

### UNet Model

To train a UNet model:

```bash
python train_tape_diffusion.py model=unet
```

## Configuration

The model settings are defined in the config files:

- `config/model/tape.yaml` - Configuration for TAPE model
- `config/model/unet.yaml` - Configuration for UNet model

You can customize other settings through the command line:

```bash
python train_tape_diffusion.py model=unet train.resolution=64 train.dataset_name=cifar10 
```

## Generated Samples

Models periodically generate sample images during training which are logged to your specified logger (TensorBoard or Weights & Biases).
