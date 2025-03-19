import inspect
import logging
import math
import os
import random
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_tensorboard_available, is_wandb_available
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from torchvision import transforms
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig

from lion_pytorch import Lion
# Add parent directory to sys.path to import the TAPE model
import sys
sys.path.append(os.getcwd())

from tape_model import TAPEModel
from diffusers import UNet2DModel
from rin_model import RINModel


logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def exists(x):
    """Check if a variable exists (is not None)."""
    return x is not None

def default(val, d):
    """Return val if it exists, otherwise return d (or call d if it's callable)."""
    if exists(val):
        return val
    return d() if callable(d) else d

def normalize_img_variance(x, eps=1e-5):
    """Normalize variance of noised image, for better high-resolution training."""
    std = torch.std(x, dim=(1, 2, 3), keepdim=True, unbiased=False)
    return x / std.clamp(min=eps)

def right_pad_dims_to(x, t):
    """Pad dims of t to match x for broadcasting."""
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def gamma_to_alpha_sigma(gamma, scale=1):
    """Convert gamma to alpha and sigma for different noise levels."""
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

def log(t, eps=1e-20):
    """Safe log function with minimum value."""
    return torch.log(t.clamp(min=eps))

def gamma_to_log_snr(gamma, scale=1, eps=1e-5):
    """Convert gamma to log SNR for loss weighting."""
    return log(gamma * (scale ** 2) / (1 - gamma), eps=eps)

def safe_div(numer, denom, eps=1e-10):
    """Safe division with small epsilon to prevent division by zero."""
    return numer / denom.clamp(min=eps)


class DDPMPipelineGeneric(DDPMPipeline):
    """DDPMPipeline that works with either a TAPE model or a UNet model."""

    @torch.no_grad()
    def __call__(
        self,
        batch_size=1,
        generator=None,
        num_inference_steps=1000,
        output_type="pil",
        return_dict=True,
        enable_self_conditioning=True,
        enable_variance_normalization=True,
        variance_scale=1.0,
        **kwargs,
    ):
        # Sample noise
        image_shape = (
            batch_size,
            self.unet.config.in_channels,
            self.unet.config.sample_size,
            self.unet.config.sample_size,
        )
        image = torch.randn(image_shape, generator=generator, device=self.device)

        # Set step values
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Initialize self-conditioning
        x_start = None
        last_latents = None

        for t in self.progress_bar(self.scheduler.timesteps):
            # Apply variance normalization if enabled
            if enable_variance_normalization and variance_scale < 1.0:
                model_input = normalize_img_variance(image)
            else:
                model_input = image
            
            # Set self-conditioning if available
            if enable_self_conditioning and hasattr(self.unet, "set_self_conditioning") and x_start is not None:
                self.unet.set_self_conditioning(x_start, last_latents)
            
            # 1. predict model output
            model_output = self.unet(model_input, t.to(image.device)).sample
            
            # 2. compute previous image: x_t -> x_t-1
            step_output = self.scheduler.step(model_output, t, image, generator=generator)
            image = step_output.prev_sample
            
            # Store predicted x0 for self-conditioning in the next step
            if enable_self_conditioning:
                # Extract the predicted x0 based on prediction type
                if hasattr(self.scheduler.config, "prediction_type"):
                    prediction_type = self.scheduler.config.prediction_type
                else:
                    prediction_type = "epsilon"  # Default for older diffusers versions
                
                if prediction_type == "epsilon":
                    # For noise prediction, convert noise to x0
                    alpha_t = self.scheduler.alphas_cumprod[t.long()]
                    alpha_t = alpha_t.to(device=image.device)[..., None, None, None]
                    sigma_t = torch.sqrt(1 - alpha_t)
                    x_start = (image - sigma_t * model_output) / alpha_t.clamp(min=1e-6)
                elif prediction_type == "sample":
                    # Direct x0 prediction
                    x_start = model_output
                elif prediction_type == "v_prediction":
                    # v-prediction, convert to x0
                    alpha_t = self.scheduler.alphas_cumprod[t.long()]
                    alpha_t = alpha_t.to(device=image.device)[..., None, None, None]
                    sigma_t = torch.sqrt(1 - alpha_t)
                    x_start = alpha_t * image - sigma_t * model_output
                
                # Clamp to valid image range
                x_start = x_start.clamp(-1.0, 1.0)
                
                # Capture latents if model supports it
                if hasattr(self.unet, "get_latents"):
                    last_latents = self.unet.get_latents()

        # Final image processing
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return {"sample": image}


def create_model(config):
    """Create a model (TAPE, UNet, or RIN) based on config."""
    if config.model.name == "tape":
        model = TAPEModel(
            sample_size=config.model.sample_size,
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            num_layers=config.model.num_layers,
            latent_dim=config.model.latent_dim,
            tape_dim=config.model.tape_dim,
            num_latents=config.model.num_latents,
            patch_size=config.model.patch_size,
            num_heads=config.model.num_heads,
            time_embedding_dim=config.model.time_embedding_dim,
            dropout=config.model.dropout,
        )
    elif config.model.name == "unet":
        model = UNet2DModel(
            sample_size=config.model.sample_size,
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            layers_per_block=config.model.layers_per_block,
            block_out_channels=config.model.block_out_channels,
            down_block_types=config.model.down_block_types,
            up_block_types=config.model.up_block_types,
        )
    elif config.model.name == "rin":
        attention_config = {
            "heads": config.model.attention.heads,
            "dim_head": config.model.attention.dim_head,
            "flash": config.model.attention.flash,
            "qk_norm": config.model.attention.qk_norm
        }
        model = RINModel(
            sample_size=config.model.sample_size,
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            dim=config.model.dim,
            image_size=config.model.image_size,
            patch_size=config.model.patch_size,
            depth=config.model.depth,
            latent_self_attn_depth=config.model.latent_self_attn_depth,
            dim_latent=config.model.dim_latent,
            num_latents=config.model.num_latents,
            learned_sinusoidal_dim=config.model.learned_sinusoidal_dim,
            latent_token_time_cond=config.model.latent_token_time_cond,
            dual_patchnorm=config.model.dual_patchnorm,
            patches_self_attn=config.model.patches_self_attn,
            attention=attention_config,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.name}")

    return model


def get_scheduler_config(name):
    """Helper function to get scheduler kwargs based on name."""
    return {
        "linear": {"beta_start": 0.00085, "beta_end": 0.012},
        "scaled_linear": {"beta_start": 0.0001, "beta_end": 0.02},
        "squaredcos_cap_v2": {"beta_start": 0.0001, "beta_end": 0.02},
    }.get(name, {})


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Initialize accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    logging_dir = os.path.join(cfg.train.output_dir, cfg.logging.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.train.output_dir,
        logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=cfg.train.mixed_precision,
        log_with=cfg.logging.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs, ddp_kwargs],
    )

    # Verify logger availability
    if cfg.logging.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif cfg.logging.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

        # Initialize wandb
        if accelerator.is_main_process:
            wandb.init(
                project=cfg.logging.wandb.project,
                entity=cfg.logging.wandb.entity,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

    # Set random seed
    if cfg.train.seed is not None:
        set_seed(cfg.train.seed)

    # Log with accelerator
    if accelerator.is_main_process:
        if cfg.train.output_dir is not None:
            os.makedirs(cfg.train.output_dir, exist_ok=True)

    # Create model, scheduler, and optimizer
    model = create_model(cfg)

    # Verify prediction type is supported
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())

    scheduler_kwargs = get_scheduler_config(cfg.diffusion.beta_schedule)

    if accepts_prediction_type:
        scheduler = DDPMScheduler(
            num_train_timesteps=cfg.diffusion.num_train_timesteps,
            prediction_type=cfg.diffusion.prediction_type,
            **scheduler_kwargs
        )
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=cfg.diffusion.num_train_timesteps,
            **scheduler_kwargs
        )

    # Initialize optimizer
    if cfg.optimizer.name=="adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.learning_rate,
            betas=(cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2),
            weight_decay=cfg.optimizer.adam_weight_decay,
            eps=cfg.optimizer.adam_epsilon,
        )
    elif cfg.optimizer.name=="lion":
        optimizer = Lion(
            model.parameters(),
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay
        )

    # Enable xformers if requested
    if cfg.train.enable_xformers_memory_efficient_attention:
        try:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                logger.info("Using xFormers for memory efficient attention")
                model.enable_xformers_memory_efficient_attention()
            else:
                logger.warning(
                    "xformers is not available. Make sure it is installed correctly or disable the option."
                )
        except Exception as e:
            logger.warning(f"Error enabling xformers: {e}")
            logger.warning("Continuing without xformers.")

    # Set up dataset
    if cfg.train.dataset_name is not None:
        dataset = load_dataset(
            cfg.train.dataset_name,
            cfg.train.dataset_config_name,
            cache_dir=cfg.train.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset(
            "imagefolder",
            data_dir=cfg.train.train_data_dir,
            cache_dir=cfg.train.cache_dir,
            split="train"
        )

    # Create image transformations
    augmentations = transforms.Compose([
        transforms.Resize(cfg.train.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(cfg.train.resolution) if cfg.train.center_crop else transforms.RandomCrop(cfg.train.resolution),
        transforms.RandomHorizontalFlip() if cfg.train.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def transform_images(examples):
        # Handle different dataset structures (CIFAR-10 uses "img", flowers dataset uses "image", etc.)
        image_key = next(key for key in examples.keys() if key in ["img", "image", "images"])
        images = [augmentations(image.convert("RGB")) for image in examples[image_key]]
        return {"input": images}

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transform_images)
    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.train_batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )

    # Create learning rate scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.train.gradient_accumulation_steps)
    max_train_steps = cfg.train.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        cfg.scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.scheduler.lr_warmup_steps * cfg.train.gradient_accumulation_steps,
        num_training_steps=max_train_steps,
    )

    # Set up EMA model
    if hasattr(cfg, "ema") and cfg.ema.use_ema:
        # Determine which model class to use for EMA
        if cfg.model.name == "tape":
            model_cls = TAPEModel
        elif cfg.model.name == "rin":
            model_cls = RINModel
        else:
            model_cls = UNet2DModel

        ema_model = EMAModel(
            model.parameters(),
            decay=cfg.ema.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=cfg.ema.ema_inv_gamma,
            power=cfg.ema.ema_power,
            model_cls=model_cls,
            model_config=model.config,
        )
    else:
        ema_model = None

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if ema_model is not None:
        ema_model.to(accelerator.device)

    # Initialize trackers for logging
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    # Calculate training batch size
    total_batch_size = cfg.train.train_batch_size * accelerator.num_processes * cfg.train.gradient_accumulation_steps

    # Log training information
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {cfg.train.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # Initialize training state
    global_step = 0
    first_epoch = 0

    # Resume from checkpoint if specified
    if cfg.checkpoint.resume_from_checkpoint:
        if cfg.checkpoint.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.checkpoint.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.train.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.warning(
                f"Checkpoint '{cfg.checkpoint.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.checkpoint.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.train.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * cfg.train.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * cfg.train.gradient_accumulation_steps)

    # Training loop
    for epoch in range(first_epoch, cfg.train.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch // accelerator.num_processes, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if cfg.checkpoint.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % cfg.train.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["input"]

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bsz = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
            
            # Apply variance normalization if enabled
            if hasattr(cfg.diffusion, "variance_normalization") and cfg.diffusion.variance_normalization.enabled:
                scale = cfg.diffusion.variance_normalization.scale
                if scale < 1.0:
                    noisy_images = normalize_img_variance(noisy_images)
            
            # Self-conditioning setup
            x_self_cond = None
            latent_self_cond = None
            
            if hasattr(cfg.diffusion, "self_conditioning") and cfg.diffusion.self_conditioning.enabled:
                # Apply self-conditioning with probability p
                if random.random() < cfg.diffusion.self_conditioning.probability:
                    with torch.no_grad():
                        # Get model prediction for conditioning
                        model_output_for_cond = model(noisy_images, timesteps).sample
                        
                        # Extract alpha and sigma for the timesteps
                        alpha_t = _extract_into_tensor(
                            scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        sigma_t = torch.sqrt(1 - alpha_t)
                        
                        # Compute self-conditioning based on prediction type
                        if cfg.diffusion.prediction_type == "epsilon":
                            x_self_cond = (noisy_images - sigma_t * model_output_for_cond) / alpha_t
                        elif cfg.diffusion.prediction_type == "sample":
                            x_self_cond = model_output_for_cond
                        elif cfg.diffusion.prediction_type == "v_prediction":
                            x_self_cond = alpha_t * noisy_images - sigma_t * model_output_for_cond
                        
                        # Clamp to valid image range
                        x_self_cond = x_self_cond.clamp(-1.0, 1.0)
                        x_self_cond = x_self_cond.detach()
                        
                        # For models that support latent self-conditioning
                        if hasattr(model, "get_latents") and cfg.diffusion.self_conditioning.enable_for_latents:
                            latent_self_cond = model.get_latents()
                            if latent_self_cond is not None:
                                latent_self_cond = latent_self_cond.detach()

            with accelerator.accumulate(model):
                # Forward pass with self-conditioning
                if hasattr(model, "set_self_conditioning") and x_self_cond is not None:
                    model.set_self_conditioning(x_self_cond, latent_self_cond)
                    
                # Predict the noise residual
                model_output = model(noisy_images, timesteps).sample
                
                # Compute loss based on prediction type with improved weighting
                if cfg.diffusion.prediction_type == "epsilon":
                    target = noise
                elif cfg.diffusion.prediction_type == "sample":
                    target = clean_images
                elif cfg.diffusion.prediction_type == "v_prediction":
                    # v-prediction target: alpha * noise - sigma * x_0
                    alpha_t = _extract_into_tensor(
                        scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    sigma_t = torch.sqrt(1 - alpha_t)
                    target = alpha_t * noise - sigma_t * clean_images
                else:
                    raise ValueError(f"Unsupported prediction type: {cfg.diffusion.prediction_type}")
                
                # Calculate unweighted loss
                loss = F.mse_loss(model_output, target, reduction="none")
                loss = torch.mean(loss, dim=(1, 2, 3))
                
                # Apply loss weighting based on SNR if enabled
                if hasattr(cfg.diffusion, "loss_weighting") and cfg.diffusion.loss_weighting.min_snr:
                    # Extract alpha and calculate SNR
                    alpha_t = _extract_into_tensor(
                        scheduler.alphas_cumprod, timesteps, (clean_images.shape[0],)
                    )
                    sigma_t = torch.sqrt(1 - alpha_t)
                    
                    # Calculate signal-to-noise ratio
                    snr = (alpha_t * alpha_t) / (sigma_t * sigma_t)
                    
                    # Apply min SNR clipping if enabled
                    if cfg.diffusion.loss_weighting.min_snr:
                        snr_gamma = cfg.diffusion.loss_weighting.min_snr_gamma
                        snr_clipped = torch.clamp(snr, max=snr_gamma)
                        
                        # Different weighting schemes based on prediction type
                        if cfg.diffusion.prediction_type == "epsilon":
                            loss_weight = snr_clipped / snr
                        elif cfg.diffusion.prediction_type == "sample":
                            loss_weight = snr_clipped
                        elif cfg.diffusion.prediction_type == "v_prediction":
                            loss_weight = snr_clipped / (snr + 1)
                            
                        # Apply the weights
                        loss = loss * loss_weight
                
                # Take mean over batch
                loss = loss.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update EMA model
            if ema_model is not None and accelerator.sync_gradients:
                ema_model.step(model.parameters())

            # Check if accelerator has performed an optimization step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log metrics
                if accelerator.is_main_process:
                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                    if ema_model is not None:
                        logs["ema_decay"] = ema_model.cur_decay_value
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                # Save checkpoint
                if accelerator.is_main_process and global_step % cfg.checkpoint.checkpointing_steps == 0:
                    # Check if we should limit the number of checkpoints
                    if cfg.checkpoint.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(cfg.train.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # Remove old checkpoints if we exceed the limit
                        if len(checkpoints) >= cfg.checkpoint.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - cfg.checkpoint.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(cfg.train.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(cfg.train.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            # Break if we've reached the maximum number of steps
            if global_step >= max_train_steps:
                break

        progress_bar.close()
        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % cfg.checkpoint.save_images_epochs == 0 or epoch == cfg.train.num_epochs - 1:
                if ema_model is not None:
                    # Generate images with the EMA model
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                pipeline = DDPMPipelineGeneric(
                    unet=accelerator.unwrap_model(model),
                    scheduler=scheduler,
                )

                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                
                # Get variance normalization and self-conditioning settings from config
                enable_variance_norm = False
                variance_scale = 1.0
                enable_self_cond = True
                
                if hasattr(cfg.diffusion, "variance_normalization"):
                    enable_variance_norm = cfg.diffusion.variance_normalization.enabled
                    variance_scale = cfg.diffusion.variance_normalization.scale
                    
                if hasattr(cfg.diffusion, "self_conditioning"):
                    enable_self_cond = cfg.diffusion.self_conditioning.enabled
                
                # Run pipeline in inference (sample random noise and denoise)
                images = pipeline(
                    generator=generator,
                    batch_size=cfg.train.eval_batch_size,
                    num_inference_steps=cfg.diffusion.num_inference_steps,
                    output_type="np",
                    return_dict=False,
                    enable_self_conditioning=enable_self_cond,
                    enable_variance_normalization=enable_variance_norm,
                    variance_scale=variance_scale,
                )

                if ema_model is not None:
                    # Restore the original model
                    ema_model.restore(model.parameters())

                # Denormalize the images
                images_processed = (images[0] * 255).round().astype("uint8")

                # Log images to tracker
                if cfg.logging.logger == "tensorboard":
                    tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                elif cfg.logging.logger == "wandb":
                    # Log to wandb
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                        step=global_step,
                    )

            # Save model
            if epoch % cfg.checkpoint.save_model_epochs == 0 or epoch == cfg.train.num_epochs - 1:
                # Save the model
                pipeline = DDPMPipelineGeneric(
                    unet=accelerator.unwrap_model(model),
                    scheduler=scheduler,
                )

                pipeline.save_pretrained(cfg.train.output_dir)

                # Push to hub if enabled
                if cfg.hub.push_to_hub:
                    from huggingface_hub import create_repo, upload_folder

                    repo_id = create_repo(
                        repo_id=cfg.hub.hub_model_id or Path(cfg.train.output_dir).name,
                        exist_ok=True,
                        token=cfg.hub.hub_token,
                    ).repo_id

                    upload_folder(
                        repo_id=repo_id,
                        folder_path=cfg.train.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

    # End training
    accelerator.end_training()


if __name__ == "__main__":
    main()
