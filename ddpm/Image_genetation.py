import math
import torch
import torchvision as tv
import diffusers
import jsonargparse
from pathlib import Path
from train import DiffusionModel
#from train_GCAT import DiffusionModel_GCAT as DiffusionModel


def main(
    checkpoint: Path = Path("checkpoints/best-checkpoint.ckpt"),
    num_timesteps: int = 1000,
    num_samples: int = 10000,
    batch_size: int = 2000,
    seed: int = 17,
    output_dir: Path = Path("generated_images"),
    generated_size: int = 32,
):
    """
    Generate images from a trained diffusion model.

    Args:
        checkpoint: Path to the model checkpoint.
        num_timesteps: Number of denoising steps.
        num_samples: Total number of images to generate.
        batch_size: Number of images per generation batch.
        seed: Random seed.
        output_dir: Path to save generated images.
        generated_size: Size of generated images.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(exist_ok=True, parents=True)

    with device:
        model = DiffusionModel(timesteps=num_timesteps, sample_size=generated_size)
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scheduler = diffusers.schedulers.DDPMScheduler(num_train_timesteps=num_timesteps, variance_type="fixed_large", timestep_spacing="leading")
    pipe = diffusers.DDPMPipeline(model.model, scheduler)
    pipe = pipe.to(device=device)

    total_batches = math.ceil(num_samples / batch_size)

    with torch.inference_mode():
        for batch_idx in range(total_batches):
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            print(f"Generating batch {batch_idx + 1}/{total_batches} with {current_batch_size} samples...")

            (pil_images,) = pipe(
                batch_size=current_batch_size,
                num_inference_steps=num_timesteps,
                output_type="pil",
                return_dict=False,
            )

            # save each image in the batch individually
            for idx, pil_image in enumerate(pil_images):
                img_idx = batch_idx * batch_size + idx
                filename = output_dir / f"image_{img_idx:04d}.png"
                pil_image.save(filename)

    print(f"Generated {num_samples} images saved to {output_dir}")


if __name__ == "__main__":
    jsonargparse.CLI(main)
