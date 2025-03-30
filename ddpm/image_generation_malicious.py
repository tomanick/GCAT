import math
import torch
import torchvision as tv
import diffusers
import jsonargparse
from pathlib import Path
from train import DiffusionModel
#from train_GCAT import DiffusionModel

from tqdm import tqdm
from PIL import Image


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def sample_simulate_malicious_attack(sample_batch_size, num_inference_steps, model, scheduler, img_depth=3, img_height=32, img_width=32, malicious_epsilon=0.3, add_noise=True):
    """
    Denoise the noisy images using the reverse diffusion process with malicious attack simulation.
    
    Parameters:
    - noisy_images: Tensor of noisy images to be denoised, shape (batch_size, channels, height, width).
    - ts: Tensor of timesteps for each image, shape (batch_size,).
    - malicious_epsilon: Magnitude of malicious noise to be added.
    - add_noise: Boolean flag to determine whether to add malicious noise.

    Returns:
    - x_denoised: Tensor of denoised images.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # clone input images to avoid modifying the original
    x = torch.randn((sample_batch_size, img_depth, img_height, img_width)).to(device)
    
    # set step values
    scheduler.set_timesteps(num_inference_steps)

    current_image = x

    # reverse diffusion process from timestep t to 0
    for j in tqdm(scheduler.timesteps, desc="Denoising steps"):
        with torch.no_grad():
            pred_noise = model(current_image, j).sample  # predict noise at timestep j
            step_output = scheduler.step(pred_noise, j, current_image)  # perform reverse step
            current_image = step_output.prev_sample  # update the current sample
            
        # add malicious noise every 100 steps
        if add_noise and j % 100 == 0 and j > 0:
            malicious_noise = torch.randn_like(current_image) * malicious_epsilon
            # introduce malicious noise
            current_image += malicious_noise

    # convert to [0, 1]
    final_samples = (current_image.clamp(-1, 1) + 1) / 2
    final_samples = final_samples.cpu().permute(0, 2, 3, 1).numpy()
    
    final_samples = numpy_to_pil(final_samples)

    return final_samples


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
    Generate images under malicious attacks from a trained diffusion model.

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
    
    # load model
    with device:
        model = DiffusionModel(timesteps=num_timesteps, sample_size=generated_size)
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # prepare pipeline
    scheduler = diffusers.schedulers.DDPMScheduler(variance_type="fixed_large", timestep_spacing="leading")
    pipe = diffusers.DDPMPipeline(model.model, scheduler).to(device=device)

    total_batches = math.ceil(num_samples / batch_size)

    with torch.inference_mode():
        for batch_idx in range(total_batches):
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            print(f"Generating batch {batch_idx + 1}/{total_batches} with {current_batch_size} samples...")
            
            pil_images = sample_simulate_malicious_attack(
                sample_batch_size=current_batch_size,
                num_inference_steps=num_timesteps,
                model=model.model,
                scheduler=scheduler,
                img_depth=3,
                img_height=32,
                img_width=32,
                malicious_epsilon=0.3,
                add_noise=True
            )

            # save each image in the batch individually
            for idx, pil_image in enumerate(pil_images):
                img_idx = batch_idx * batch_size + idx
                filename = output_dir / f"image_{img_idx:04d}.png"
                pil_image.save(filename)

    print(f"Generated {num_samples} images saved to {output_dir}")


if __name__ == "__main__":
    jsonargparse.CLI(main)