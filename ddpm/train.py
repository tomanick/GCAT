import torch
import diffusers
from datasets import load_dataset
from torchvision import transforms
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from diffusers.optimization import get_scheduler


class DiffusionModel(L.LightningModule):
    def __init__(self, timesteps=1000, sample_size=32):
        super().__init__()
        self.model = diffusers.models.UNet2DModel(
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            sample_size=sample_size,
            dropout=0.1,
        )
        self.scheduler = diffusers.schedulers.DDPMScheduler(num_train_timesteps=timesteps,
            variance_type="fixed_large", clip_sample=False, timestep_spacing="leading",
        )

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        pred_eps = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(pred_eps, noise)
        self.log("train_loss", loss, prog_bar=True)

        # Log the learning rate
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log("train/learning_rate", lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        pred_eps = self.model(noisy_images, steps).sample
        val_loss = torch.nn.functional.mse_loss(pred_eps, noise)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, 
                                      betas=(0.95, 0.999), weight_decay=1e-5,
                                      eps=1e-8)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        return [optimizer], [scheduler]
    

class DiffusionData(L.LightningDataModule):
    def __init__(self, batch_size=128, img_size=32):
        super().__init__()

        self.batch_size = batch_size
        
        self.augment = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            #transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        load_dataset("./data/cifar10")

    def train_dataloader(self):
        dataset = load_dataset("./data/cifar10")
        dataset.set_transform(lambda sample: {"images": [self.augment(image) for image in sample["img"]]})
        return torch.utils.data.DataLoader(dataset["train"], batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        dataset = load_dataset("./data/cifar10")
        dataset.set_transform(lambda sample: {"images": [self.augment(image) for image in sample["img"]]})
        return torch.utils.data.DataLoader(dataset["test"], batch_size=self.batch_size, shuffle=False, num_workers=4)


from lightning.pytorch.loggers import TensorBoardLogger


if __name__ == "__main__":

    def create_checkpoint_callback(trainer, monitor="val_loss", mode="min"):
        return ModelCheckpoint(
                    monitor=monitor,
                    dirpath=f"{trainer.logger.log_dir}/checkpoints",
                    filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",
                    save_top_k=1,
                    mode=mode,
                    verbose=True,
                )

    # specify the log directory
    logger = TensorBoardLogger(save_dir="lightning_logs/cifar10/", name="ddpm")
    
    model = DiffusionModel(timesteps=1000, sample_size=32)
    
    data = DiffusionData(batch_size=128)
    
    trainer = L.Trainer(max_epochs=200, logger=logger, precision="16")

    # create ModelCheckpoint
    checkpoint_callback = create_checkpoint_callback(trainer)
    trainer.callbacks.append(checkpoint_callback)

    # train
    trainer.fit(model, data)
