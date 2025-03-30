import torch
import torch.nn as nn
import diffusers
from datasets import load_dataset
from torchvision import models, transforms, datasets
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import TensorBoardLogger


def get_feature_extractor(model_name="resnet50"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name == "resnet50":
        feature_extractor = models.resnet50(pretrained=True)
        # remove the last fc layer
        modules = list(feature_extractor.children())[:-1]
        feature_extractor = nn.Sequential(*modules)

    elif model_name == "resnet101":
        feature_extractor = models.resnet101(weights=None)
        # remove the last fc layer
        modules = list(feature_extractor.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
    
    elif model_name == "efficientnet_v2_s":
        feature_extractor = models.efficientnet_v2_s(weights='DEFAULT')
        # remove the last fc layer
        modules = list(feature_extractor.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
    
    elif model_name == "mobilenet_v3_large":
        feature_extractor = models.mobilenet_v3_large(weights='DEFAULT')
        # remove the last fc layer
        modules = list(feature_extractor.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
    
    elif model_name == "mobilenet_v3_small":
        feature_extractor = models.mobilenet_v3_small(weights='DEFAULT')
        # remove the last fc layer
        modules = list(feature_extractor.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
    
    elif model_name == "shufflenet_v2_x0_5":
        feature_extractor = models.shufflenet_v2_x0_5(weights='DEFAULT')
        # remove the last fc layer
        modules = list(feature_extractor.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        
    elif model_name == "shufflenet_v2_x1_0":
        feature_extractor = models.shufflenet_v2_x1_0(weights='DEFAULT')
        # remove the last fc layer
        modules = list(feature_extractor.children())[:-1]
        feature_extractor = nn.Sequential(*modules)

    elif model_name == "inception_v3":
        feature_extractor = models.inception_v3(weights='DEFAULT').to(device)
        # remove the final classification layer
        feature_extractor.fc = torch.nn.Identity()  
    
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    feature_extractor.eval()  # set the model to evaluation mode
    return feature_extractor


class FeatureDistributionLoss(nn.Module):
    def __init__(self, feature_extractor, eps=1e-6, inception=False):
        super(FeatureDistributionLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.eps = eps

        if inception == True:
            self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),  # resize for ResNet-50, MobileNet, and ShuffleNet
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize using ImageNet mean and std
            ])

    def precompute_features(self, images, chunk_size=128):
        features_list = []

        # process images in chunks
        for i in range(0, images.size(0), chunk_size):
            chunk = images[i:i + chunk_size]
            
            chunk_preprocessed = torch.stack([self.preprocess(img) for img in chunk])
            
            with torch.no_grad():
                chunk_features = self.feature_extractor(chunk_preprocessed).view(chunk_preprocessed.size(0), -1)
            features_list.append(chunk_features)
        
        features = torch.cat(features_list, dim=0)
    
        # compute mean and covariance of the features
        mean = torch.mean(features, dim=0)
        features_centered = features - mean
        cov = torch.mm(features_centered.T, features_centered) / (features.size(0) - 1)
        cov += self.eps * torch.eye(cov.size(0), device=cov.device)  # add small value to diagonal for numerical stability
    
        return mean.detach(), cov.detach(), features
    
    def compute_unbiased_mmd(self, features_real, features_perturbed, kernel_bandwidths=None):
        """
        Computes the unbiased MMD loss between real and perturbed features.
        
        Args:
            features_real: Tensor of shape (n, d), real samples
            features_perturbed: Tensor of shape (m, d), perturbed samples
            kernel_bandwidths: List of bandwidths for RBF kernels. If None, use median heuristic.
            
        Returns:
            mmd_loss: Scalar tensor representing the MMD loss
        """
        def rbf_kernel(x, y, bandwidth):
            dist_sq = torch.cdist(x, y, p=2) ** 2
            return torch.exp(-dist_sq / (2 * bandwidth ** 2))
        
        n = features_real.size(0)
        m = features_perturbed.size(0)
        
        if kernel_bandwidths is None:
            # median heuristic for bandwidth
            combined = torch.cat([features_real, features_perturbed], dim=0)
            pairwise_dist = torch.cdist(combined, combined, p=2)
            median_dist = torch.median(pairwise_dist)
            kernel_bandwidths = [median_dist.item()]
        elif kernel_bandwidths == "multi":
            # multi choices for bandwidth
            combined = torch.cat([features_real, features_perturbed], dim=0)
            pairwise_dist = torch.cdist(combined, combined, p=2).flatten()
            # calculate percentiles
            percentiles = [25, 50, 75, 90]
            kernel_bandwidths = [torch.quantile(pairwise_dist, p / 100).item() for p in percentiles]
                    
        mmd_loss = 0
        for bandwidth in kernel_bandwidths:
            Kxx = rbf_kernel(features_real, features_real, bandwidth)
            Kyy = rbf_kernel(features_perturbed, features_perturbed, bandwidth)
            Kxy = rbf_kernel(features_real, features_perturbed, bandwidth)
            
            # remove diagonal elements for unbiased estimation
            Kxx = Kxx - torch.diag(torch.diag(Kxx))
            Kyy = Kyy - torch.diag(torch.diag(Kyy))
            
            mmd_real = Kxx.sum() / (n * (n - 1))
            mmd_perturbed = Kyy.sum() / (m * (m - 1))
            mmd_cross = Kxy.sum() / (n * m)
            
            mmd_loss += mmd_real + mmd_perturbed - 2 * mmd_cross
        
        mmd_loss /= len(kernel_bandwidths)
        return mmd_loss
        
    def compute_loss(self, real_mean, real_cov, real_features, generated_images, chunk_size=128):
        """
        Compute the FDL between real and generated images.
        Args:
            real_mean (Tensor): Precomputed mean of real features.
            real_cov (Tensor): Precomputed covariance of real features.
            generated_images (Tensor): Batch of generated images.
            chunk_size (int): Size of chunks to process images in smaller batches.
        Returns:
            fdl_loss (Tensor): FDL loss value.
        """
        generated_features_list = []
    
        # process generated images in chunks
        for i in range(0, generated_images.size(0), chunk_size):
            chunk = generated_images[i:i + chunk_size]
            
            chunk_preprocessed = torch.stack([self.preprocess(img) for img in chunk])
            
            chunk_features = self.feature_extractor(chunk_preprocessed).view(chunk_preprocessed.size(0), -1)
            generated_features_list.append(chunk_features)
        
        generated_features = torch.cat(generated_features_list, dim=0)

        fdl_loss = self.compute_unbiased_mmd(real_features, generated_features, kernel_bandwidths="multi")

        # safeguard to prevent NaN values
        if torch.isnan(fdl_loss).any() or torch.isinf(fdl_loss).any():
            return None

        fdl_loss = torch.clamp(fdl_loss, min=1e-6, max=1000.0)

        safe_fdl_loss = -fdl_loss
        
        # ensure safe_fdl_loss has requires_grad=True
        safe_fdl_loss = safe_fdl_loss.requires_grad_(True).to(generated_images.device)
        return safe_fdl_loss


class DiffusionModel_GCAT(L.LightningModule):
    def __init__(self, timesteps=1000, epsilon=0.3, malicious_epsilon=0.3, model_name="mobilenet_v3_large", pretrained=None, num_iterations=10, lr=0.0001, wd=0.00001, sample_size=32):
        super().__init__()
        self.model = diffusers.models.UNet2DModel(
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            sample_size=sample_size,
        )
        self.scheduler = diffusers.schedulers.DDPMScheduler(num_train_timesteps=1000,
            variance_type="fixed_large", clip_sample=False, timestep_spacing="leading",
        )

        self.t_range = timesteps

        self.epsilon = epsilon
        self.malicious_epsilon = malicious_epsilon

        self.num_iterations = num_iterations 
        self.lr = lr
        self.wd = wd
        self.prev_eta = None
        
        # adversarial Training
        if pretrained is None:
            self.feature_extractor = get_feature_extractor(model_name = model_name).to(self.device)
        else:
            self.feature_extractor = get_pretrained_feature_extractor(model_name = model_name, pretrained_weight_path=pretrained).to(self.device)

        if model_name == "inception_v3":
            self.fdl_loss_fn = FeatureDistributionLoss(self.feature_extractor, inception=True).to(self.device)
        else:
            self.fdl_loss_fn = FeatureDistributionLoss(self.feature_extractor, inception=False).to(self.device)

    def get_combined_noise_loss(self, batch, batch_idx, perturbations):
        """
        Compute the loss between the true combined noise and the predicted combined noise.
        """
        
        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=batch.device)
        epsilons = torch.randn(batch.shape, device=batch.device)

        combined_noise = epsilons + perturbations

        # log combined_noise as histogram
        self.logger.experiment.add_histogram("histogram/combined_noise", combined_noise, self.global_step)

        noisy_images = self.scheduler.add_noise(batch, combined_noise, ts)

        # predict combined_noise
        pred_combined_noise = self.model(noisy_images, ts).sample
    
        # compute the loss
        loss = torch.nn.functional.mse_loss(pred_combined_noise, combined_noise)
        
        return loss


    def remove_adversarial_perturbation_training(self, x, ts):
        """
        Remove adversarial perturbations from the input images.
    
        Parameters:
        - x: Input batch of adversarial images.
        - ts: Time steps for the reverse diffusion process.
    
        Returns:
        - x_denoised: Tensor of denoised images.
        """
        # ensure ts has the correct shape and length
        assert len(ts) == x.size(0), "Length of ts must match the batch size of x"
        #ts = ts.view(-1)
        
        perturbed_images = x.clone()

        e_hat = self.model(perturbed_images, ts).sample

        x_denoised = perturbed_images - e_hat
        
        return x_denoised

    def beta_attack_diffusion(self, x, epsilon, num_iterations, lr=0.0001, wd=1e-4, reuse_eta=None):
        
        # set the device where the computations will be performed
        device = x.device

        # create a clone of the original images and ensure it is detached from the computation graph
        # original_images [0, 1]
        original_images = x.clone().detach().to(device)

        # ensure the original images require gradients
        if not original_images.requires_grad:
            original_images.requires_grad = True

        # reuse eta
        if reuse_eta is not None:
            eta = reuse_eta.clone().to(device)
        else:
            # Initialize the perturbation tensor eta with zeros, having the same shape as x
            eta = torch.zeros_like(x, device=device, requires_grad=True)

        # ensure eta requires gradients
        if not eta.requires_grad:
            eta.requires_grad = True

        # initialize the optimizer to optimize eta using Adam with a learning rate of alpha
        optimizer = torch.optim.AdamW([eta], lr=lr, weight_decay=wd)
        
        # precompute the real features and real mean
        real_mean, real_cov, real_features = self.fdl_loss_fn.precompute_features(original_images)
        
        # iteratively update eta for a specified number of iterations
        for iteration in range(num_iterations):

            # generate random timesteps for each image in the batch
            ts = torch.randint(0, self.t_range, [x.shape[0]], device=device)

            # zero the gradients of the optimizer
            optimizer.zero_grad()

            # create perturbed images by adding eta to the original images
            perturbed_images = original_images + eta
            
            # forward pass: Generate denoised images from the perturbed images
            recon_images = self.remove_adversarial_perturbation_training(perturbed_images, ts)
            
            # compute the FDL loss between adv_imgs and original images
            fdl_loss = self.fdl_loss_fn.compute_loss(real_mean, real_cov, real_features, recon_images)

            if fdl_loss is None:
                continue
            
            # ensure the original images require gradients
            if not fdl_loss.requires_grad:
                fdl_loss.requires_grad = True

            # compute gradients of the loss with respect to eta
            fdl_loss.backward(retain_graph=True)

            # perform an optimization step to update eta
            optimizer.step()

            # clamp the perturbation eta to ensure it stays within the valid range [-epsilon, epsilon]
            eta.data = torch.clamp(eta.data, -epsilon, epsilon)
            
            self.log("train/FDL_loss", fdl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        adv_imgs = perturbed_images
                
        # return the final perturbation eta after detaching it from the computation graph
        return adv_imgs, eta.detach()

    def collaborative_loss(self, recon_loss, adv_recon_loss, margin_weight=1.0):
        return recon_loss + margin_weight * adv_recon_loss

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        #images, labels = batch
        
        noise = torch.randn_like(images)

        # log gaussian noise as histogram
        self.logger.experiment.add_histogram("histogram/gaussian_noise", noise, self.global_step)
        
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        # predict the noise
        pred_eps = self.model(noisy_images, steps).sample

        epsilon_loss = torch.nn.functional.mse_loss(pred_eps, noise)

        _, perturbations = self.beta_attack_diffusion(images, epsilon=self.epsilon, num_iterations=self.num_iterations, lr=self.lr, wd=self.wd, reuse_eta=self.prev_eta)

        self.prev_eta = perturbations

        # log eta as histogram
        self.logger.experiment.add_histogram("histogram/eta", perturbations, self.global_step)

        # compute perturbation loss on adversarial images
        adv_recon_loss = self.get_combined_noise_loss(images, batch_idx, perturbations)

        # compute collaborative loss
        total_loss = self.collaborative_loss(epsilon_loss, adv_recon_loss)

        # log eta (perturbations) statistics
        self.log("train/eta_min", perturbations.min(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/eta_max", perturbations.max(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/eta_mean", perturbations.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # log the losses
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/epsilon_loss", epsilon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/combined_noise_loss", adv_recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # log the learning rate
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log("train/learning_rate", lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        #images, labels = batch
        
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
            #transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        load_dataset("./data/cifar10")

    def train_dataloader(self):
        dataset = load_dataset("./data/cifar10")
        dataset.set_transform(lambda sample: {"images": [self.augment(image) for image in sample["img"]]})
        return torch.utils.data.DataLoader(dataset["train"], batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)

    def val_dataloader(self):
        dataset = load_dataset("/home/xdjf/Desktop/Diffusion/ddpm/data/cifar10")
        dataset.set_transform(lambda sample: {"images": [self.augment(image) for image in sample["img"]]})
        return torch.utils.data.DataLoader(dataset["test"], batch_size=self.batch_size, shuffle=False, num_workers=4)


if __name__ == "__main__":

    def create_checkpoint_callback(trainer, monitor="val_loss", mode="min"):
        return ModelCheckpoint(
                    monitor=monitor,
                    dirpath=f"{trainer.logger.log_dir}/checkpoints",  # 动态设置路径
                    filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",
                    save_top_k=1,
                    mode=mode,
                    verbose=True,
                )

    # specify the log directory
    logger = TensorBoardLogger(save_dir="./DDPM_GCAT_logs/cifar10/", name="GCAT")
    
    model = DiffusionModel_GCAT(timesteps=1000, epsilon=0.3, malicious_epsilon=0.3, model_name="mobilenet_v3_large", num_iterations=10, lr=0.0001, wd=0.00001, sample_size=32)

    data = DiffusionData(batch_size=128, img_size=32)
    
    trainer = L.Trainer(max_epochs=200, logger=logger, precision="16")

    # create ModelCheckpoint
    checkpoint_callback = create_checkpoint_callback(trainer)
    trainer.callbacks.append(checkpoint_callback)

    # train
    trainer.fit(model, data)
