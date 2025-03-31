# coding: utf-8

import os
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image


# custom dataset
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        with Image.open(img_path) as img:
            if self.transform:
                img = self.transform(img)
        return img


def main(real_dir, gen_dir, batch_size=1000):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define image transformation
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # load datasets
    real_dataset = ImageDataset(real_dir, transform)
    generated_dataset = ImageDataset(gen_dir, transform)

    # create dataloaders
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    generated_loader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # initialize FID
    fid = FrechetInceptionDistance(normalize=True).to(device)

    # update FID with real images
    for real_batch in real_loader:
        real_batch = real_batch.to(device)
        fid.update(real_batch, real=True)

    # update FID with generated images
    for generated_batch in generated_loader:
        generated_batch = generated_batch.to(device)
        fid.update(generated_batch, real=False)

    # compute FID
    fid_score = float(fid.compute())
    print(f"fid: {fid_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True, help="path to real images")
    parser.add_argument("--gen_dir", type=str, required=True, help="path to generated images")
    parser.add_argument("--batch_size", type=int, default=1000, help="batch size for dataloaders")
    args = parser.parse_args()

    main(args.real_dir, args.gen_dir, args.batch_size)
