# custom_dataset.py

from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import os
from IPython import embed
from torchvision import transforms
import numpy as np

transform = transforms.Compose([
    transforms.Resize((1024, 1600)),
    transforms.ToTensor(),
])

class MipNeRFImages(Dataset):
    def __init__(self, gt_data):
        self.image_paths = []
        
        # Traverse through the main directory and its subdirectories
        for subdir, _, files in os.walk(gt_data):
            for file in files:
                if file.endswith(('.png','jpg','JPG')):
                    self.image_paths.append(os.path.join(subdir, file))
        self.image_paths_pre = [i.replace('gt', 'comp') for i in self.image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_gt = Image.open(self.image_paths[idx])
        image_render = Image.open(self.image_paths_pre[idx])
        
        # Convert the PIL image to a PyTorch tensor or apply other transformations
        if self.transform:
            image_gt = self.transform(image_gt)
            image_render = self.transform(image_render)
            
        return self.image_paths_pre[idx], self.image_paths[idx], image_render, image_gt


def get_image_shapes(directory):
    image_shapes = []

    # Traverse through the main directory and its subdirectories
    for subdir, _, files in os.walk(directory):
        for file in files:
            # Load only .png images
            if file.endswith('.JPG') or file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(subdir, file)
                with Image.open(image_path) as img:
                    # Convert the PIL image to a numpy array and store its shape
                    image_shapes.append(np.array(img).shape)

    return image_shapes

if __name__ == "__main__":
    directory_path = "/DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/train/ours_0/gt/"
    # "/DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/train/ours_0/gt/"
    # "/DATA_EDS2/yangry/dataset/nerfstudio-data-mipnerf360/bicycle/"  # Replace with your directory path

    dataset = MipNeRFImages(directory_path)
   
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for images in dataloader:
        print(images.shape)
        
    
