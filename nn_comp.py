import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from NN_Comp.UNet_old import UNet
from NN_Comp.Dataloader import MipNeRFImages
from argparse import ArgumentParser, Namespace
from tqdm.contrib import tenumerate
from tqdm.auto import trange
from tqdm import tqdm
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as tf
import os
import torchvision
from torch.utils.tensorboard import SummaryWriter
from IPython import embed
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir + fname)
        gt = Image.open(gt_dir + fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(data_paths):
    print("Evaluation Metrics!")
    
    gt_dir = data_paths
    renders_dir = data_paths.replace('gt', 'comp')
    renders, gts, image_names = readImages(renders_dir, gt_dir)

    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model_path", type=str, default = None)
    parser.add_argument("--unet", type=str)
    parser.add_argument("--dataset_gt", type=str)
    parser.add_argument("--data", type=str, default = None)
    parser.add_argument("--device", type=str, default = "cuda")
    args = parser.parse_args()
    writer = SummaryWriter(log_dir=args.unet)
    
    model = UNet()
    model.to(args.device)
    
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print("Loaded model!", args.model_path)
    
    if args.mode == "train":
        model.train()
        dataset = MipNeRFImages(args.dataset_gt)
        dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
        criterion = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        total_loss = 0.0
        steps = 0
        for epoch in trange(args.epochs):
        #     if epoch % 10 == 0 or epoch == 0:
        #         ssims, psnrs, lpipss = [], [], []
        #         for batch_idx, (_ , _, input, gts) in tenumerate(dataloader, leave=False):
        #             gts = gts.to(args.device)
        #             input = input.to(args.device)
        #             renders = model(input)
        #             ssims.append(ssim(renders, gts))
        #             psnrs.append(psnr(renders, gts))
        #             lpipss.append(lpips(renders, gts, net_type='vgg'))

        #         print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        #         print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        #         print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        #         writer.add_scalar("ssim", torch.tensor(ssims).mean(), epoch)
        #         writer.add_scalar("psnr", torch.tensor(psnrs).mean(), epoch)
        #         writer.add_scalar("lpips", torch.tensor(lpipss).mean(), epoch)
            for batch_idx, (_, _, inputs, targets) in tenumerate(dataloader, leave=False):
                steps+=1
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                optimizer.zero_grad()  # Zero the gradient buffers
                # Forward pass
                outputs = model(inputs)
                
                # Compute the photometric loss
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                writer.add_scalar("loss", loss.detach(), steps)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()

        writer.close()
        save_path = args.unet + "model.pth" 
        torch.save(model.state_dict(), save_path)
        print("Saved model!", save_path)
    
    elif args.mode == "render":
        model.eval()
        dataset = MipNeRFImages(args.dataset_gt)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch_idx, (inputs_path, gt_path, inputs, targets) in tenumerate(dataloader, leave=False):
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                outputs = model(inputs).squeeze()
                
                # save the images
                output_path = inputs_path[0].replace('renders', 'comp')
                
                torchvision.utils.save_image(outputs, output_path)
    
    elif args.mode == "metric":
        dataset = MipNeRFImages(args.dataset_gt)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        ssims = []
        psnrs = []
        lpipss = [] 
        for batch_idx, (inputs_path, gt_path, gts, renders) in tenumerate(dataloader, leave=False):
            gts = gts.to(args.device)
            renders = renders.to(args.device)
            ssims.append(ssim(renders, gts))
            psnrs.append(psnr(renders, gts))
            lpipss.append(lpips(renders, gts, net_type='vgg'))

        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("") 

                
                
    

    
    