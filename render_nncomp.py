#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from NN_Comp.UNet import UNet
import pdb
import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, NN_comp=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    nn_comp_path = os.path.join(model_path, "NN_Comp{}.pth".format(iteration))

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    print(nn_comp_path)
    if NN_comp:
        NN_comp.load_state_dict(torch.load(nn_comp_path, map_location="cuda"))
    all_time = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        t1 = time.time()
        rendering = render(view, gaussians, pipeline, background)["render"]
        comp = NN_comp(rendering.unsqueeze(0)).squeeze(0)
        t2 = time.time()
        all_time.append(t2-t1)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(comp, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    print("Average time per image: {}".format(sum(all_time)/len(all_time)))
    print("Render FPS: {}".format(1/(sum(all_time)/len(all_time))))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        NN_comp = UNet(3,3).cuda()
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, NN_comp)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, NN_comp)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)