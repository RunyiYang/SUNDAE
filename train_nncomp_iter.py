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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.S3IM import S3IM
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.tensorboard import SummaryWriter
from NN_Comp.UNet_old import UNet
# from NN_Comp.swin_unet import SUNet_model
import yaml
import pdb
TENSORBOARD_FOUND = True

def training(dataset, 
             opt, 
             pipe, 
             testing_iterations, 
             saving_iterations, 
             checkpoint_iterations, 
             checkpoint, 
             debug_from, 
             no_xyz=False, 
             sample_mode="FPRC", 
             sample_schedule='every 3000', 
             sample_rate=0.1, 
             kernal=103):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    NN_Comp = UNet(3, 3).cuda()
    with open('NN_Comp/NN_comp.yaml', 'r') as config:
        nn_comp_config = yaml.safe_load(config)
        
    # NN_Comp = SUNet_model(nn_comp_config).cuda()
    s3im = S3IM(s3im_patch_height=kernal)
    optimizer = torch.optim.Adam(NN_Comp.parameters(), lr=0.001, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [40000, 60000], gamma = 0.8)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, _) = torch.load(checkpoint)
        if no_xyz:
            gaussians.restore_no_xyz(model_params, opt)
        else:
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # import pdb; pdb.set_trace()
    if 'every' in sample_schedule:
        sample_only_once = False
        sample_freq = int(sample_schedule.split('-')[-1]) # NOTE: sample_schedule format should be: [every-N]
    else:
        sample_only_once = True
        sample_freq = 8000 # TODO:

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    scene.save(0)
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()
        optimizer.zero_grad()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        if iteration <= opt.densify_until_iter:
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = 1.0 - ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
            loss.backward()

        if iteration > opt.densify_until_iter * 2.0:
            image = NN_Comp(image.unsqueeze(0)).squeeze(0)
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = 1.0 - ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        # if iteration > opt.densify_until_iter*2:
        #     image = NN_Comp(image.unsqueeze(0)).squeeze(0)
        #     Ll1 = l1_loss(image, gt_image)
        #     ssim_loss = 1.0 - ssim(image, gt_image)
        # s3im_loss = s3im(image.flatten(1), gt_image.flatten(1))
        
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        # loss.backward()
        # optimizer.step()
        # lr_scheduler.step()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), NN_Comp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                # Graph Downsampling # TODO: add strategy for different sample frequencies
                if not sample_only_once:
                    if iteration % sample_freq == 0:
                        print("\n[ITER {}] Graph Downsampling with Sample Rate: {}".format(iteration, sample_rate))
                        gaussians.graph_downsampling(sample_mode, sample_rate) 
                else:
                    if iteration == sample_freq:
                        print("\n[ITER {}] Graph Downsampling with Sample Rate: {}".format(iteration, sample_rate))
                        gaussians.graph_downsampling(sample_mode, sample_rate) 
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                torch.save(NN_Comp.state_dict(), scene.model_path + "/NN_Comp" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, NN_Comp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('cuda_memory', torch.cuda.memory_allocated() / 1024**2, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    if iteration > 30000:
                        image = NN_Comp(image.unsqueeze(0)).squeeze(0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--no_xyz", type=bool, default=False, help="restore point cloud?")
    parser.add_argument("--ply_dir", type=str, default = None)
    parser.add_argument("--sample_rate", default=0.1, type=float, help="number of samples")
    parser.add_argument("--sample_mode", default="FPRC", type=str, help="sample mode")
    parser.add_argument("--sample_schedule", default="8k", type=str, help="sample schedule")
    parser.add_argument("--kernal", type=int, default=103)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, 
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
             args.debug_from, args.no_xyz, args.sample_mode, args.sample_schedule,
             args.sample_rate, args.kernal)

    # All done
    print("\nTraining complete.")
