import numpy as np
import open3d as o3d
import argparse
import torch
from utils import read_ply, write_ply, graphFilter, data_sample
from Graph import GraphConstructor
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from IPython import embed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path_ply", help="path to ply file")
    parser.add_argument("-idx", "--index_file", type=str, help="index file")
    parser.add_argument("-s", "--save", default="", help="path to save output ply file")
    args = parser.parse_args()
    
    pcd_origin, _ = read_ply(args.path_ply)
    idxs = np.loadtxt(args.index_file, delimiter=',').astype(int)
    idxs = idxs - 1
    sampled_pc = pcd_origin[idxs]
    write_ply(args.save, sampled_pc)