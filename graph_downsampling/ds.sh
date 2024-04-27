#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python sample_point_cloud.py /DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/point_cloud/iteration_30000/point_cloud.ply -r 0.3 -s /DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/point_cloud/iteration_30/point_cloud.ply &

# CUDA_VISIBLE_DEVICES=0 python sample_point_cloud.py /DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/point_cloud/iteration_30000/point_cloud.ply -r 0.3 -s /DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/point_cloud/iteration_31/point_cloud.ply -m random &

# python sample_from_idx.py -p /DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/point_cloud/iteration_30000/point_cloud.ply -idx /DATA_EDS2/yangry/experiments/downsampled/bicycle/sample_idx/Sampledidx1601718.txt -s /DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/point_cloud/iteration_32/point_cloud.ply &

# CUDA_VISIBLE_DEVICES=1 python sample_point_cloud.py /DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/point_cloud/iteration_30000/point_cloud.ply -r 0.5 -s /DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/point_cloud/iteration_50/point_cloud.ply &

# CUDA_VISIBLE_DEVICES=1 python sample_point_cloud.py /DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/point_cloud/iteration_30000/point_cloud.ply -r 0.5 -s /DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/point_cloud/iteration_51/point_cloud.ply -m random &

# python sample_from_idx.py -p /DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/point_cloud/iteration_30000/point_cloud.ply -idx /DATA_EDS2/yangry/experiments/downsampled/bicycle/sample_idx/Sampledidx2669531.txt -s /DATA_EDS2/yangry/experiments/mipnerf360/bicycle/origin/point_cloud/iteration_52/point_cloud.ply 

python test_ctypes.py /DATA_EDS2/yangry/experiments/mipnerf360_full/bicycle/point_cloud/iteration_30000/point_cloud.ply -r 0.3 -s /DATA_EDS2/yangry/experiments/mipnerf360/graph_30_new_net_moreiter/bicycle/only_sample/point_cloud/iteration_601/point_cloud.ply -w 0.1 &

python test_ctypes.py /DATA_EDS2/yangry/experiments/mipnerf360_full/bicycle/point_cloud/iteration_30000/point_cloud.ply -r 0.3 -s /DATA_EDS2/yangry/experiments/mipnerf360/graph_30_new_net_moreiter/bicycle/only_sample/point_cloud/iteration_600/point_cloud.ply -m random &

python test_ctypes.py /DATA_EDS2/yangry/experiments/mipnerf360_full/bicycle/point_cloud/iteration_30000/point_cloud.ply -r 0.3 -s /DATA_EDS2/yangry/experiments/mipnerf360/graph_30_new_net_moreiter/bicycle/only_sample/point_cloud/iteration_603/point_cloud.ply -w 0.3 &

python test_ctypes.py /DATA_EDS2/yangry/experiments/mipnerf360_full/bicycle/point_cloud/iteration_30000/point_cloud.ply -r 0.3 -s /DATA_EDS2/yangry/experiments/mipnerf360/graph_30_new_net_moreiter/bicycle/only_sample/point_cloud/iteration_605/point_cloud.ply -w 0.5 &

python test_ctypes.py /DATA_EDS2/yangry/experiments/mipnerf360_full/bicycle/point_cloud/iteration_30000/point_cloud.ply -r 0.3 -s /DATA_EDS2/yangry/experiments/mipnerf360/graph_30_new_net_moreiter/bicycle/only_sample/point_cloud/iteration_607/point_cloud.ply -w 0.7 &

python test_ctypes.py /DATA_EDS2/yangry/experiments/mipnerf360_full/bicycle/point_cloud/iteration_30000/point_cloud.ply -r 0.3 -s /DATA_EDS2/yangry/experiments/mipnerf360/graph_30_new_net_moreiter/bicycle/only_sample/point_cloud/iteration_609/point_cloud.ply -w 0.9