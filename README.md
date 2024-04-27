# SUNDAE: <u>S</u>pectrally Pr<u>un</u>e<u>d</u> G<u>a</u>ussian Fi<u>e</u>lds with Neural Compensation

[[Paper]()] | [[Project Page](https://runyiyang.github.io/projects/SUNDAE/)] | [[3DGS Model]()]

This repository is an official implementation for:

**SUNDAE: <u>S</u>pectrally Pr<u>un</u>e<u>d</u> G<u>a</u>ussian Fi<u>e</u>lds with Neural Compensation**

> Authors:  [_Runyi Yang_](https://runyiyang.github.io/), [_Zhenxin Zhu_](https://github.com/jike5/), [_Zhou Jiang_](https://github.com/Jzian), _Baijun Ye_, _Xiaoxue Chen_, _Yifei Zhang_, [_Yuantao Chen_](https://tao-11-chen.github.io/), [_Jian Zhao†_](https://zhaoj9014.github.io/), [_Hao Zhao†_](https://sites.google.com/view/fromandto)
>
> † Corresponding Author(s)


## Introduction
Recently, 3D Gaussian Splatting, as a novel 3D representation, has garnered attention for its fast rendering speed and high rendering quality.  However, this comes with high memory consumption, e.g., a well-trained Gaussian field may utilize three million Gaussian primitives and over 700 MB of memory. We credit this high memory footprint to the lack of consideration for the **relationship** between primitives. In this paper, we propose a memory-efficient Gaussian field named SUNDAE with spectral pruning and neural compensation. On one hand, we construct a graph on the set of Gaussian primitives to model their **relationship** and design a spectral down-sampling module to prune out primitives while preserving desired signals. On the other hand, to compensate for the quality loss of pruning Gaussians, we exploit a lightweight neural network head to mix splatted features, which effectively compensates for quality losses while capturing the **relationship** between primitives in its weights. We demonstrate the performance of SUNDAE with extensive results. For example, SUNDAE can achieve 26.80 PSNR at 145 FPS using 104 MB memory while the vanilla Gaussian splatting algorithm achieves 25.60 PSNR at 160 FPS using 523 MB memory, on the Mip-NeRF360 dataset. 

## Installation


```
git clone --recursive git@github.com:RunyiYang/SUNDAE.git
```
## Environment
```shell
conda env create --file environment.yml
conda activate gaussian_downsampling
```

And several C++ libs are required for fast sampling, refer to <a href="graph_downsampling/graphScore/README.md">graphScore/README.md</a>.

## Train
For MipNeRF360 dataset, different resolutions are used for different scenes. For indoor scenes bonsai, counter, room, stump, use images_2 for training and evaluation. And for rest of the outdoor scenes, use images_4 for training and evaluation.

For example,
```
train_nncomp.py -s <dataset> -i images_4 -m <model_save_path>  --eval --checkpoint_iterations 30000 --sample_rate <float>
```

For other datasets, use the default set.

See <a href="bash/run_init.sh"> bash/run_init.sh</a> to fast start.

## Render
```
CUDA_VISIBLE_DEVICES=0 python render_nncomp.py -m <model_save_path>
```
See <a href="bash/render_init.sh"> bash/render_init.sh</a> to fast start.

## Evaluation
```
CUDA_VISIBLE_DEVICES=0 python metrics.py -m <model_save_path>
```
See <a href="bash/eval_init.sh"> bash/eval_init.sh</a> to fast start.