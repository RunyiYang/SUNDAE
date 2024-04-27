# SUNDAE: <u>S</u>pectrally Pr<u>un</u>e<u>d</u> G<u>a</u>ussian Fi<u>e</u>lds with Neural Compensation

[[Paper]()] | [[Project Page](https://runyiyang.github.io/projects/SUNDAE/)] | [[3DGS Model]()]

This repository is an official implementation for:

**SUNDAE: <u>S</u>pectrally Pr<u>un</u>e<u>d</u> G<u>a</u>ussian Fi<u>e</u>lds with Neural Compensation**

> Authors:  [_Runyi Yang_](https://runyiyang.github.io/), [_Zhenxin Zhu_](https://github.com/jike5/), [_Zhou Jiang_](https://github.com/Jzian), _Baijun Ye_, _Xiaoxue Chen_, _Yifei Zhang_, [_Yuantao Chen_](https://tao-11-chen.github.io/), [_Jian Zhao†_](https://zhaoj9014.github.io/), [_Hao Zhao†_](https://sites.google.com/view/fromandto)
>
> † Corresponding Author(s)


## Abstract
Recently, 3D Gaussian Splatting, as a novel 3D representation, has garnered attention for its fast rendering speed and high rendering quality.  However, this comes with high memory consumption, e.g., a well-trained Gaussian field may utilize three million Gaussian primitives and over 700 MB of memory. We credit this high memory footprint to the lack of consideration for the **relationship** between primitives. In this paper, we propose a memory-efficient Gaussian field named SUNDAE with spectral pruning and neural compensation. On one hand, we construct a graph on the set of Gaussian primitives to model their **relationship** and design a spectral down-sampling module to prune out primitives while preserving desired signals. On the other hand, to compensate for the quality loss of pruning Gaussians, we exploit a lightweight neural network head to mix splatted features, which effectively compensates for quality losses while capturing the **relationship** between primitives in its weights. We demonstrate the performance of SUNDAE with extensive results. For example, SUNDAE can achieve 26.80 PSNR at 145 FPS using 104 MB memory while the vanilla Gaussian splatting algorithm achieves 25.60 PSNR at 160 FPS using 523 MB memory, on the Mip-NeRF360 dataset. 

## Installation


```
git clone --recursive git@github.com:RunyiYang/SUNDAE.git
conda env create --file environment.yml
conda activate SUNDAE
```

Several C++ libs are required for fast sampling, refer to <a href="[graph_downsampling/graphScore/README.md](https://github.com/RunyiYang/GraphDownsampling/blob/master/README.md)">GraphDownsampling</a>.

```shell
cd GraphDownsampling/graphScore
g++ -shared -o filtergraphv2.so pyGraphFilterV2.cpp graphFilter.cpp pccProcessing.cpp -I./ -I/usr/local/include/eigen3/ -O3 -fopenmp -fPIC -Wall
```

## Run
### Train
For MipNeRF360 dataset, different resolutions are used for different scenes. For indoor scenes, use images_2 for training and evaluation. And for rest of the outdoor scenes, use images_4 for training and evaluation.
For other datasets, please follow the default setting.

For example,
```
train.py -s <dataset> -i images_4 -m <model_save_path>  --eval --checkpoint_iterations 30000 --sample_rate <float>
```

### Render
```
python render.py -m <model_save_path>
```
### Evaluation
```
python metrics.py -m <model_save_path>
```
### Viewer
To enable viewer, please refer to <a href="https://github.com/RunyiYang/SUNDAE-viewer/blob/master/README.md">SUNDAE-Viewer</a>

## Acknowledgement
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). Please follow the license of 3DGS. We thank all the authors for their great work and repos. 