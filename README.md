# 3D Gaussian Splatting Downsampling and NN Compensation
```
git clone --recursive git@github.com:RunyiYang/3dgs-downsample-backbone.git

git checkout origin pipeline
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