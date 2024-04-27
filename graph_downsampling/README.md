# 1、编译.so文件

```
cd graphScore
g++ -shared -o filtergraphv2.so pyGraphFilterV2.cpp graphFilter.cpp pccProcessing.cpp -I./ -I/usr/local/include/eigen3/   -O3 -fopenmp -fPIC -Wall
```


# 2、运行
```
python test_ctypes.py ply/point_cloud.ply
```

输出如下时间测试：
```
Time taken to read ply:  0.3337249755859375 seconds
Processing 5339063 points with sample rate of 0.1
x1: 2.07133 y1: 2.02422 z1: -1.16529
x2: 1.48653 y2: 1.41477 z2: 1.65127
x3: 1.43754 y3: 1.37962 z3: 1.16425
Point Cloud size in C++: 5339063
Time: numpyArrayToDoubleArray and loadBlock cost 318 ms
Time: findNNdistances cost 12012 ms
Time: buildGraphEigen cost 41614 ms
Time: graphFilter cost 1388 ms
Time: computeScore cost 55037 ms
graphfilter.computeScore cost:  55.42829728126526 seconds
Time taken to sample:  56.352569818496704 seconds
```

# 3、Tips
* 找到conda环境下的Python头文件
```
python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
```

* 查找NumPy的包含路径
```
python -c "import numpy; print(numpy.get_include())"
```

* 编译V1版本：
```
g++ -shared -o filtergraphv1.so pyGraphFilter.cpp graphFilter.cpp pccProcessing.cpp -I./ -I/usr/local/include/eigen3/ -I/DATA_EDS2/zhuzx/miniconda3/envs/gaussian_splatting/include/python3.7m/  -I/DATA_EDS2/zhuzx/miniconda3/envs/gaussian_splatting/lib/python3.7/site-packages/numpy/core/include/  -O3 -fopenmp -fPIC -Wall
```

> V1版本测试cubic.ply无问题，测试point_cloud_500w存在C++中打印的点云信息不对，应该是数据处理部分还是有问题

正确的输出：
```x1: 2.07133 y1: 2.02422 z1: -1.16529
x2: 1.48653 y2: 1.41477 z2: 1.65127
x3: 1.43754 y3: 1.37962 z3: 1.16425
```
当前V1的输出：
```
x1: 0.0435803 y1: 0.0458223 z1: 0.0904121
x2: 0.0496849 y2: 0.662647 z2: 0.63962
x3: 0.532232 y3: 0.37154 z3: 0.397041
x4: 0.163577 y4: 0.193415 z4: 0.293161
```