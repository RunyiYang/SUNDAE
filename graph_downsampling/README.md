# Compilation Instructions for Shared Object File

## 1. Compilation for .so File

Navigate to the graphScore directory and use the following command to compile the shared object file:

```bash
cd graphScore
g++ -shared -o filtergraphv2.so pyGraphFilterV2.cpp graphFilter.cpp pccProcessing.cpp -I./ -I/usr/local/include/eigen3/ -O3 -fopenmp -fPIC -Wall
```

## 2. Running the File

Execute the script using Python with a specified PLY file:

```bash
python test_ctypes.py ply/point_cloud.ply
```

### Runtime for 5 Million Points:

```plaintext
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

## 3. Adding to the Conda Environment

### Find Python Include Path:

Execute the following command to find the Python include path:

```bash
python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
```

### Find NumPy Include Path:

To find the include path for NumPy, use:

```bash
python -c "import numpy; print(numpy.get_include())"
```

### Compiling Version 1:

Compile the V1 version with the command below, ensuring the include paths are correctly specified for your Conda environment:

```bash
g++ -shared -o filtergraphv1.so pyGraphFilter.cpp graphFilter.cpp pccProcessing.cpp -I./ -I/usr/local/include/eigen3/ -I/path/to/conda/env/include/python3.7m/ -I/path/to/conda/env/lib/python3.7/site-packages/numpy/core/include/ -O3 -fopenmp -fPIC -Wall
```

> The V1 version works without issues for `cubic.ply` but shows incorrect point cloud information for `point_cloud_500w.ply`, suggesting a problem in the data processing.

**Correct Output:**

```plaintext
x1: 2.07133 y1: 2.02422 z1: -1.16529
x2: 1.48653 y2: 1.41477 z2: 1.65127
x3: 1.43754 y3: 1.37962 z3: 1.16425
```

**Current Output for V1:**

```plaintext
x1: 0.0435803 y1: 0.0458223 z1: 0.0904121
x2: 0.0496849 y2: 0.662647 z2: 0.63962
x3: 0.532232 y3: 0.37154 z3: 0.397041
x4: 0.163577 y4: 0.193415 z4: 0.293161
```

## 3. Refer to Fast Resampling of 3D Point Clouds via Graphs
A python implement to https://arxiv.org/abs/1702.06397.