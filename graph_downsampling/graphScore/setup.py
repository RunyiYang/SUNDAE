from setuptools import setup, Extension
import numpy as np

graphfilter_module = Extension(
    'graphfilter',
    sources=['pyGraphFilter.cpp', 'graphFilter.cpp', 'pccProcessing.cpp'],
    include_dirs=['.', '/usr/local/include/eigen3/', np.get_include()],
    language='c++',
    extra_compile_args=['-O3', '-fopenmp']
)

setup(
    name='GraphFilter',
    version='1.0',
    description='Python wrapper for GraphFilter C++ code',
    ext_modules=[graphfilter_module],
    debug=False
)
