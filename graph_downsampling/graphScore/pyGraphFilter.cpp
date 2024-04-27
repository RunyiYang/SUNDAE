#include <Python.h>
#include <numpy/arrayobject.h>
#include "graphFilter.hpp"
#include "pccProcessing.hpp"
#include <vector>
#include <chrono>

using namespace pcc_processing;
using namespace graphFiltering;

// Structure to represent a 3D point (assuming the point cloud is in 3D)
struct Point {
    double x, y, z;
};

// convert a 2D NumPy array to a double array
double* numpyArrayToDoubleArray(PyArrayObject* npArray) {
    long int numPoints = PyArray_DIM(npArray, 0);
    double* points = new double[numPoints * 3]; // [X(N)...Y(N)...Z(N)]

    for (int i = 0; i < numPoints; i++) {
        for (int j = 0; j < 3; j++)
            points[i + j*numPoints] = *(double*)PyArray_GETPTR2(npArray, i, j);
    }

    return points;
}

// 将numpy数组转换为C++中的double数组
double* numpyArrayToDoubleArrayV2(PyObject* numpyArray) {
    PyArrayObject* npArray = reinterpret_cast<PyArrayObject*>(numpyArray);
    npy_intp size = PyArray_SIZE(npArray);
    double* doubleArray = new double[size];

    if (PyArray_TYPE(npArray) == NPY_DOUBLE) {
        memcpy(doubleArray, PyArray_DATA(npArray), size * sizeof(double));
    } else {
        PyArrayObject* npArrayDouble = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(numpyArray, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
        memcpy(doubleArray, PyArray_DATA(npArrayDouble), size * sizeof(double));
        Py_DECREF(npArrayDouble);
    }

    return doubleArray;
}

// Wrapper function for computeScore
static PyObject* py_computeScore(PyObject* self, PyObject* args) {
    PyArrayObject* py_cloud;
    // double pScore;
    // Expecting a 2D numpy array as input
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &py_cloud)) {
        return NULL;
    }

    auto start = std::chrono::system_clock::now();
    double* pCoords = numpyArrayToDoubleArray(py_cloud);

    std::cout << "x1: " << pCoords[0] << " y1: " << pCoords[1] << " z1: " << pCoords[2] << std::endl;
    std::cout << "x2: " << pCoords[3] << " y2: " << pCoords[4] << " z2: " << pCoords[5] << std::endl;
    std::cout << "x3: " << pCoords[6] << " y3: " << pCoords[7] << " z3: " << pCoords[8] << std::endl;
    
    long int size = PyArray_DIM(py_cloud, 0);
    
    PccPointCloud inCloud;
    inCloud.loadBlock(pCoords, size);

    std::cout << "Point Cloud size in C++: " << inCloud.size << std::endl;

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time: numpyArrayToDoubleArray and loadBlock cost " <<  duration << " ms" << std::endl;

    // TODO: Convert std::vector<Point> to the appropriate data type for computeScore function
    // and call the function
    start = std::chrono::system_clock::now();
    double *pScore =  new double[size];
    computeScore(inCloud, pScore);
    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time: computeScore cost " <<  duration << " ms" << std::endl;

    PyObject* pyArray = PyList_New(size);

    // 将数组的元素添加到Python列表中
    for (int i = 0; i < size; i++) {
        PyObject* item = Py_BuildValue("d", pScore[i]);
        PyList_SetItem(pyArray, i, item);
    }

    delete[] pScore;
    delete[] pCoords;

    return pyArray;

    // For now, we're returning a dummy score as we still need to match the data types
    // pScore = 42.0;

    // return Py_BuildValue("d", pScore);
}

extern "C" {
    double* ctypes_computeScore(PyArrayObject* py_cloud) {

    auto start = std::chrono::system_clock::now();
    double* pCoords = numpyArrayToDoubleArray(py_cloud);
    long int size = PyArray_DIM(py_cloud, 0);
    
    std::cout << "x1: " << pCoords[0] << " y1: " << pCoords[1] << " z1: " << pCoords[2] << std::endl;
    std::cout << "x2: " << pCoords[3] << " y2: " << pCoords[4] << " z2: " << pCoords[5] << std::endl;
    std::cout << "x3: " << pCoords[6] << " y3: " << pCoords[7] << " z3: " << pCoords[8] << std::endl;
    std::cout << "x4: " << pCoords[9] << " y4: " << pCoords[10] << " z4: " << pCoords[11] << std::endl;
    
    PccPointCloud inCloud;
    inCloud.loadBlock(pCoords, size);
    std::cout << "Point Cloud size in C++: " << inCloud.size << std::endl;

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time: numpyArrayToDoubleArray and loadBlock cost " <<  duration << " ms" << std::endl;

    // TODO: Convert std::vector<Point> to the appropriate data type for computeScore function
    // and call the function
    start = std::chrono::system_clock::now();
    double *pScore =  new double[size];
    computeScore(inCloud, pScore);
    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time: computeScore cost " <<  duration << " ms" << std::endl;

    delete[] pCoords;

    return pScore;
}
}

static PyMethodDef graphfilter_methods[] = {
    {"computeScore", py_computeScore, METH_VARARGS, "Compute the score for graph filtering"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef graphfiltermodule = {
    PyModuleDef_HEAD_INIT,
    "graphfilter",
    NULL,
    -1,
    graphfilter_methods
};

PyMODINIT_FUNC PyInit_graphfilter(void) {
    import_array();
    return PyModule_Create(&graphfiltermodule);
}