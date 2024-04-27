#include "graphFilter.hpp"
#include "pccProcessing.hpp"
#include <vector>
#include <chrono>

using namespace pcc_processing;
using namespace graphFiltering;

extern "C" double* ctypes_computeScore(double* pCoords, int size)
{
    auto start = std::chrono::system_clock::now();
    // double* pCoords = numpyArrayToDoubleArray(py_cloud);
    // long int size = PyArray_DIM(py_cloud, 0);
    // Not XYZ format
    // std::cout << "x1: " << pCoords[0] << " y1: " << pCoords[1] << " z1: " << pCoords[2] << std::endl;
    // std::cout << "x2: " << pCoords[3] << " y2: " << pCoords[4] << " z2: " << pCoords[5] << std::endl;
    // std::cout << "x3: " << pCoords[6] << " y3: " << pCoords[7] << " z3: " << pCoords[8] << std::endl;
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

    // delete[] pCoords;

    return pScore;
}

extern "C" void freepScores(double *pScore){
    delete[] pScore;
}

