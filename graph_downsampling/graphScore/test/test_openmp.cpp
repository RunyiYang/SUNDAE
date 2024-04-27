// g++ -fopenmp test_openmp.cpp -o test_openmp
#include <iostream>
int main()
{
  #pragma omp parallel
  {
    std::cout << "Hello World!\n";  //我的CPU为四核，所以生成了4个线程
  }
}
