import ctypes
import numpy as np
 
lib = ctypes.cdll.LoadLibrary("./test_array.so")
arr = np.array([[1.0,2,3],[4,5,6]])
#arr = np.array([[1,2],[3,4]])
tmp = np.asarray(arr)
rows, cols = tmp.shape
dataptr = tmp.ctypes.data_as(ctypes.c_char_p)
lib.show_matrix(dataptr, rows, cols)

'''
输出：
matrix[0][0] = 1.000000
matrix[0][1] = 2.000000
matrix[0][2] = 3.000000
matrix[1][0] = 4.000000
matrix[1][1] = 5.000000
matrix[1][2] = 6.000000

numpy的数组在内存中按行存储，
对于我们的Nx3的点云，所需要的格式为[x(N)+y(N)+z(N)]
因此，传入时需要进行转置
'''