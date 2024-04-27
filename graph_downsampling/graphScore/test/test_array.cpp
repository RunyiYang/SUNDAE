#include <cstdio>
extern "C"
void show_matrix(double *matrix, int rows, int columns)
{
    int i, j;
    for (i=0; i<rows; i++) {
        for (j=0; j<columns; j++) {
            printf("matrix[%d][%d] = %f\n", i, j, matrix[i*columns + j]);
        }
    }
}