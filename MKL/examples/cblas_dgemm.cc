/*
编译方式：
[参考](https://www.cnblogs.com/GeophysicsWorker/p/16175589.html)

g++ cblas_dgemm.cc -I /weishengying/download/intel/mkl/latest/include/ \
-L /weishengying/download/intel/mkl/latest/lib/intel64/ -lmkl_rt -liomp5

一些定义：
row-major: 在同一行的元素在内存中是相邻的；
column-major: 同一列的元素在内存中是相邻的。
*/

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

#define min(x,y) (((x) < (y)) ? (x) : (y))

int main()
{
    double* A, * B, * C;
    int m, n, k, i, j;
    double alpha, beta;


    m = 2000, k = 200, n = 1000;

    alpha = 1.0; beta = 0.0;

    A = (double*)mkl_malloc(m * k * sizeof(double), 64);
    B = (double*)mkl_malloc(k * n * sizeof(double), 64);
    C = (double*)mkl_malloc(m * n * sizeof(double), 64);
    if (A == NULL || B == NULL || C == NULL) {

        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return 1;
    }


    for (i = 0; i < (m * k); i++) {
        A[i] = (double)(i + 1);
    }

    for (i = 0; i < (k * n); i++) {
        B[i] = (double)(-i - 1);
    }

    for (i = 0; i < (m * n); i++) {
        C[i] = 0.0;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k, alpha, A, k, B, n, beta, C, n);


    for (i = 0; i < min(m, 6); i++) {
        for (j = 0; j < min(k, 6); j++) {
            printf("%12.0f", A[j + i * k]);
        }
        printf("\n");
    }

    for (i = 0; i < min(k, 6); i++) {
        for (j = 0; j < min(n, 6); j++) {
            printf("%12.0f", B[j + i * n]);
        }
        printf("\n");
    }

    for (i = 0; i < min(m, 6); i++) {
        for (j = 0; j < min(n, 6); j++) {
            printf("%12.5G", C[j + i * n]);
        }
        printf("\n");
    }

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}
