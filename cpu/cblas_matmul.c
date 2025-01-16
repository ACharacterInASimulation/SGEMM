#include <cblas.h>

void matmul_cblas(float* A, float* B, float* C, const int M, const int N, const int K) {
    const float alpha = 1.0;
    const float beta = 0.0;

    // Call CBLAS SGEMM for matrix multiplication
    cblas_sgemm(CblasRowMajor,      //Row major order matrix storage
                CblasNoTrans,       //Matrix A is not transposed
                CblasNoTrans,       //Matrix B is not ttransposed
                M,                  // #Rows in A and C (M)
                N,                  // #Cols in B and C (N)  
                K,                  // #Cols and #Rows in A,B (K)
                alpha, 
                A, K,               //Matrix A and its leading dimension (K)
                B, N,               //Matrix B and its leading dimension (N)
                beta, 
                C, N);              //Matrix C and its leading dimension (N)
}
