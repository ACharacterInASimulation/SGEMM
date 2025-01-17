#include <stddef.h>
#include <cblas.h>
#include "kernels.h"
#include <assert.h>

#define Nr 16
#define Mr 6


void naive(float* A, float*B, float*C, const int M, const int N, const int K){
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            for(int p =0; p < K; p++){
                C[i * N + j] += A[i * K + p] * B[p * N + j];
            }
        }
    }
}

void cblas(float* A, float* B, float* C, const int M, const int N, const int K) {
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


void reorder(float* A, float*B, float*C, const int M, const int N, const int K){
    for (int i = 0; i < M; i++){
        for (int p = 0; p < K; p++){
            for(int j =0; j < N; j++){
                C[i * N + j] += A[i * K + p] * B[p * N + j];
            }
        }
    }
}


void avx(float* A, float*B, float*C, const int M, const int N, const int K){
    for (int i = 0; i < M; i += Mr){
        for(int j = 0; j < N; j += Nr){
            kernel_6_16(&A[i * K], &B[j], &C[i * N + j], M, N, K);
        }
    }
}







