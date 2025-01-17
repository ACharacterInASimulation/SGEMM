#pragma once


#define Nr 16
#define Mr 6

typedef void (*matmul_func_t)(float *A, float *B, float *C, int M, int N , int K);
void naive(float* A, float* B, float* C, const int M, const int N, const int K);
void cblas(float* A, float* B, float* C, const int M, const int N, const int K);
void reorder(float* A, float* B, float* C, const int M, const int N, const int K);
void avx(float* A, float* B, float* C, const int M, const int N, const int K);
