#pragma once

typedef void (*kernel_func_t)(float *A, float *B, float *C, int M, int N , int K);

void kernel_6_16(float* A, float* B, float* C, int M, int N, int K);