#include <stdio.h>
#include <immintrin.h>


void kernel_6_16(float* A, float* B, float* C, int M, int N, int K){
    __m256 a_ijbroadcast;
    __m256 b0_packFloat8;
    __m256 b1_packFloat8;
    __m256 c_buffer[12];

    //load C to C_buffer
    for(int i = 0; i < 6; i++){
        c_buffer[2 * i] = _mm256_loadu_ps(&C[i*N]);
        c_buffer[2 * i + 1] = _mm256_loadu_ps(&C[i*N + 8]);
    }

    for(int row = 0; row < 6; row++){
        for(int inner = 0; inner < K; inner++){
            b0_packFloat8 = _mm256_loadu_ps(&B[inner * N]);
            b1_packFloat8 = _mm256_loadu_ps(&B[inner * N + 8]);
            a_ijbroadcast =  _mm256_broadcast_ss(&A[row * K + inner]);
            c_buffer[2 * row] = _mm256_fmadd_ps(a_ijbroadcast, b0_packFloat8, c_buffer[2 * row]);
            c_buffer[2*row + 1] = _mm256_fmadd_ps(a_ijbroadcast, b1_packFloat8, c_buffer[2 * row + 1]);
        }
    }

    //store c-buffer to c
    for(int i = 0; i < 6; i++){
        _mm256_storeu_ps(&C[i*N], c_buffer[2 * i]);
        _mm256_storeu_ps(&C[i*N + 8], c_buffer[2 * i + 1]);
    }
}