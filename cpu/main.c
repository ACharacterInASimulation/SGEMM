#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "matmul.h"


void benchmark(const char* name, 
               matmul_func_t matmul_func, 
               float* A, float* B, float* C, const int M, const int N, const int K){

    double total_flops = 2 * (double)M * N * K;

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    matmul_func(A, B, C, M, N, K);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("%s : %.4f GFLOPS\n", name, (total_flops / elapsed_time) / 1e9);
}


void is_accurate(float* C, float* G, const int M, const int N){
    for(int i=0; i<M*N; i++){
        if(abs(C[i] - G[i]) > 1e-3){
            printf("Mismatch at %d: %.3f , %.3f\n",i, C[i], G[i]);
            break;
        }
    }

}


int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("Usage: %s <M> <N> <K> <naive|cblas>\n", argv[0]);
        return 1;
    }

    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);
    const char* program = argv[4];

    if (M <= 0 || N <= 0 || K <= 0) {
        printf("Error: M, N, and K must be positive integers.\n");
        return 1;
    }
    if (strcmp(program, "naive") != 0 && strcmp(program, "cblas") != 0 && strcmp(program, "reorder") != 0 &&  strcmp(program, "avx") != 0) {
        printf("Error: Invalid program option. Use 'naive' or 'cblas'.\n");
        return 1;
    }

    //Allocate mmeory for A,B,C,G
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C = (float*)malloc(M * N * sizeof(float));
    float* G = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices A and B with random values and C,G with zeros 
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0;
        G[i] = 0.0;
    }


    //To check the program accuracy with cblas
    cblas(A, B, G, M, N, K);


    if (strcmp(program, "naive") == 0) {
        benchmark("Naive", naive, A, B, C, M, N, K); 
    } else if (strcmp(program, "cblas") == 0){
        benchmark("CBLAS", cblas, A, B, C, M, N, K);
    }else if (strcmp(program, "reorder") == 0){
        benchmark("Reorder_loop", reorder, A, B, C, M, N, K);
    }else if (strcmp(program, "avx") == 0){
        benchmark("AVX", avx, A, B, C, M, N, K);
    }

    is_accurate(C,G,M,N);

    // Free allocated memory
    free(A);
    free(B);
    free(C);
    free(G);

    return 0;
}