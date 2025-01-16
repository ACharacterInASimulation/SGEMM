#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>


extern void matmul_naive(float* A, float* B, float* C, const int M, const int N, const int K);
extern void matmul_cblas(float* A, float* B, float* C, const int M, const int N, const int K);


void benchmark(const char* name, 
               void (*matmul_func)(float*, float*, float*, const int, const int, const int), 
               float* A, float* B, float* C, const int M, const int N, const int K){

    float total_flops = 2.0*M*N*K;

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    matmul_func(A, B, C, M, N, K);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("%s : %f GFLOPS\n", name, (total_flops / elapsed_time) / 1e9);
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
    if (strcmp(program, "naive") != 0 && strcmp(program, "cblas") != 0) {
        printf("Error: Invalid program option. Use 'naive' or 'cblas'.\n");
        return 1;
    }

    
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices A and B with random values
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (float)rand() / RAND_MAX;
    }

    if (strcmp(program, "naive") == 0) {
        benchmark("Naive", matmul_naive, A, B, C, M, N, K);
    } else if (strcmp(program, "cblas") == 0){
        benchmark("CBLAS", matmul_cblas, A, B, C, M, N, K);
    }

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}