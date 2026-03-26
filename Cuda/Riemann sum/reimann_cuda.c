/*
 * reimann_cuda.c
 * Approximation de l'intégrale de exp(-x^2) sur [0,1]
 * par la somme de Riemann, parallélisée sur GPU (CUDA).
 *
 * Compilation : nvcc -O2 -o reimann_cuda reimann_cuda.c -lm
 */

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 1000000000LL

/* ------------------------------------------------------------------ */
/*  Kernel GPU : grid-stride loop, chaque thread traite N/total cases  */
/* ------------------------------------------------------------------ */
__global__ void riemann_kernel(double a, double b, long long n, double *partial_sums)
{
    long long idx   = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)gridDim.x  * blockDim.x;

    double delta_x   = (b - a) / (double)n;
    double local_sum = 0.0;

    for (long long i = idx; i < n; i += total) {
        double x_i = a + i * delta_x;
        local_sum += exp(-x_i * x_i) * delta_x;
    }

    partial_sums[idx] = local_sum;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */
int main(void)
{
    const double a = 0.0, b = 1.0;

    const int BLOCKS  = 256;
    const int THREADS = 256;
    const int TOTAL   = BLOCKS * THREADS;

    /* Allocation mémoire GPU */
    double *d_partial;
    cudaMalloc((void **)&d_partial, TOTAL * sizeof(double));

    /* Chronométrage avec événements CUDA */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    riemann_kernel<<<BLOCKS, THREADS>>>(a, b, N, d_partial);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    /* Copie résultat vers CPU */
    double *h_partial = (double *)malloc(TOTAL * sizeof(double));
    cudaMemcpy(h_partial, d_partial, TOTAL * sizeof(double), cudaMemcpyDeviceToHost);

    /* Réduction finale sur CPU */
    double result = 0.0;
    for (int i = 0; i < TOTAL; i++)
        result += h_partial[i];

    printf("Temps d'exécution GPU : %.4f ms\n", ms);
    printf("Somme de Riemann (intégrale approx.) : %.10f\n", result);

    /* Nettoyage */
    free(h_partial);
    cudaFree(d_partial);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
