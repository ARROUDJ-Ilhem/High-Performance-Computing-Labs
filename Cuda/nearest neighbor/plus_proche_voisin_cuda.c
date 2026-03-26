/*
 * plus_proche_voisin_cuda.c
 * Recherche du plus proche voisin parmi 10 millions de points 2D,
 * parallélisée sur GPU (CUDA) avec réduction par bloc (shared memory).
 *
 * Compilation : nvcc -O2 -o ppv_cuda plus_proche_voisin_cuda.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/*  Structure Point (float suffit pour 10M points)                     */
/* ------------------------------------------------------------------ */
typedef struct {
    float x, y;
} Point;

/* ------------------------------------------------------------------ */
/*  Kernel GPU : réduction par bloc avec mémoire partagée              */
/*                                                                      */
/*  Chaque bloc trouve son candidat local (distance min),              */
/*  écrit dans d_best_dist / d_best_idx.                               */
/* ------------------------------------------------------------------ */
__global__ void nn_kernel(const Point *points, int n,
                           float tx, float ty,
                           float *d_best_dist,
                           int   *d_best_idx)
{
    /* Mémoire partagée allouée dynamiquement à l'appel du kernel :
       première moitié = distances, deuxième moitié = indices (réinterprétés en float) */
    extern __shared__ float smem[];
    int *sidx = (int *)(smem + blockDim.x);

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;
    int step = gridDim.x  * blockDim.x;

    float local_dist = FLT_MAX;
    int   local_idx  = 0;

    /* Chaque thread parcourt sa tranche (grid-stride) */
    for (int i = gid; i < n; i += step) {
        float dx = points[i].x - tx;
        float dy = points[i].y - ty;
        float d  = dx * dx + dy * dy;   /* carré suffit pour comparer */
        if (d < local_dist) {
            local_dist = d;
            local_idx  = i;
        }
    }

    smem[tid] = local_dist;
    sidx[tid] = local_idx;
    __syncthreads();

    /* Réduction arborescente dans le bloc */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] < smem[tid]) {
            smem[tid] = smem[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }

    /* Le thread 0 de chaque bloc écrit le résultat du bloc */
    if (tid == 0) {
        d_best_dist[blockIdx.x] = smem[0];
        d_best_idx [blockIdx.x] = sidx[0];
    }
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */
int main(void)
{
    const int N = 10000000;

    /* Génération des points (même graine que la version séquentielle) */
    Point *h_points = (Point *)malloc(N * sizeof(Point));
    srand(42);
    for (int i = 0; i < N; i++) {
        h_points[i].x = (float)(rand() % 1000);
        h_points[i].y = (float)(rand() % 1000);
    }

    Point target = {500.f, 500.f};

    /* Transfert vers GPU */
    Point *d_points;
    cudaMalloc((void **)&d_points, N * sizeof(Point));
    cudaMemcpy(d_points, h_points, N * sizeof(Point), cudaMemcpyHostToDevice);

    /* Paramètres de lancement */
    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;
    const int SMEM  = 2 * BLOCK * sizeof(float);   /* dist (float) + idx (int, même taille) */

    float *d_best_dist;
    int   *d_best_idx;
    cudaMalloc((void **)&d_best_dist, GRID * sizeof(float));
    cudaMalloc((void **)&d_best_idx,  GRID * sizeof(int));

    /* Chronométrage GPU */
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);
    nn_kernel<<<GRID, BLOCK, SMEM>>>(d_points, N,
                                      target.x, target.y,
                                      d_best_dist, d_best_idx);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float gpu_ms = 0.f;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);

    /* Copie résultats partiels vers CPU */
    float *h_dist = (float *)malloc(GRID * sizeof(float));
    int   *h_idx  = (int   *)malloc(GRID * sizeof(int));
    cudaMemcpy(h_dist, d_best_dist, GRID * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx,  d_best_idx,  GRID * sizeof(int),   cudaMemcpyDeviceToHost);

    /* Réduction finale sur CPU */
    float best_dist = FLT_MAX;
    int   best_idx  = 0;
    for (int i = 0; i < GRID; i++) {
        if (h_dist[i] < best_dist) {
            best_dist = h_dist[i];
            best_idx  = h_idx[i];
        }
    }

    Point nearest = h_points[best_idx];
    printf("Plus proche voisin (CUDA) : (%.2f, %.2f)\n", nearest.x, nearest.y);
    printf("Distance réelle           : %.6f\n", sqrtf(best_dist));
    printf("Temps d'exécution GPU     : %.4f ms\n", gpu_ms);

    /* Nettoyage */
    free(h_points);
    free(h_dist);
    free(h_idx);
    cudaFree(d_points);
    cudaFree(d_best_dist);
    cudaFree(d_best_idx);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}
