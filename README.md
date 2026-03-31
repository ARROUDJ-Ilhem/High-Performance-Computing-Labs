# High-Performance-Computing-Labs - Parallel Programming Paradigms
 
> A hands-on comparison of **sequential**, **Pthreads**, **OpenMP**, **CUDA** and **MPI**  implementations across five classic HPC problems, written in C.
 
This repository was built as part of a High Performance Computing course (2024–2025). Five different problems are solved once as a sequential baseline and once per parallelism paradigm (Cuda, OpenMP, and/or Pthreads), so that execution times and speedups can be measured and compared directly.
 
---
## Problems
 
### 1 — Riemann Sum
Numerical integration of ∫₀¹ e^(−x²) dx using N = 10⁹ sub-intervals (left Riemann sum). Each sub-interval contributes `f(xᵢ) × Δx` to the total. The work is embarrassingly parallel — the interval is split evenly across threads/blocks with no dependencies between parts.
 
### 2 — 2D Convolution
Application of a 3×3 Gaussian blur kernel to a small RGB image (20×22 pixels). Each output pixel is computed independently as a weighted sum of its neighborhood, making this a natural fit for 2D thread grids in CUDA and loop parallelism in OpenMP/Pthreads. Border pixels use clamp (edge replication) padding.
 
### 3 — Nearest Neighbor Search
Finding the closest point to a fixed target `(500, 500)` among 10 million randomly generated 2D points. Each thread scans its own partition of the point array for a local minimum; the global minimum is found by reducing the per-thread results.
 
### 4 — Gaussian Elimination (OpenMP only)
Row-echelon reduction of a random N×N matrix using Gauss elimination without pivoting. This problem has loop-carried dependencies (outer loop `i` must complete before `i+1` starts), so only the inner loop over rows `j > i` can be parallelized. Three OpenMP strategies are explored to understand where and how to place `#pragma omp parallel` and `#pragma omp for`.
 
### 5 — Array Squaring (MPI only)
Distributed squaring of an integer array (N = 16 elements) using the **master-worker pattern** across multiple MPI processes. The master process splits the array into fixed-size chunks (size 4) and dispatches them dynamically to worker processes. Each worker squares the elements of its chunk and sends the result back. This problem demonstrates point-to-point communication, dynamic task scheduling, and process coordination in a distributed-memory model.
 
---

## Parallelism Concepts by Paradigm
 
| Paradigm | Memory model | Unit of parallelism | Synchronization |
|----------|-------------|---------------------|-----------------|
| **Pthreads** | Shared (CPU) | POSIX thread | `pthread_join`, `pthread_mutex_t` |
| **OpenMP** | Shared (CPU) | OpenMP thread | `#pragma omp parallel`, `#pragma omp for` |
| **CUDA** | Separate (GPU ↔ CPU) | CUDA thread (warp → block → grid) | `cudaDeviceSynchronize`, `__syncthreads()` |
| **MPI** | Distributed (multi-process) | MPI process | `MPI_Send`, `MPI_Recv`, `MPI_Finalize` |
 
---
 
## OpenMP — Gaussian Elimination: 2 Versions Explained
 
The Gauss problem has a strict **outer loop dependency**: iteration `i` must finish before `i+1` begins (each pivot row depends on the previous one). Only the **inner loop over rows `j`** can be safely parallelized.
 
| Version | `#pragma omp parallel` | `#pragma omp for` | Notes |
|---------|------------------------|-------------------|-------|
| **V1** | Inside `gaussian()` | Inside `gaussian()` | Self-contained parallel region per pivot step |
| **V2** | Inside `main()` | Inside `main()` | Both directives together in main, gaussian() is a pure computation function |
 
All versions produce the same correct result. The differences are structural — they demonstrate how OpenMP's fork-join model can be applied at different levels of the call stack.
 
---
## Key CUDA Patterns Used
 
| File | CUDA technique |
|------|----------------|
| `reimann_cuda.c` | Grid-stride loop — one kernel covers 10⁹ iterations regardless of grid size |
| `conv_cuda.c` | 2D thread grid — one thread per output pixel; kernel stored in `__constant__` memory |
| `plus_proche_voisin_cuda.c` | Per-block reduction with `__shared__` memory — each block finds its local minimum, CPU reduces the block results |
 
---
 
## MPI — Master-Worker Pattern Explained
 
The MPI implementation follows the classic **master-worker** (or bag-of-tasks) model:
 
- **Process 0 (master)**: holds the full array, splits it into chunks, and dispatches them one at a time to whichever worker asks. Once all chunks are sent, it sends a termination signal (tag = 1) to each idle worker, then collects all processed results.
- **Processes 1…N-1 (workers)**: repeatedly request work from the master, process the received chunk (squaring each element), and send the result back until they receive the stop signal.
 
This design naturally load-balances across any number of processes and requires at least 2 MPI processes to run.
 
| Concept | How it's used |
|---|---|
| Point-to-point communication | `MPI_Send` / `MPI_Recv` for chunk dispatch and result collection |
| Dynamic scheduling | Master assigns the next available chunk to whichever worker finishes first |
| Tag-based signaling | Tag `0` = work available; Tag `1` = no more work (stop signal) |
| `MPI_ANY_SOURCE` | Master accepts requests from any worker without a fixed order |
 
---
## Compilation
 
### Sequential
 
```bash
gcc -O2 -o reimann_sq       Pthreads/Riemann sum/reimann_sq.c               -lm
gcc -O2 -o reimann_sq       Cuda/Riemann sum/reimann_sq.c               -lm
gcc -O2 -o conv_sq           Pthreads/2D convolution/conv_sq.c
gcc -O2 -o conv_sq           Cuda/2D convolution/conv_sq.c
gcc -O2 -o ppv_sq            Pthreads/nearest neighbor/plus_proche_voisin_sq.c   -lm
gcc -O2 -o ppv_sq            Cuda/nearest neighbor/plus_proche_voisin_sq.c   -lm
gcc -O2 -o gauss_sq          OpenMP/gauss_sq.c                -lm
```
 
### Pthreads
 
```bash
gcc -O2 -o reimann_p         pthreads/Riemann sum/reimann_p.c                 -lpthread -lm
gcc -O2 -o conv_p             pthreads/2D convolution/conv_p.c                   -lpthread
gcc -O2 -o ppv_p              pthreads/nearest neighbor/plus_proche_voisin_p.c     -lpthread -lm
```
 
### OpenMP
 
```bash
gcc -O2 -fopenmp -o gauss_v1      OpenMP/gauss_openmp_v1.c           -lm
gcc -O2 -fopenmp -o gauss_v2      OpenMP/gauss_openmp_v2.c           -lm

```
 
### CUDA (requires NVIDIA GPU + CUDA Toolkit ≥ 11)
 
```bash
nvcc -O2 -o reimann_cuda    Cuda/Riemann sum/reimann_cuda.c                   -lm
nvcc -O2 -o conv_cuda        Cuda/2D convolution/conv_cuda.c
nvcc -O2 -o ppv_cuda         Cuda/nearest neighbor/plus_proche_voisin_cuda.c       -lm
```

---


### MPI (requires MPI implementation, e.g. OpenMPI or MPICH)
```bash
mpicc -O2 -o mpi_worker      MPI/array_sq_mpi.c                          -lm
```
To run with P processes:
```bash
mpirun -np <P> ./mpi_worker   # P must be >= 2
```

---
 
## Requirements
 
- **GCC ≥ 9** with OpenMP support (flag: `-fopenmp`)
- **OpenMPI ≥ 4** or **MPICH ≥ 3** (for MPI files, compile with `mpicc`)
- **CUDA Toolkit ≥ 11** + compatible NVIDIA driver (for CUDA files)
- `make` (optional, for build automation)
 
---
