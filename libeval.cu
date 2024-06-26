#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>


__global__ void _evaluate(
        bool *adjs,
        uint16_t *perms,
        uint16_t *degrees,
        int *fitnesses,
        size_t B,
        size_t N
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B) {
        int penalty = 0;
        int vertex;
        uint16_t degree;
        uint16_t nz[3000];
        int adj_offset = idx * N * N;
        for (int step = 0; step < N; step++) {
            vertex = perms[idx * N + step];

            // Find adjacent nodes
            degree = 0;
            for (uint16_t i = 0; i < N; i++) {
                if (adjs[adj_offset + (vertex * N) + i] == 1) {
                    nz[degree] = i;
                    degree ++;
                }
            }

            // Penalize 
            if (degree > 500) {
                if (penalty == 0) {
                    penalty = 501;
                } else {
                    penalty += degree - 500;
                }
            }

            // Update degrees
            degrees[(idx * N) + step] = degree;

            // Early stopping
            if (penalty > 1000) {
                penalty += 1000;
                // printf("[%d] [%d] Breaking early!\n", idx, step);
                break;
            }

            // Add edges to the right
            for (uint16_t i = 0; i < degree; i ++) {
                int ii = nz[i];
                for (uint16_t j = i; j < degree; j ++) {
                    int jj = nz[j];
                    adjs[adj_offset + (ii * N) + jj] = 1;
                    adjs[adj_offset + (jj * N) + ii] = 1;
                }
            }
            // Remove edges to the left
            for (uint16_t i = 0; i < degree; i++) {
                int ii = nz[i];
                adjs[adj_offset + (ii * N) + vertex] = 0;
                adjs[adj_offset + (vertex * N) + ii] = 0;
            }
        }

        // Apply penalty
        if (penalty > 0) {
            for (int i = 0; i < N; i++) {
                degrees[(idx * N) + i] += penalty;
            }
        }

        // Calculate final scores
        for (int i = 0; i < N; i++) {
            int max_degree = 0;
            for (int j = i; j < N; j++) {
                uint16_t x = degrees[(idx * N) + j];
                if (x > max_degree) {
                    max_degree = x;
                }
            }
            fitnesses[(idx * N) + i] = max_degree;
        }
    }
}

extern "C" {
    void evaluate(
            bool *adjs,
            uint16_t *perms,
            uint16_t *degrees,
            int *fitnesses,
            size_t B,
            size_t N
    ) {
        _evaluate<<<128, B / 128>>>(adjs, perms, degrees, fitnesses, B, N);
        cudaDeviceSynchronize();
    }
}
