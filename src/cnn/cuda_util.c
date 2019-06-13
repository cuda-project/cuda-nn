//
// Created by tonye on 2019-06-13.
//

int gpu_index = 0;

#ifdef GPU

#include "cuda_util.h"

void check_error(cudaError_t status)
{
    cudaError_t status2 = cudaGetLastError();// Returns the last error that has been produced by any of the runtime calls in the same host thread and resets it to cudaSuccess.
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);// Returns the description string for an error code.
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    }
}

// 从总的线程的个数n compute grid的个数
dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

#endif
