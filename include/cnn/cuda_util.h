//
// Created by tonye on 2019-06-13.
//

#ifdef CUDA_UTIL_H
#define CUDA_UTIL_H

extern int gpu_index;

#ifdef GPU

#define BLOCK 512

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

void check_error(cudaError_t status);

dim3 cuda_gridsize(size_t n);

#endif
#endif //CUDA_UTIL_H
