int gpu_index = 0;

#ifdef GPU

#include "cuda_util.h"
#include "utils.h"
#include "blas.h"
#include "assert.h"
#include <stdlib.h>
#include <time.h>


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

cublasHandle_t blas_handle()
{
    static int init = 0;
    static cublasHandle_t handle;
    if(!init) {
        cublasCreate(&handle);
        init = 1;
    }
    return handle;
}

float *cuda_make_array(float *x, int n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;// n 个 float的大小
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);// 将数据从Host copy到 GPU device
        check_error(status);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_random(float *x_gpu, int n)
{
    static curandGenerator_t gen;
    static int init = 0;
    if(!init){
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, time(0));
        init = 1;
    }
    curandGenerateUniform(gen, x_gpu, n);
    check_error(cudaPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, int n, char *s)
{
    float *tmp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, tmp, n);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);// tmp= tmp-x
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err/n));
    free(tmp);
    return err;
}

int *cuda_make_int_array(int n)
{
    int *x_gpu;
    size_t size = sizeof(int)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    return x_gpu;
}

void cuda_free(float *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}
// 将x中的数据压入到 x_gpu
void cuda_push_array(float *x_gpu, float *x, int n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);// from X to x_gpu
    check_error(status);
}

// x_gpu当中的数据压入到x
void cuda_pull_array(float *x_gpu, float *x, int n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

#endif
