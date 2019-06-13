#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C"{
#include "activations.h"
#include "cuda_util.h"
}

//在cuda中在设备（device）中声明一个全局变量用__device__关键字修饰
__device__ float linear_activate_kernel(float x){
    return x;
}

__device__ float relu_activate_kernel(float x){
    return x*(x>0);
}


__device__ float linear_gradient_kernel(float x){
    return 1;
}

__device__ float relu_gradient_kernel(float x){
    return (x>0);
}

__device__ float activate_kernel(float x, ACTIVATION a){
    switch(a){
        case LINEAR:
            return linear_activate_kernel(x);
        case RELU:
            return relu_activate_kernel(x);
    }
    return 0;
}

__device__ float gradient_kernel(float x, ACTIVATION a){
    switch(a){
        case LINEAR:
            return linear_gradient_kernel(x);
        case RELU:
            return relu_gradient_kernel(x);
    }
    return 0;
}

/**
 * __global__函数必须由CPU调用，并将并行计算任务发射到GPU的任务调用单元
 * cuda中threadIdx、blockIdx、blockDim和gridDim的使用
 *
 * blockIdx是一个uint3类型，表示一个线程块的索引，一个线程块中通常有多个线程。
 * gridDim是一个dim3类型，表示网格的大小，一个网格中通常有多个线程块。
 * blockDim是一个dim3类型，表示线程块的大小。
 * threadIdx是一个uint3类型，表示一个线程的索引。
 *
 * BLOCK 默认为512
 */
__global__ void activate_array_kernel(float *x, int n, ACTIVATION a){

    //获取线程ID
    //线程索引的计算公式: grid划分成2维，block划分为1维
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * blockDim.x + threadIdx.x;

    int i = threadId;

    if(i < n) {
        x[i] = activate_kernel(x[i], a);
    }

}

__global__ void gradient_array_kernel(float *x, int n, ACTIVATION a){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * blockDim.x + threadIdx.x;

    int i = threadId;
    if(i < n){
        x[i] = gradient_kernel(x[i], a);
    }
}



//extern "C" void activate_array_ongpu(float *x, int n, ACTIVATION a){
//    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a);
//    check_error(cudaPeekAtLastError());
//}
//
//extern "C" void gradient_array_ongpu(float *x, int n, ACTIVATION a){
//    gradient_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a);
//    check_error(cudaPeekAtLastError());
//}