//
// Created by tonye on 2019-06-13.
//

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cuda.h>
#include <math.h>

typedef  enum{
    LINEAR, RELU
}ACTIVATION;

ACTIVATION get_activation(char *s);
char *get_activation_string(ACTIVATION a);
float activate(float x, ACTIVATION a);
float gardient(float x, ACTIVATION a);

//正向传播中如何对一个输入数据求解其经过激活函数的值
void activate_array(float *x, const int n, const ACTIVATION a);

//反向传播中如何回传值进行梯度计算
void gardient_array(const float x, const int n, ACTIVATION a, float *delta);

/**
 * 内联函数可以加快调用的速度,
 * static 修饰的内联函数inline，一般情况下不会产生函数本身的代码，而是全部被嵌入在被调用的地方
 */
static inline float
linear_activate(float x){
    return x;
}

static inline float
relu_activate(float x){
    return x*(x>0);
}


static inline float
linear_gardient(float x){
    return 1;
}
static inline float
relu_gardient(float x){
    return (x>0);
}


#endif //ACTIVATIONS_H
