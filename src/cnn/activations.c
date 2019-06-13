//
// Created by tonye on 2019-06-13.
//

#include "activations.h"
#include <stdio.h>
#include <string.h>

ACTIVATION get_activation(char *s){
    if(strcmp(s, "linear")==0) return LINEAR;
    if(strcmp(s, "relu")==0) return RELU;
    fprintf(stderr, "Couldn't find activation funcation %s, going with ReLU\n", s);
    return RELU;

}

char *get_activation_string(ACTIVATION a){
    switch(a){
        case LINEAR:
            return "linear";
        case RELU:
            return "relu";
        default:
            break;
    }
    return "relu";
}

float activate(float x, ACTIVATION a){
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case RELU:
            return relu_activate(x);
    }
}

float gardient(float x, ACTIVATION a){
    switch(a){
        case LINEAR:
            return linear_gardient(x);
        case RELU:
            return relu_gardient(x);

    }
    return 0;
}

// 通过激活函数输出各个特征
void activate_array(float *x, const int n, const ACTIVATION a){
    for(int i = 0;i<n;i++){
        x[i] = activate(x[i], a);
    }

}

void gardient_arrray(float *x, const int n, const ACTIVATION a, float *delta){
    for(int i=0;i<n;i++){
        delta[i] *= gardient(x[i], a);
    }
}