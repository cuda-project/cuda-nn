//
// Created by tonye on 2019-06-13.
//

#ifndef LAYER_H
#define LAYER_H

#include "activations.h"

typedef enum{
    CONVOLUTIONAL, DROPOUT
}LAYER_TYPE;

typedef enum{
    SSE
}COST_TYPE;


typedef struct layer layer;
struct layer{
  LAYER_TYPE type;
  ACTIVATION activation;
  COST_TYPE cost_type;
  float *rand;
  int *indexes;
  float *cost; //经过一个网络以后的损失，一般来讲取第一个
  float *filters; //x*n个 filter的size*size大小的参数
  float *filter_updates;
  float *biases;
  float *bias_updates;
  float *weights;
  float *weight_updates;
  float *col_image;// 为了便于卷积，专门将数据转化为n＊out_h*out_w,并且对应于各个filter
  int *input_layers;
  int *input_sizes;
  float *delta; //梯度
  float *output;
  float *squared;
  float *norms;

  int batch;
  int h,w,c;


  #ifdef GPU
  float *rand_gpu;
  float *indexes_gpu;
  float *filters_gpu;
  float *filter_updates_gpu;
  float *col_image_gpu;
  float *weights_gpu;
  float *weight_updates_gpu;
  float *biases_gpu;
  float *bias_updates_gpu;
  float *output_gpu;
  float *delta_gpu;
  float *squared_gpu;
  float *norms_gpu;
  #endif
};

void free_layer(layer);


#endif //LAYER_H
