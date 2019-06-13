//
// Created by tonye on 2019-06-13.
//

#include "../../include/cnn/avgpool_layer.h"

void forward_avgpool_layer(const avgpool_layer l, network_state state){
    int b, i, k;

    for(b=0; b < l.batch; b++){
        for(k=0;k<l.c;k++){
            int out_index = k + b*l.c;
        }
    }
}