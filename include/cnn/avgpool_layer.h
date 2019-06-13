//
// Created by tonye on 2019-06-13.
//

#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "layer.h"
#include "network.h"


typedef layer avgpool_layer;

void forward_avgpool_layer(const avgpool_layer l, network_state state);

#endif //AVGPOOL_LAYER_H
