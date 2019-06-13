//
// Created by tonye on 2019-06-13.
//

#ifndef NETWORK_H
#define NETWORK_H

typedef struct network{
    int n;
} network;

typedef struct network_state{
    float *truth; //数据的标签，或者是真实值
} network_state;

#endif //NETWORK_H
