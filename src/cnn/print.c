#include<stdio.h>
#include"../../include/cnn/print.h"

void print(int *a, int n){
    printf("\tHola\n");
    for(int i=0;i<n;i++)
        printf("    %d\n",*(a+i));
    printf("\tAdios\n");
}