#include<stdio.h>

/**
 * 使用extern和包含头文件来引用函数的区别：
 * extern的引用方式比包含头文件要间接得多。
 * extern的使用方法是直接了当的，想引用哪个函数就用extern声明哪个函数。这大概是kiss原则的一种体现。
 * 这样做的一个明显的好处是，会加速程序的编译(确切地说是预处理)的过程，节省时间。
 * 在大型C程序编译过程中，这种差异是非常明显的。
 *
 * KISS 原则是用户体验的高层境界，简单地理解这句话，
 * 就是要把一个产品做得连白痴都会用，因而也被称为“懒人原则”。换句话说来，“简单就是美”.
 */
//extern "C" void cube(int *a, int n);
//extern "C" void print(int *a, int n);

#include "include/cnn/cube.cuh"
#include "include/cnn/print.h"
#include "include/cnn/activations.h"

int main(){
    int N=10;
    int a[10];
    cube(a,N);
    print(&a[0],N);
    get_activation("a");

}