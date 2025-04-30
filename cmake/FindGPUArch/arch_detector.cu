#include <iostream>

#include <cuda_runtime.h>

int main(void) {
    int ngpu = 0;
    ::cudaGetDeviceCount(&ngpu);
    for (int i_gpu = 0; i_gpu < ngpu; i_gpu++) {
        if (i_gpu != 0) {
            std::cout << ";";
        }
        ::cudaDeviceProp prop;
        ::cudaGetDeviceProperties(&prop, i_gpu);
        std::cout << prop.major << prop.minor;
    }
}
