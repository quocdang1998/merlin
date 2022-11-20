#include "merlin/logger.hpp"

#include "merlin/cuda_decorator.hpp"

__cuhostdev__ void errfunction(void) {
    CUHDERR(std::runtime_error, "An error message %d.\n", 2);
}

__global__ void an_errored_kernel(void) {
    errfunction();
    // this message shouldn''t be printed out
    CUDAOUT("A message shoudn''t be printed.\n");
}

int main(void) {
    // CPU code
    try {
        errfunction();
    } catch(std::exception & e) {}
    // GPU code
    an_errored_kernel<<<1,1>>>();
}
