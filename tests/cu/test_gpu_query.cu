#include "merlin/cuda/gpu_query.hpp"
#include "merlin/cuda/context.hpp"
#include "merlin/cuda/stream.hpp"

#include "merlin/logger.hpp"

#include <cinttypes>

__global__ void foo(void) {
    CUDAOUT("A message.\n");
}

void CUDART_CB str_callback(void * data) {
    int & a = *(static_cast<int *>(data));
    MESSAGE("Callback argument: %d\n", a);
}

int main(void) {
    merlin::cuda::print_all_gpu_specification();
    std::uint64_t stack_size = merlin::cuda::Device::limit(merlin::cuda::Device::Limit::StackSize);
    MESSAGE("Stack size: %" PRIu64 ".\n", stack_size);
    // merlin::cuda::test_all_gpu();

    merlin::cuda::Context c = merlin::cuda::Context::get_current();

    merlin::cuda::Stream s(c);
    int data = 1;
    ::foo<<<2, 2, 0, reinterpret_cast<cudaStream_t>(s.stream())>>>();
    s.launch_cpu_function(str_callback, &data);
    s.synchronize();
}
