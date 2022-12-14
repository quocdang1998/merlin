#include "merlin/cuda/device.hpp"
#include "merlin/cuda/context.hpp"
#include "merlin/cuda/stream.hpp"

#include "merlin/logger.hpp"

#include <cinttypes>

__global__ void foo(void) {
    CUDAOUT("A message.\n");
}

void str_callback(::cudaStream_t stream, ::cudaError_t status, void * data) {
    int & a = *(static_cast<int *>(data));
    MESSAGE("Callback argument: %d\n", a);
}

int main(void) {
    merlin::cuda::print_all_gpu_specification();
    std::uint64_t stack_size = merlin::cuda::Device::limit(merlin::cuda::Device::Limit::StackSize);
    MESSAGE("Stack size: %" PRIu64 ".\n", stack_size);
    merlin::cuda::test_all_gpu();

    merlin::cuda::Context c = merlin::cuda::Context::get_current();
    MESSAGE("Current context: %s.\n", c.str().c_str());

    merlin::cuda::Stream s;
    MESSAGE("Default stream: %s.\n", s.str().c_str());
    int data = 1;
    s.check_cuda_context();
    ::foo<<<2, 2, 0, reinterpret_cast<cudaStream_t>(s.get_stream_ptr())>>>();
    s.add_callback(str_callback, &data);
    s.synchronize();
}
