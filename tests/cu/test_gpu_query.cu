#include "merlin/cuda/gpu_query.hpp"
#include "merlin/cuda/context.hpp"
#include "merlin/logger.hpp"

#include <cinttypes>

#include "omp.h"

int main(void) {
    merlin::cuda::print_all_gpu_specification();
    merlin::cuda::test_all_gpu();
    std::uint64_t stack_size = merlin::cuda::Device::limit(merlin::cuda::Device::Limit::StackSize);
    MESSAGE("Stack size: %" PRIu64 ".\n", stack_size);

    merlin::cuda::Context c = merlin::cuda::Context::get_current();
}
