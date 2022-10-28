#include "merlin/device/gpu_query.hpp"
#include "merlin/logger.hpp"

#include <cinttypes>

int main(void) {
    merlin::device::print_all_gpu_specification();
    merlin::device::test_all_gpu();
    std::uint64_t stack_size = merlin::device::Device::limit(merlin::device::Device::Limit::StackSize);
    MESSAGE("Stack size: %" PRIu64 ".\n", stack_size);
}
