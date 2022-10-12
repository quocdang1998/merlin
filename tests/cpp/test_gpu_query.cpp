#include "merlin/device/gpu_query.hpp"

int main(void) {
    merlin::device::print_all_gpu_specification();
    merlin::device::test_all_gpu();
}
