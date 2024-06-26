#include "merlin/cuda/device.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/env.hpp"
#include "merlin/grid/regular_grid.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"

using namespace merlin;

__global__ void print_reg_grid(const grid::RegularGrid * grid) {
    double point[3];
    std::uint64_t thread_idx = flatten_thread_index(), block_size = size_of_block();
    grid->get(thread_idx, point);
    CudaOut("Point %u: %f %f %f\n", unsigned(thread_idx), point[0], point[1], point[2]);
}

int main(void) {
    // create Environment
    Environment::init_cuda(0);

    // create grid
    grid::RegularGrid grid(3);
    grid.push_back({1.1, 2.3, 4.5});
    grid.push_back({-1.9, -2.5, 6.2});
    grid.push_back({1.6, 7.3, 4.8});
    grid.push_back({-9.0, 2.4, 7.2});
    grid.push_back({5.0, 7.1, 8.2});
    grid.pop_back();

    // print
    Message("RegularGrid CPU: %s\n", grid.str().c_str());

    // copy to GPU
    cuda::Memory mem(0, grid);
    print_reg_grid<<<1, grid.size()>>>(mem.get<0>());
    cuda::Device::synchronize();
}
