#include <cinttypes>

#include "merlin/config.hpp"
#include "merlin/cuda/device.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/env.hpp"
#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"

using namespace merlin;

__global__ void print_grid_from_shared_mem(grid::CartesianGrid * grid_ptr) {
    std::uint64_t thread_idx = flatten_thread_index(), block_size = size_of_block();
    extern __shared__ grid::CartesianGrid share_ptr[];
    grid_ptr->copy_by_block(share_ptr, share_ptr+1, thread_idx, block_size);
    CudaOut("Cartesian Grid on GPU (shared mem):\n");
    for (int i = 0; i < share_ptr->ndim(); i++) {
        std::printf("Vector %d:", i);
        for (int j = 0; j < share_ptr->shape()[i]; j++) {
            std::printf(" %.2f", share_ptr->grid_vectors()[i][j]);
        }
        std::printf("\n");
    }
    CudaOut("Cartesian Grid size on GPU (shared mem): %" PRIu64 "\n", share_ptr->size());
}

__global__ void print_grid(grid::CartesianGrid * grid_ptr) {
    CudaOut("Cartesian Grid on GPU (ndim = %u):\n", unsigned(grid_ptr->ndim()));
    for (int i = 0; i < grid_ptr->ndim(); i++) {
        std::printf("Vector %d:", i);
        for (int j = 0; j < grid_ptr->shape()[i]; j++) {
            std::printf(" %.2f", grid_ptr->grid_vectors()[i][j]);
        }
        std::printf("\n");
    }
}

int main(void) {
    Environment::init_cuda(0);

    DoubleVec v1 = {0.1, 0.2, 0.3};
    DoubleVec v2 = {1.0, 2.0, 3.0, 4.0};
    DoubleVec v3 = {0.0, 0.25};
    grid::CartesianGrid cart_gr({v1, v2, v3});

    cuda::Memory mem(0, cart_gr);

    grid::CartesianGrid * gpu_gr = mem.get<0>();
    print_grid<<<1,1>>>(gpu_gr);
    print_grid_from_shared_mem<<<1,1,cart_gr.sharedmem_size()>>>(gpu_gr);
    cuda::Device::synchronize();
}
