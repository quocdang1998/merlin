#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"
#include "merlin/settings.hpp"

#include <cstdio>

__global__ void print_trivial_copyable(merlin::Index * p_index, double * p_a) {
    std::uint64_t thread_idx = merlin::flatten_thread_index(), block_size = merlin::size_of_block();
    CUDAOUT("Index element: %u\nFloat element %f\n", unsigned((*p_index)[thread_idx]), *p_a);
}

__global__ void print_grid_from_shared_mem(merlin::grid::CartesianGrid * grid_ptr) {
    std::uint64_t thread_idx = merlin::flatten_thread_index(), block_size = merlin::size_of_block();
    extern __shared__ merlin::grid::CartesianGrid share_ptr[];
    grid_ptr->copy_by_block(share_ptr, share_ptr+1, thread_idx, block_size);
    CUDAOUT("Cartesian Grid on GPU (shared mem):\n");
    for (int i = 0; i < share_ptr->ndim(); i++) {
        std::printf("Vector %d:", i);
        for (int j = 0; j < share_ptr->shape()[i]; j++) {
            std::printf(" %.2f", share_ptr->grid_vectors()[i][j]);
        }
        std::printf("\n");
    }
    CUDAOUT("Cartesian Grid size on GPU (shared mem): %lu\n", share_ptr->size());
}

__global__ void print_grid(merlin::grid::CartesianGrid * grid_ptr) {
    CUDAOUT("Cartesian Grid on GPU (ndim = %u):\n", unsigned(grid_ptr->ndim()));
    for (int i = 0; i < grid_ptr->ndim(); i++) {
        std::printf("Vector %d:", i);
        for (int j = 0; j < grid_ptr->shape()[i]; j++) {
            std::printf(" %.2f", grid_ptr->grid_vectors()[i][j]);
        }
        std::printf("\n");
    }
}

int main(void) {
    merlin::Vector<double> v1 = {0.1, 0.2, 0.3};
    merlin::Vector<double> v2 = {1.0, 2.0, 3.0, 4.0};
    merlin::Vector<double> v3 = {0.0, 0.25};
    merlin::grid::CartesianGrid cart_gr({v1, v2, v3});
    merlin::Index random_array;
    random_array.fill(5);
    double a = 7;

    merlin::cuda::Memory mem(0, cart_gr, random_array, a);

    merlin::grid::CartesianGrid * gpu_gr = mem.get<0>();
    print_grid<<<1,1>>>(gpu_gr);
    print_grid_from_shared_mem<<<1,1,cart_gr.sharedmem_size()>>>(gpu_gr);

    print_trivial_copyable<<<1, 2>>>(mem.get<1>(), mem.get<2>());
}
