#include "merlin/intpl/cartesian_grid.hpp"
#include "merlin/utils.hpp"

#include "cstdio"

__global__ void print_grid_from_shared_mem(merlin::intpl::CartesianGrid * grid_ptr) {
    CUDAOUT("Cartesian Grid on GPU (shared mem):\n");
    std::uint64_t thread_idx = merlin::flatten_thread_index(), block_size = merlin::size_of_block();
    extern __shared__ merlin::intpl::CartesianGrid share_ptr[];
    grid_ptr->copy_by_block(share_ptr, share_ptr+1, thread_idx, block_size);
    for (int i = 0; i < share_ptr->ndim(); i++) {
        std::printf("Vector %d:", i);
        for (int j = 0; j < share_ptr->grid_vectors()[i].size(); j++) {
            std::printf(" %.2f", share_ptr->grid_vectors()[i][j]);
        }
        std::printf("\n");
    }
}

__global__ void print_grid(merlin::intpl::CartesianGrid * grid_ptr) {
    CUDAOUT("Cartesian Grid on GPU:\n");
    for (int i = 0; i < grid_ptr->ndim(); i++) {
        std::printf("Vector %d:", i);
        for (int j = 0; j < grid_ptr->grid_vectors()[i].size(); j++) {
            std::printf(" %.2f", grid_ptr->grid_vectors()[i][j]);
        }
        std::printf("\n");
    }
}

int main(void) {
    merlin::Vector<double> v1 = {0.1, 0.2, 0.3};
    merlin::Vector<double> v2 = {1.0, 2.0, 3.0, 4.0};
    merlin::Vector<double> v3 = {0.0, 0.25};
    merlin::intpl::CartesianGrid cart_gr({v1, v2, v3});

    merlin::intpl::CartesianGrid * gpu_gr;
    cudaMalloc(&gpu_gr, cart_gr.cumalloc_size());
    cart_gr.copy_to_gpu(gpu_gr, gpu_gr+1);
    print_grid<<<1,1>>>(gpu_gr);
    print_grid_from_shared_mem<<<1,1,cart_gr.sharedmem_size()>>>(gpu_gr);
    cudaFree(gpu_gr);
}
