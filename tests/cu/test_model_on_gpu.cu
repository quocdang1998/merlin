#include <cinttypes>

#include "merlin/candy/model.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/logger.hpp"

__global__ void print_model_shared_mem(merlin::candy::Model * model_ptr) {
    extern __shared__ merlin::candy::Model share_ptr[];
    auto [_, __] = merlin::cuda::copy_objects(share_ptr, *model_ptr);
    CUDAOUT("Candecomp Model on GPU (rank = %" PRIu64 "):\n", share_ptr->rank());
    for (int i = 0; i < share_ptr->ndim(); i++) {
        std::printf("Vector %d:", i);
        for (int j = 0; j < share_ptr->rshape()[i]; j++) {
            std::printf(" %.2f", share_ptr->param_vectors()[i][j]);
        }
        std::printf("\n");
    }
    merlin::intvec index = {0, 0};
    CUDAOUT("Model get: %f.\n", share_ptr->get(0,0,0));
}

__global__ void print_model(merlin::candy::Model * model_ptr) {
    CUDAOUT("Candecomp Model on GPU (rank = %" PRIu64 "):\n", model_ptr->rank());
    for (int i = 0; i < model_ptr->ndim(); i++) {
        std::printf("Vector %d:", i);
        for (int j = 0; j < model_ptr->rshape()[i]; j++) {
            std::printf(" %.2f", model_ptr->param_vectors()[i][j]);
        }
        std::printf("\n");
    }
    merlin::intvec index = {0, 0};
    CUDAOUT("Model eval: %f.\n", model_ptr->eval(index));
    model_ptr->get(1, 2, 1) = 1.70;
}

int main(void) {
    // Initialize model
    merlin::candy::Model model({{1.0, 0.5, 2.1, 0.25}, {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}}, 2);
    MESSAGE("Model: %s\n", model.str().c_str());

    // Copy model on GPU
    merlin::cuda::Memory mem(0, model);
    merlin::candy::Model * gpu_model = mem.get<0>();
    print_model<<<1, 1>>>(gpu_model);
    print_model_shared_mem<<<1, 1, model.sharedmem_size() + sizeof(double)>>>(gpu_model);
    cudaDeviceSynchronize();

    // Copy model from GPU
    merlin::candy::Model model_cpu({{0.1, 0.1, 0.1, 0.1}, {0.2, 0.2, 0.2, 0.2, 0.2, 0.2}}, 2);
    model_cpu.copy_from_gpu(reinterpret_cast<double *>(gpu_model+1));
    MESSAGE("Model copied from GPU: %s\n", model_cpu.str().c_str());
}
