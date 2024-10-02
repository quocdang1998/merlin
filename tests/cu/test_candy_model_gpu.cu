#include <cinttypes>

#include "merlin/candy/model.hpp"
#include "merlin/config.hpp"
#include "merlin/cuda/copy_helpers.hpp"
#include "merlin/cuda/device.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

__global__ void print_model_shr(candy::Model * model_ptr) {
    extern __shared__ candy::Model share_ptr[];
    std::uint64_t thread_idx = flatten_thread_index(), block_size = size_of_block();
    auto [_, __] = cuda::copy_objects(share_ptr, thread_idx, block_size, *model_ptr);
    CudaOut("Candecomp Model on GPU (rank = %" PRIu64 "):\n", share_ptr->rank());
    for (int i = 0; i < share_ptr->ndim(); i++) {
        std::printf("Vector %d:", i);
        for (int j = 0; j < share_ptr->rshape()[i]; j++) {
            std::printf(" %.2f", share_ptr->param_vectors()[i][j]);
        }
        std::printf("\n");
    }
    Index index;
    index.fill(0);
    CudaOut("Model get: %f.\n", share_ptr->get(0,0,0));
}

__global__ void print_model(candy::Model * model_ptr) {
    CudaOut("Candecomp Model on GPU (rank = %" PRIu64 "):\n", model_ptr->rank());
    for (int i = 0; i < model_ptr->ndim(); i++) {
        std::printf("Vector %d:", i);
        for (int j = 0; j < model_ptr->rshape()[i]; j++) {
            std::printf(" %.2f", model_ptr->param_vectors()[i][j]);
        }
        std::printf("\n");
    }
    Index index;
    index.fill(0);
    CudaOut("Model eval: %f.\n", model_ptr->eval(index));
    model_ptr->get(1, 2, 1) = 1.70;
}

int main(void) {
    // set GPU
    cuda::Device gpu(0);
    gpu.set_as_current();

    // Initialize model
    candy::Model model(
        {
            {1.0, 0.5, 2.1, 0.25},
            {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}
        },
        2);
    Message("Model: %s\n", model.str().c_str());

    // Copy model on GPU
    cuda::Dispatcher mem(0, model);
    candy::Model * gpu_model = mem.get<0>();
    print_model<<<1, 1>>>(gpu_model);
    print_model_shr<<<1, 1, model.sharedmem_size() + sizeof(double)>>>(gpu_model);
    cuda::Device::synchronize();

    // Copy model from GPU
    candy::Model model_cpu(
        {
            {0.1, 0.1, 0.1, 0.1},
            {0.2, 0.2, 0.2, 0.2, 0.2, 0.2}
        },
        2);
    model_cpu.copy_from_gpu(reinterpret_cast<double *>(gpu_model + 1));
    Message("Model copied from GPU: %s\n", model_cpu.str().c_str());
}
