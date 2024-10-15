#include <cinttypes>

#include "merlin/array/array.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/candy/optimizer.hpp"
#include "merlin/config.hpp"
#include "merlin/cuda/copy_helpers.hpp"
#include "merlin/cuda/device.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

#include <algorithm>

using namespace merlin;

__global__ void print_model_shr(candy::Model * model_ptr, candy::Optimizer * optmz_ptr, int i) {
    extern __shared__ candy::Model share_ptr[];
    std::uint64_t thread_idx = flatten_thread_index(), block_size = size_of_block();
    auto [_, model_shr, optmz_shr, i_shr] = cuda::copy_objects(share_ptr, thread_idx, block_size, *model_ptr,
                                                               *optmz_ptr, i);
    CudaOut("Candecomp Model on GPU (rank = %" PRIu64 "):\n", model_shr->rank());
    for (int i = 0; i < model_shr->rank(); i++) {
        std::printf("Rank %d:", i);
        for (int j = 0; j < model_shr->rstride(); j++) {
            std::printf(" %.2f", model_shr->get_concatenated_param_vectors(i)[j]);
        }
        std::printf("\n");
    }
    CudaOut("Model get: %f.\n", model_shr->get(0,0,0));
    CudaOut("shared integer: %d.\n", *i_shr);
    CudaOut("Optimizer type: %u.\n", (unsigned) optmz_shr->static_data().index());
    CudaOut("Optimizer dynamic size: %" PRIu64 ".\n", optmz_shr->dynamic_size());
}

__global__ void print_model(candy::Model * model_ptr, candy::Optimizer * optmz_ptr) {
    CudaOut("Candecomp Model on GPU (rank = %" PRIu64 "):\n", model_ptr->rank());
    for (int i = 0; i < model_ptr->rank(); i++) {
        std::printf("Rank %d:", i);
        for (int j = 0; j < model_ptr->rstride(); j++) {
            std::printf(" %.2f", model_ptr->get_concatenated_param_vectors(i)[j]);
        }
        std::printf("\n");
    }
    Index index = {1, 1};
    CudaOut("Model eval: %f.\n", model_ptr->eval(index));
    model_ptr->get(1, 2, 1) = 1.70;
    CudaOut("Optimizer type: %u.\n", (unsigned) optmz_ptr->static_data().index());
    CudaOut("Optimizer dynamic size: %" PRIu64 ".\n", optmz_ptr->dynamic_size());
}

int main(void) {
    // set GPU
    cuda::Device gpu(0);
    gpu.set_as_current();

    // Initialize model
    candy::Model model(
        {
            {1.0, 0.5,      2.1, 0.25    },
            {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}
        },
        2);
    Message("Model: {}\n", model.str());
    std::cout << model.eval({1, 1}) << "\n";

    // initialize optimizer
    candy::Optimizer optmz = candy::optmz::create_adagrad(0.1, model.num_params());

    // Copy model on GPU
    cuda::Dispatcher mem(0, model, optmz);
    candy::Model * gpu_model = mem.get<0>();
    candy::Optimizer * gpu_optmz = mem.get<1>();
    print_model<<<1, 1>>>(gpu_model, gpu_optmz);
    print_model_shr<<<1, 1, model.sharedmem_size() + optmz.sharedmem_size() + sizeof(int)>>>(gpu_model, gpu_optmz, 100);
    cuda::Device::synchronize();

    // Copy model from GPU
    candy::Model model_cpu(
        {
            {0.1, 0.1, 0.1, 0.1},
            {0.2, 0.2, 0.2, 0.2, 0.2, 0.2}
        },
        2);
    model_cpu.copy_from_gpu(reinterpret_cast<double *>(gpu_model + 1));
    Message("Model copied from GPU: {}\n", model_cpu.str());
}
