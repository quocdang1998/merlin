// Copyright 2023 quocdang1998
#include "merlin/candy/trainer.hpp"

#include "merlin/array/array.hpp"      // merlin::array::Array
#include "merlin/array/parcel.hpp"     // merlin::array::Parcel
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/cuda/memory.hpp"      // merlin::cuda::Memory
#include "merlin/cuda/stream.hpp"      // merlin::cuda::Stream

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Allocate memory on GPU for the trainer
void candy::create_trainer_gpu_ptr(const candy::Model & cpu_model, const array::Array & cpu_data,
                                   const candy::Optimizer & cpu_optimizer, candy::Model *& gpu_model,
                                   array::NdData *& gpu_data, candy::Optimizer *& gpu_optimizer,
                                   array::Parcel *& parcel_data, cuda::Stream & stream) {
    // create Parcel object
    parcel_data = new array::Parcel(cpu_data.shape(), stream);
    parcel_data->transfer_data_to_gpu(cpu_data, stream);
    // copy objects to GPU
    cuda::Memory gpu_mem(stream.get_stream_ptr(), cpu_model, *parcel_data, cpu_optimizer);
    gpu_model = gpu_mem.get<0>();
    gpu_data = gpu_mem.get<1>();
    gpu_optimizer = gpu_mem.get<2>();
    gpu_mem.disown();
}

// ---------------------------------------------------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------------------------------------------------

// Get a copy to the current CP model
candy::Model candy::Trainer::get_model(void) const {
    // CPU direct dereference
    if (!(this->on_gpu())) {
        return candy::Model(*(this->p_model_));
    }
    // GPU dereference
    char * model_buffer = new char[sizeof(candy::Model)];
    ::cudaMemcpy(model_buffer, this->p_model_, sizeof(candy::Model), ::cudaMemcpyDeviceToHost);
    candy::Model & model_gpu = *(reinterpret_cast<candy::Model *>(model_buffer));
    // get ndim and rank
    std::uint64_t ndim = model_gpu.ndim();
    std::uint64_t rank = model_gpu.rank();
    // get rshape
    Index model_rshape(model_gpu.rshape());
    // allocate result model
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        model_rshape[i_dim] /= rank;
    }
    candy::Model result(model_rshape, rank);
    // copy data to result model
    result.copy_from_gpu(reinterpret_cast<double *>(this->p_model_ + 1));
    delete[] model_buffer;
    return result;
}

}  // namespace merlin
