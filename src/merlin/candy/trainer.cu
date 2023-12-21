// Copyright 2023 quocdang1998
#include "merlin/candy/trainer.hpp"

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/cuda/memory.hpp"  // merlin::cuda::Memory
#include "merlin/cuda/stream.hpp"  // merlin::cuda::Stream

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

/** @brief Allocate memory on GPU for the trainer.*/
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

}  // namespace merlin
