// Copyright 2022 quocdang1998
#include "spgrid/interpolator.hpp"

#include <array>   // std::array
#include <vector>  // std::vector

#include "merlin/array/array.hpp"      // merlin::array::Array
#include "merlin/array/operation.hpp"  // merlin::array::contiguous_strides
#include "merlin/array/parcel.hpp"     // merlin::array::Parcel
#include "merlin/cuda_interface.hpp"   // merlin::cuda_mem_alloc, merlin::cuda_mem_free,
                                       // merlin::cuda_mem_cpy_device_to_host
#include "merlin/cuda/memory.hpp"      // merlin::cuda::Memory
#include "merlin/logger.hpp"           // FAILURE
#include "merlin/splint/tools.hpp"     // merlin::splint::eval_intpl_gpu

#define push_gpu(gpu)                                                                                                  \
    std::uintptr_t current_ctx = gpu.push_context()
#define pop_gpu()                                                                                                      \
    merlin::cuda::Device::pop_context(current_ctx)

namespace spgrid {

// Add result from GPU back to CPU
static void update_result(::cudaStream_t stream, ::cudaError_t status, void * callback_array) {
    merlin::floatvec ** argv = reinterpret_cast<merlin::floatvec **>(callback_array);
    merlin::floatvec & evaluated_values = *(argv[0]);
    merlin::floatvec & result_from_gpu = *(argv[1]);
    for (std::uint64_t i_point = 0; i_point < evaluated_values.size(); i_point++) {
        evaluated_values[i_point] += result_from_gpu[i_point];
    }
}

// Evaluate interpolation by GPU parallelism
merlin::floatvec Interpolator::evaluate(const merlin::array::Parcel & points, std::uint64_t n_threads,
                                        const merlin::cuda::Stream & stream) {
    // check points array
    if (points.ndim() != 2) {
        FAILURE(std::invalid_argument, "Expected array of coordinates a 2D table.\n");
    }
    if (!points.is_c_contiguous()) {
        FAILURE(std::invalid_argument, "Expected array of coordinates to be C-contiguous.\n");
    }
    if (points.shape()[1] != this->grid_.ndim()) {
        FAILURE(std::invalid_argument, "Array of coordinates and interpolator have different dimension.\n");
    }
    // push context corresponding to the stream
    merlin::floatvec evaluated_values(points.shape()[0]), result_from_gpu(points.shape()[0]);
    std::array<void *, 2> callback_array = {&evaluated_values, &result_from_gpu};
    std::vector<std::vector<char>> grid_buffers;
    grid_buffers.reserve(this->grid_.nlevel());
    push_gpu(stream.get_gpu());
    // allocate memory storing result on GPU
    void * result_gpu = merlin::cuda_mem_alloc(points.shape()[0] * sizeof(double), stream.get_stream_ptr());
    // evaluate for each level
    for (std::uint64_t i_level = 0; i_level < this->grid_.nlevel(); i_level++) {
        std::printf("Level: %llu:\n", i_level);
        // allocate memory and get Cartesian grid
        grid_buffers.push_back(std::vector<char>(this->grid_.fullgrid().cumalloc_size()));
        std::vector<char> & grid_buffer = grid_buffers[i_level];
        ::cudaHostRegister(grid_buffer.data(), grid_buffer.size(), cudaHostRegisterDefault);
        merlin::grid::CartesianGrid level_grid(this->grid_.get_grid_at_level(i_level, grid_buffer.data()));
        std::printf("    Grid: %s\n", level_grid.str().c_str());
        // copy coefficient of the grid to GPU
        std::printf("    Coeff pointer: %p\n", this->coeff_by_level_[i_level]);
        merlin::array::Array coeff_cpu(this->coeff_by_level_[i_level], level_grid.shape(),
                                       merlin::array::contiguous_strides(level_grid.shape(), sizeof(double)), false);
        merlin::array::Parcel coeff_gpu(coeff_cpu.shape(), stream);
        coeff_gpu.transfer_data_to_gpu(coeff_cpu, stream);
        stream.synchronize();
        std::printf("    Coeffs: %s\n", coeff_gpu.str().c_str());
        // copy to GPU and execute
        /*
        merlin::cuda::Memory mem(stream.get_stream_ptr(), level_grid, this->method_);
        std::uint64_t sharedmem_size = level_grid.sharedmem_size() + this->method_.sharedmem_size();
        merlin::splint::eval_intpl_gpu(coeff_gpu.data(), mem.get<0>(), mem.get<1>(), points.data(), points.shape()[0],
                                       reinterpret_cast<double *>(result_gpu), n_threads, this->grid_.ndim(),
                                       sharedmem_size, &stream);
        // copy value back to CPU and update
        merlin::cuda_mem_cpy_device_to_host(result_from_gpu.data(), result_gpu, points.shape()[0] * sizeof(double),
                                            stream.get_stream_ptr());
        stream.add_callback(update_result, callback_array.data());
        */
    }
    merlin::cuda_mem_free(result_gpu, stream.get_stream_ptr());
    pop_gpu();
    return evaluated_values;
}

}  // namespace spgrid
