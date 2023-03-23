// Copyright 2022 quocdang1998

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin {

// Calculate Lagrange interpolation coefficients on a full Cartesian grid using GPU
void interpolant::calc_newton_coeffs_gpu(const interpolant::CartesianGrid & grid, const array::Parcel & value,
                                         array::Parcel & coeff, const cuda::Stream & stream) {
    // check for validity
    stream.check_cuda_context();
    // get associated 1D grid to calculate on
    std::uint64_t ndim = grid.ndim();
    // copy value to coeff
    if (&coeff != &value) {
        ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
        if (coeff.device() != value.device()) {
            auto copy_func = std::bind(::cudaMemcpyPeerAsync, std::placeholders::_1, coeff.device().id(),
                                       std::placeholders::_2, value.device().id(), std::placeholders::_3, cuda_stream);
            array::array_copy(&coeff, &value, copy_func);
        } else {
            auto copy_func = std::bind(::cudaMemcpyAsync, std::placeholders::_1, std::placeholders::_2,
                                    std::placeholders::_3, ::cudaMemcpyDeviceToDevice, cuda_stream);
            array::array_copy(&coeff, &value, copy_func);
        }
    }
    calc_newton_coeff_single_core_gpu(grid, &coeff, 1, stream);
    #ifdef __deprecated
    // get max recursive dimension
    static std::uint64_t parallel_limit = Environment::parallel_chunk;
    intvec total_shape = grid.get_grid_shape();
    std::uint64_t cumulative_size = 1, dim_max = 0;
    while (dim_max < ndim) {
        cumulative_size *= total_shape[dim_max];
        if (cumulative_size >= parallel_limit) {
            break;
        }
        dim_max++;
    }
    // trivial case: size too small
    if (dim_max == ndim) {
        calc_newton_coeff_single_core_gpu(grid, &coeff, 1, stream);
        return;
    }
    // recursive calculation
    Vector<array::Parcel> sub_slices = make_vector<array::Parcel>(cumulative_size);
    calc_newton_coeffs_gpu_recursive(grid, coeff, dim_max, sub_slices, 0, stream);
    // parallel calculation after that
    calc_newton_coeff_single_core_gpu(grid, sub_slices.data(), sub_slices.size(), stream);
    #endif  // __deprecated
}

// Evaluate Newton interpolation on a full Cartesian grid using CPU (supposed shape of grid == shape of coeff)
double eval_newton_cpu2(const interpolant::CartesianGrid & grid, const array::Array & coeff,
                        const Vector<double> & x) {
    long double result = 0;
    std::uint64_t ndim = grid.ndim();
    const Vector<double> & grid_vector = grid.grid_vectors()[ndim - coeff.ndim()];
    // trivial case
    if (coeff.ndim() == 1) {
        const std::uint64_t & shape = coeff.shape()[0];
        result += coeff.get({shape-1});
        for (std::int64_t i = shape-2; i >= 0; i--) {
            result *= (x[ndim - coeff.ndim()] - grid_vector[i]);
            result += coeff.get({static_cast<std::uint64_t>(i)});
        }
        return result;
    }
    // recursively calculate for non-trivial case
    const std::uint64_t & shape = coeff.shape()[0];
    Vector<array::Slice> slice_i(coeff.ndim());
    slice_i[0] = array::Slice({shape-1});
    array::Array array_coeff_i(coeff, slice_i);
    array_coeff_i.remove_dim(0);
    result += interpolant::eval_newton_cpu2(grid, array_coeff_i, x);
    for (std::int64_t i = shape-2; i >= 0; i--) {
        result *= (x[ndim - coeff.ndim()] - grid_vector[i]);
        slice_i[0] = array::Slice({static_cast<std::uint64_t>(i)});
        array_coeff_i = array::Array(coeff, slice_i);
        array_coeff_i.remove_dim(0);
        result += interpolant::eval_newton_cpu2(grid, array_coeff_i, x);
    }
    return result;
}

void calc_newton_coeff_single_core_gpu(const interpolant::CartesianGrid & grid, array::Parcel * p_coeff,
                                       std::uint64_t size, const cuda::Stream & stream) {
    // check for validity
    stream.check_cuda_context();
    // copy grid and coeff to GPU
    void * gpu_memory;
    std::uint64_t coeff_size = p_coeff->malloc_size();
    ::cudaError_t err_ = ::cudaMalloc(&gpu_memory, grid.malloc_size() + size * coeff_size);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Alloc data faile with message \"%s\"\n", ::cudaGetErrorString(err_));
    }
    interpolant::CartesianGrid * grid_gpu = reinterpret_cast<interpolant::CartesianGrid *>(gpu_memory);
    void * coeff_gpu_data = grid.copy_to_gpu(grid_gpu, grid_gpu+1, stream.get_stream_ptr());
    std::uintptr_t coeff_i_destination_ptr = reinterpret_cast<std::uintptr_t>(coeff_gpu_data);
    for (std::uint64_t i_coeff = 0; i_coeff < size; i_coeff++) {
        array::Parcel * p_coeff_dest = reinterpret_cast<array::Parcel *>(coeff_i_destination_ptr);
        p_coeff[i_coeff].copy_to_gpu(p_coeff_dest, p_coeff_dest+1);
        coeff_i_destination_ptr += coeff_size;
    }
    // call kernel
    std::uint64_t shared_mem_size = grid.malloc_size() + Environment::default_block_size * coeff_size;
    interpolant::call_single_core_kernel(grid_gpu, reinterpret_cast<array::Parcel *>(coeff_gpu_data), size,
                                         shared_mem_size, stream.get_stream_ptr());
    // defer deallocation
    int gpu = stream.get_gpu().id();
    Environment::deferred_gpu_pointer.push_back(std::pair(gpu, gpu_memory));
}

}  // namespace merlin
