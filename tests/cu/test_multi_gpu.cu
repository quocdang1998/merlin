#include "merlin/cuda/device.hpp"
#include "merlin/cuda/context.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/array/array.hpp"
#include "merlin/array/copy.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/array/slice.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

#include <omp.h>  // #pragma
#include <cstdint>  // std::uint64_t
#include <numeric>  // std::iota

__global__ void print_parcel_data(merlin::array::Parcel * pointer_on_gpu) {
    merlin::array::Parcel & parcel_on_gpu = *pointer_on_gpu;
    CUDAOUT("Result from GPU %d: %.1f %.1f %.1f %.1f %.1f.\n", parcel_on_gpu.device().id(), parcel_on_gpu[0],
            parcel_on_gpu[1], parcel_on_gpu[2], parcel_on_gpu[3], parcel_on_gpu[4]);
}

int main(void) {
    double A_ptr[65536];
    std::iota<double *, double>(A_ptr, A_ptr + 65536, 0.0);  // A = {0, 1, 2, ...}
    merlin::intvec A_shape({64, 128, 8});
    merlin::intvec A_strides = merlin::array::contiguous_strides(A_shape, sizeof(double));
    merlin::array::Array A_array(A_ptr, 3, A_shape.data(), A_strides.data(), false);
    merlin::cuda::Context default_context = merlin::cuda::Context::get_current();

    std::uint64_t num_gpu = merlin::cuda::Device::get_num_gpu();
    #pragma omp parallel for
    for (std::int64_t i_gpu = 0; i_gpu < num_gpu; i_gpu++) {
        merlin::cuda::Device gpu(i_gpu);
        gpu.set_as_current();
        merlin::cuda::Context current_thread = merlin::cuda::Context::get_current();
        if (current_thread != default_context) {
            MESSAGE("Context changed when changing GPU.\n");
        }
        merlin::cuda::Stream s(merlin::cuda::Stream::Setting::Default, 0);
        merlin::array::Array sliced_array(A_array, {{}, {}, {static_cast<std::uint64_t>(i_gpu)}});
        merlin::array::Parcel array_on_gpu(sliced_array.shape());
        array_on_gpu.transfer_data_to_gpu(sliced_array, s);
        merlin::array::Parcel * pointer_on_gpu;
        ::cudaMalloc(&pointer_on_gpu, array_on_gpu.malloc_size());
        array_on_gpu.copy_to_gpu(pointer_on_gpu, pointer_on_gpu+1);
        print_parcel_data<<<1,1,0,reinterpret_cast<::cudaStream_t>(s.get_stream_ptr())>>>(pointer_on_gpu);
        ::cudaFree(pointer_on_gpu);
    }
}
