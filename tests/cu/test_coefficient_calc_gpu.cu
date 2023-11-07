#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/cuda_interface.hpp"
#include "merlin/env.hpp"
#include "merlin/splint/cartesian_grid.hpp"
#include "merlin/splint/interpolator.hpp"
#include "merlin/splint/tools.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

double foo(const merlin::floatvec & v) {
    return (2.f*v[0] + v[2])*v[2] + 3.f*v[1];
}

int main(void) {
    // initialize data and grid
    merlin::splint::CartesianGrid cart_gr({{0.1, 0.2, 0.3}, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.25, 0.5}});
    merlin::array::Array value(cart_gr.shape());
    for (std::uint64_t i = 0; i < cart_gr.size(); i++) {
        merlin::intvec index(merlin::contiguous_to_ndim_idx(i, cart_gr.shape()));
        value[index] = foo(cart_gr[index]);
    }
    MESSAGE("Value: %s\n", value.str().c_str());

    // calculate Newton coefficients (CPU)
    merlin::array::Array coeff(value);
    merlin::Vector<merlin::splint::Method> methods = {
        merlin::splint::Method::Newton,
        merlin::splint::Method::Lagrange,
        merlin::splint::Method::Newton
    };
    merlin::splint::construct_coeff_cpu(coeff.data(), cart_gr, methods, 1);
    MESSAGE("Reference coefficients: %s\n", coeff.str().c_str());

    // calculate Newton coefficients (GPU)
    merlin::cuda::Stream stream;
    merlin::array::Parcel coeff_gpu(value.shape(), stream);
    coeff_gpu.transfer_data_to_gpu(value, stream);
    merlin::splint::Interpolator interp(cart_gr, coeff_gpu, methods, stream, 32);
    stream.synchronize();
    MESSAGE("GPU calculated coefficients: %s\n", coeff_gpu.str().c_str());
/*
    // calculate interpolation (CPU)
    merlin::floatvec point_coordinates({0.0, 2.0, 1.0, 1.0, 1.0, 1.2, 0.5, 0.25, 2.4});
    merlin::floatvec result_cpu(3, 0.0);
    merlin::splint::eval_intpl_cpu(coeff.data(), cart_gr, methods, point_coordinates.data(), 3, result_cpu.data(), 3);
    MESSAGE("Reference interpolated values: %s\n", result_cpu.str().c_str());

    // calculate interpolation (GPU)
    double * points_gpu;
    double * result_gpu_ptr;
    ::cudaMalloc(&points_gpu, 9*sizeof(double));
    ::cudaMemcpy(points_gpu, point_coordinates.data(), 9*sizeof(double), ::cudaMemcpyHostToDevice);
    merlin::floatvec result_gpu(3, 0.0);
    ::cudaMalloc(&result_gpu_ptr, 3*sizeof(double));
    ::cudaMemcpy(result_gpu_ptr, result_gpu.data(), 3*sizeof(double), ::cudaMemcpyHostToDevice);
    merlin::splint::eval_intpl_gpu(coeff_gpu.data(), mem.get<0>(), mem.get<1>(),
                                   points_gpu, 3, result_gpu_ptr, 18, 3, shared_mem, &stream);
    ::cudaMemcpy(result_gpu.data(), result_gpu_ptr, 3*sizeof(double), ::cudaMemcpyDeviceToHost);
    MESSAGE("GPU calculated values: %s\n", result_gpu.str().c_str());
    ::cudaFree(points_gpu);
    ::cudaFree(result_gpu_ptr);
*/
}
