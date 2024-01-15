#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/cuda_interface.hpp"
#include "merlin/env.hpp"
#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/splint/interpolator.hpp"
#include "merlin/splint/tools.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

double foo(const merlin::floatvec & v) {
    return (2.f*v[0] + v[2])*v[2] + 3.f*v[1];
}

int main(void) {
    // initialize data and grid
    merlin::grid::CartesianGrid cart_gr({{0.1, 0.2, 0.3}, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.25, 0.5}});
    merlin::array::Array value(cart_gr.shape());
    for (std::uint64_t i = 0; i < cart_gr.size(); i++) {
        merlin::intvec index(merlin::contiguous_to_ndim_idx(i, cart_gr.shape()));
        value[index] = foo(cart_gr[index]);
    }
    MESSAGE("Value: %s\n", value.str().c_str());

    // calculate Newton coefficients (CPU)
    merlin::Vector<merlin::splint::Method> methods = {
        merlin::splint::Method::Newton,
        merlin::splint::Method::Lagrange,
        merlin::splint::Method::Newton
    };

    // initialize interpolator
    merlin::splint::Interpolator interp_cpu(cart_gr, value, methods, merlin::ProcessorType::Cpu);
    merlin::splint::Interpolator interp_gpu(cart_gr, value, methods, merlin::ProcessorType::Gpu);

    // calculate coefficients
    interp_cpu.build_coefficients(8);
    interp_gpu.build_coefficients(32);
    interp_cpu.synchronize();
    MESSAGE("Coefficients calculated by CPU: %s\n", interp_cpu.get_coeff().str().c_str());
    interp_gpu.synchronize();
    MESSAGE("Coefficients calculated by GPU: %s\n", interp_gpu.get_coeff().str().c_str());

    // initialize point
    double point_coordinates_data[9] = {0.0, 2.0, 1.0, 1.0, 1.0, 1.2, 0.5, 0.25, 2.4};
    merlin::array::Array point_cpu(point_coordinates_data, {3, 3}, {3*sizeof(double), sizeof(double)}, false);
    merlin::array::Parcel point_gpu(point_cpu.shape());
    point_gpu.transfer_data_to_gpu(point_cpu);

    // calculate evaluation
    merlin::floatvec cpu_result(3);
    interp_cpu.evaluate(point_cpu, cpu_result, 8);
    merlin::floatvec gpu_result(3);
    interp_gpu.evaluate(point_gpu, gpu_result, 32);
    interp_cpu.synchronize();
    MESSAGE("Value evaluated by CPU: %s\n", cpu_result.str().c_str());
    interp_gpu.synchronize();
    MESSAGE("Value evaluated by GPU: %s\n", gpu_result.str().c_str());

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
