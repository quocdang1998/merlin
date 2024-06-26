#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/env.hpp"
#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/splint/interpolator.hpp"
#include "merlin/splint/tools.hpp"
#include "merlin/synchronizer.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

double foo(const DoubleVec & v) {
    return (2.f*v[0] + v[2])*v[2] + 3.f*v[1];
}

int main(void) {
    // create Environment
    Environment::init_cuda(0);

    // initialize data and grid
    grid::CartesianGrid cart_gr({{0.1, 0.2, 0.3}, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.25, 0.5}});
    array::Array value(cart_gr.shape());
    for (std::uint64_t i = 0; i < cart_gr.size(); i++) {
        value[i] = foo(cart_gr[i]);
    }
    Message("Value: %s\n", value.str().c_str());

    // calculate Newton coefficients (CPU)
    Vector<splint::Method> methods = {
        splint::Method::Newton,
        splint::Method::Lagrange,
        splint::Method::Newton
    };

    // initialize interpolator
    Synchronizer cpu_sync(ProcessorType::Cpu);
    splint::Interpolator interp_cpu(cart_gr, value, methods.data(), cpu_sync);
    Synchronizer gpu_sync(ProcessorType::Gpu);
    splint::Interpolator interp_gpu(cart_gr, value, methods.data(), gpu_sync);

    // calculate coefficients
    interp_cpu.build_coefficients(8);
    interp_gpu.build_coefficients(32);
    cpu_sync.synchronize();
    Message("Coefficients calculated by CPU: %s\n", interp_cpu.get_coeff().str().c_str());
    gpu_sync.synchronize();
    Message("Coefficients calculated by GPU: %s\n", interp_gpu.get_coeff().str().c_str());

    // initialize point
    double point_coordinates_data[9] = {0.0, 2.0, 1.0, 1.0, 1.0, 1.2, 0.5, 0.25, 2.4};
    array::Array point_cpu(point_coordinates_data, {3, 3}, {3*sizeof(double), sizeof(double)}, false);
    array::Parcel point_gpu(point_cpu.shape());
    point_gpu.transfer_data_to_gpu(point_cpu);

    // calculate evaluation
    DoubleVec cpu_result(3);
    interp_cpu.evaluate(point_cpu, cpu_result, 8);
    DoubleVec gpu_result(3);
    interp_gpu.evaluate(point_gpu, gpu_result, 32);
    cpu_sync.synchronize();
    Message("Value evaluated by CPU: %s\n", cpu_result.str().c_str());
    gpu_sync.synchronize();
    Message("Value evaluated by GPU: %s\n", gpu_result.str().c_str());
}
