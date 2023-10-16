#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/cuda_interface.hpp"
#include "merlin/env.hpp"
#include "merlin/splint/cartesian_grid.hpp"
#include "merlin/splint/tools.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

double foo(const merlin::floatvec & v) {
    return (2.f*v[0] + v[2])*v[2] + 3.f*v[1];
}

int main(void) {
    /*
    merlin::intvec dims = {2, 3, 3};
    merlin::array::Array value_cpu(dims);

    merlin::intpl::CartesianGrid grid({{0, 4.8}, {0.0, 1.0, 1.5}, {0.0, 1.0, 2.0}});
    for (std::uint64_t i_point = 0; i_point < grid.size(); i_point++) {
        merlin::Vector<double> point = grid[i_point];
        value_cpu.set(i_point, f(point[0], point[1], point[2]));
    }
    MESSAGE("Initial array: %s\n", value_cpu.str().c_str());

    merlin::cuda::Stream stream(merlin::cuda::StreamSetting::Default);
    merlin::array::Parcel coeff(value_cpu.shape());
    coeff.transfer_data_to_gpu(value_cpu, stream);
    merlin::intpl::PolynomialInterpolant plm_int(grid, coeff, merlin::intpl::Method::Lagrange, stream, 32);
    stream.synchronize();
    MESSAGE("Result GPU: %s\n", plm_int.get_coeff().str().c_str());
    merlin::intpl::PolynomialInterpolant plm_int_cpu(grid, value_cpu, merlin::intpl::Method::Lagrange);
    MESSAGE("Result CPU: %s\n", plm_int_cpu.get_coeff().str().c_str());

    merlin::array::Array points(merlin::intvec({1, value_cpu.ndim()}));
    points[{0,0}] = 2.2; points[{0,1}] = 1.0; points[{0,2}] = 2.0;
    merlin::array::Parcel points_gpu(points.shape());
    points_gpu.transfer_data_to_gpu(points, stream);
    merlin::Vector<double> eval_value = plm_int(points_gpu, stream, 32);
    MESSAGE("Expected value: %f\n", f(points[{0,0}], points[{0,1}], points[{0,2}]));
    MESSAGE("Evaluated value GPU vs CPU: %f %f\n", eval_value[0], plm_int_cpu({2.2, 1.0, 2.0}));
    */
    
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
        merlin::splint::Method::Newton,
        merlin::splint::Method::Newton
    };
    merlin::splint::construct_coeff_cpu(coeff.data(), cart_gr, methods, 1);
    MESSAGE("Reference coefficients: %s\n", coeff.str().c_str());

    // calculate Newton coefficients (GPU)
    merlin::cuda::Stream stream;
    merlin::array::Parcel coeff_gpu(value.shape(), stream);
    coeff_gpu.transfer_data_to_gpu(value, stream);
    merlin::cuda::Memory mem(stream.get_stream_ptr(), cart_gr, methods);
    std::uint64_t shared_mem = cart_gr.sharedmem_size() + methods.sharedmem_size();
    merlin::splint::construct_coeff_gpu(coeff_gpu.data(), mem.get<0>(), mem.get<1>(), 4, shared_mem, &stream);
    // ::cudaDeviceSynchronize();
    MESSAGE("GPU calculated coefficients: %s\n", coeff_gpu.str().c_str());
}
