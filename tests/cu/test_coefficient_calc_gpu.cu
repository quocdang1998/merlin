#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/env.hpp"
#include "merlin/interpolant/cartesian_grid.hpp"
#include "merlin/interpolant/lagrange.hpp"
#include "merlin/interpolant/newton.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

double f(double x, double y, double z) {
    return (2.f*x + y*y + x*y) * z;
}

int main(void) {
    merlin::intvec dims = {2, 3, 3};
    merlin::array::Array value_cpu(dims);

    merlin::interpolant::CartesianGrid grid({{0, 4.8}, {0.0, 1.0, 1.5}, {0.0, 1.0, 2.0}});
    for (std::uint64_t i_point = 0; i_point < grid.size(); i_point++) {
        merlin::Vector<double> point = grid[i_point];
        value_cpu.set(i_point, f(point[0], point[1], point[2]));
    }
    MESSAGE("Initial array: %s\n", value_cpu.str().c_str());

    merlin::cuda::Stream stream(merlin::cuda::Stream::Setting::Default);
    merlin::array::Parcel coeff(value_cpu.shape());
    coeff.transfer_data_to_gpu(value_cpu, stream);
    merlin::interpolant::calc_lagrange_coeffs_gpu(grid, coeff, coeff, stream);
    // merlin::interpolant::calc_newton_coeffs_gpu(grid, coeff, coeff, stream, 32);
    stream.synchronize();
    merlin::Environment::flush_cuda_deferred_deallocation();
    MESSAGE("Result GPU: %s\n", coeff.str().c_str());

    merlin::array::Array coeff_cpu(value_cpu.shape());
    merlin::interpolant::calc_lagrange_coeffs_cpu(grid, value_cpu, coeff_cpu);
    // merlin::interpolant::calc_newton_coeffs_cpu(grid, value_cpu, coeff_cpu);
    MESSAGE("Result CPU: %s\n", coeff_cpu.str().c_str());

    merlin::array::Array points(merlin::intvec({1,value_cpu.ndim()}));
    points[{0,0}] = 2.2; points[{0,1}] = 1.4; points[{0,2}] = 0.25;
    merlin::array::Parcel points_gpu(points.shape());
    points_gpu.transfer_data_to_gpu(points, stream);
    merlin::Vector<double> eval_value = merlin::interpolant::eval_lagrange_gpu(grid, coeff, points_gpu, stream, 32);
    // merlin::Vector<double> eval_value = merlin::interpolant::eval_newton_gpu(grid, coeff, points_gpu, stream, 32);
    merlin::Environment::flush_cuda_deferred_deallocation();
    MESSAGE("Expected value: %f\n", f(points[{0,0}], points[{0,1}], points[{0,2}]));
    MESSAGE("Evaluated value GPU vs CPU: %f %f\n", eval_value[0], merlin::interpolant::eval_lagrange_cpu(grid, coeff_cpu, {2.2, 1.4, 0.25}));
    // MESSAGE("Evaluated value GPU vs CPU: %f %f\n", eval_value[0], merlin::interpolant::eval_newton_cpu(grid, coeff_cpu, {2.2, 1.4, 0.25}));
}
