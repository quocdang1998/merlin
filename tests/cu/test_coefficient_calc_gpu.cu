#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/interpolant/cartesian_grid.hpp"
#include "merlin/interpolant/lagrange.hpp"
#include "merlin/interpolant/newton.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

int main(void) {
    double data[9] = {1.0, 3.0, 9.0, 2.0, 4.0, 10.0, 2.5, 4.5, 10.5};
    merlin::intvec dims = {3, 3};
    merlin::intvec strides = {dims[1] * sizeof(double), sizeof(double)};
    merlin::array::Array value_cpu(data, dims, strides);
    MESSAGE("Initial array: %s\n", value_cpu.str().c_str());

    merlin::interpolant::CartesianGrid grid({{0.0, 1.0, 1.5}, {0.0, 1.0, 2.0}});

    merlin::cuda::Stream stream(merlin::cuda::Stream::Setting::Default);
    merlin::array::Parcel value(value_cpu.shape());
    value.transfer_data_to_gpu(value_cpu, stream);
    merlin::array::Parcel coeff(value_cpu.shape());
    stream.synchronize();

    // merlin::interpolant::calc_lagrange_coeffs_gpu(grid, value, coeff, stream);
    merlin::interpolant::calc_newton_coeffs_gpu(grid, value, coeff, stream);
    stream.synchronize();
    MESSAGE("Result GPU: %s\n", coeff.str().c_str());

    merlin::array::Array coeff_cpu(value_cpu.shape());
    // merlin::interpolant::calc_lagrange_coeffs_cpu(grid, value_cpu, coeff_cpu);
    merlin::interpolant::calc_newton_coeffs_cpu(grid, value_cpu, coeff_cpu);
    MESSAGE("Result CPU: %s\n", coeff_cpu.str().c_str());

}
