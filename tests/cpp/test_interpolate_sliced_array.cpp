#include "merlin/array/array.hpp"
#include "merlin/array/slice.hpp"
#include "merlin/interpolant/lagrange.hpp"
#include "merlin/interpolant/cartesian_grid.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

#include <cinttypes>

int main(void) {
    // whole array
    double data[6] = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
    merlin::intvec dims({2, 3});
    merlin::intvec strides({dims[1] * sizeof(double), sizeof(double)});
    merlin::array::Array value(data, dims, strides);
    // sub-array 1
    merlin::Vector<merlin::array::Slice> slice_1({{0}, {}});
    merlin::array::Array value_1(value, slice_1);
    MESSAGE("Shape of sub-array 1 : %" PRIu64 ", %" PRIu64 ".\n", value_1.shape()[0], value_1.shape()[1]);
    MESSAGE("Element of sub-array 1 : ");
    for (std::uint64_t i = 0; i < value_1.size(); i++) {
        std::printf("%.1f ", value_1.get(i));
    }
    std::printf("\n");
    // sub-array 2
    merlin::Vector<merlin::array::Slice> slice_2({{1}, {}});
    merlin::array::Array value_2(value, slice_2);
    MESSAGE("Shape of sub-array 2 : %" PRIu64 ", %" PRIu64 ".\n", value_2.shape()[0], value_2.shape()[1]);
    MESSAGE("Element of sub-array 2 : ");
    for (std::uint64_t i = 0; i < value_2.size(); i++) {
        std::printf("%.1f ", value_2.get(i));
    }
    std::printf("\n");
/*
    // interpolation
    merlin::interpolant::CartesianGrid grid({{0.0, 1.0}, {0.0, 1.0, 2.0}});
    merlin::array::Array coeff(value.shape());
    // coeff 1
    merlin::array::Array coeff_1(coeff, slice_1);
    merlin::interpolant::calc_lagrange_coeffs_cpu(grid, value_1, slice_1, coeff_1);
    MESSAGE("Element of coeff 1 : ");
    for (std::uint64_t i = 0; i < coeff_1.size(); i++) {
        std::printf("%.1f ", coeff_1.get(i));
    }
    std::printf("\n");
    MESSAGE("Theorical calculated values (C-contiguous order): -1/2 3 -5/2.\n");
    // coeff 2
    merlin::array::Array coeff_2(coeff, slice_2);
    merlin::interpolant::calc_lagrange_coeffs_cpu(grid, value_2, slice_2, coeff_2);
    MESSAGE("Element of coeff 2 : ");
    for (std::uint64_t i = 0; i < coeff_2.size(); i++) {
        std::printf("%.1f ", coeff_2.get(i));
    }
    std::printf("\n");
    MESSAGE("Theorical calculated values (C-contiguous order): 1 -4 3.\n");
*/
}
