#include "merlin/array/array.hpp"
#include "merlin/logger.hpp"
#include "merlin/statistics/moment.hpp"
#include "merlin/vector.hpp"

int main(void) {
    // initialize array
    double data[6] = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
    merlin::intvec dims({2, 3});
    merlin::intvec strides({dims[1] * sizeof(double), sizeof(double)});
    merlin::array::Array value(data, dims, strides);
    MESSAGE("Data: %s.\n", value.str().c_str());

    std::array<double, 2> moments = merlin::statistics::powered_mean<2>(value, 24);
    MESSAGE("Mean of whole array: %f.\n", moments[0]);
    MESSAGE("Variance of whole array: %f.\n", merlin::statistics::moment_cpu(moments));

    merlin::array::Array mean_dim0 = merlin::statistics::mean_cpu(value, merlin::intvec({0}));
    MESSAGE("Mean of dimension 0: %s.\n", mean_dim0.str().c_str());
    merlin::array::Array mean_dim1 = merlin::statistics::mean_cpu(value, merlin::intvec({1}));
    MESSAGE("Mean of dimension 1: %s.\n", mean_dim1.str().c_str());
}
