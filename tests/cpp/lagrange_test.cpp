#include "merlin/interpolant.hpp"
#include "merlin/grid.hpp"
#include "merlin/tensor.hpp"

int main (void) {
    merlin::CartesianGrid grid({{0.0, 1.0}, {0.0, 1.0}});
    float A[4] = {1,2,3,4};
    unsigned int dims[2] = {2, 2};
    unsigned int strides[2] = {2*sizeof(float), sizeof(float)};
    merlin::Array value(A, 2, dims, strides, false);

    merlin::LagrangeInterpolant lgr(grid, value);
}