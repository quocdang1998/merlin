#include "merlin/array/array.hpp"
#include "merlin/splint/cartesian_grid.hpp"
#include "merlin/splint/interpolant.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

#include "merlin/intpl/cartesian_grid.hpp"
#include "merlin/intpl/interpolant.hpp"

#include <cinttypes>

double foo(const merlin::floatvec & v) {
    return (2.f*v[0] + v[2])*v[2] + 3.f*v[1];
}

int main(void) {
    // initialize data and grid
    merlin::splint::CartesianGrid cart_gr({{0.1, 0.2, 0.3}, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.25}});
    merlin::array::Array value(cart_gr.shape());
    for (std::uint64_t i = 0; i < cart_gr.size(); i++) {
        merlin::intvec index(merlin::contiguous_to_ndim_idx(i, cart_gr.shape()));
        value[index] = foo(cart_gr[index]);
    }

    // calculate Lagrange coefficients
    merlin::array::Array coeff(value);
    merlin::Vector<merlin::splint::Method> methods = {
        merlin::splint::Method::Lagrange,
        merlin::splint::Method::Lagrange,
        merlin::splint::Method::Lagrange
    };
    construct_coeff_cpu(coeff.data(), cart_gr, methods, 100);

    // print
    MESSAGE("Value: %f\n", value[{1,2,0}]);
    MESSAGE("Lagrange coefficients: %f\n", coeff[{1,2,0}]);
    merlin::intpl::CartesianGrid grid({{0.1, 0.2, 0.3}, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.25}});
    merlin::intpl::PolynomialInterpolant pl_int(grid, value, merlin::intpl::Method::Lagrange);
    MESSAGE("Reference coefficients: %f\n", pl_int.get_coeff().get({1,2,0}));
}
