#include "merlin/array/array.hpp"
#include "merlin/splint/cartesian_grid.hpp"
#include "merlin/splint/tools.hpp"
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
    construct_coeff_cpu(coeff.data(), cart_gr, methods, 10);

    // print coefficients
    MESSAGE("Value: %s\n", value.str().c_str());
    MESSAGE("Lagrange coefficients: %s\n", coeff.str().c_str());
    merlin::intpl::CartesianGrid grid({{0.1, 0.2, 0.3}, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.25}});
    merlin::intpl::PolynomialInterpolant pl_int(grid, value, merlin::intpl::Method::Lagrange);
    MESSAGE("Reference coefficients: %s\n", pl_int.get_coeff().str().c_str());

    // interpolation
    merlin::Vector<double> p1({0.0, 2.0, 1.0});
    double result_p1;
    merlin::splint::eval_intpl_cpu(coeff.data(), cart_gr, methods, p1.data(), 1, &result_p1, 1);
    MESSAGE("Evaluated value: %f.\n", result_p1);
    MESSAGE("Evaluated value: %f.\n", pl_int(p1));
    MESSAGE("Reference value: %f.\n", foo(p1));

    merlin::Vector<double> p2({1.0, 1.0, 1.2});  // 2nd dim on grid
    double result_p2;
    merlin::splint::eval_intpl_cpu(coeff.data(), cart_gr, methods, p2.data(), 1, &result_p2, 1);
    MESSAGE("Evaluated value: %f.\n", result_p2);
    MESSAGE("Evaluated value: %f.\n", pl_int(p2));
    MESSAGE("Reference value: %f.\n", foo(p2));

    merlin::Vector<double> p3({0.5, 0.25, 2.4});  // both dim off grid
    double result_p3;
    merlin::splint::eval_intpl_cpu(coeff.data(), cart_gr, methods, p3.data(), 1, &result_p3, 1);
    MESSAGE("Evaluated value: %f.\n", result_p3);
    MESSAGE("Evaluated value: %f.\n", pl_int(p3));
    MESSAGE("Reference value: %f.\n", foo(p3));

}
