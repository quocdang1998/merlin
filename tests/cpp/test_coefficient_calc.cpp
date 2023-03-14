#include <omp.h>

#include "merlin/array/array.hpp"
#include "merlin/array/copy.hpp"
#include "merlin/array/slice.hpp"
#include "merlin/interpolant/cartesian_grid.hpp"
#include "merlin/interpolant/lagrange.hpp"
#include "merlin/interpolant/newton.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

double f(double x, double y, double z) {
    return (2.f*x + y*y + x*y) * z;
}

int main(void) {
    merlin::intvec dims = {2, 4, 3};
    merlin::array::Array value(dims);

    merlin::interpolant::CartesianGrid grid({{0.0, 2.5}, {0.0, 1.0, 2.0, 3.0}, {-1.0, 0.0, 1.0}});
    MESSAGE("Grid: %s\n", grid.str().c_str());
    for (std::uint64_t i_point = 0; i_point < grid.size(); i_point++) {
        merlin::Vector<double> point = grid[i_point];
        value.set(i_point, f(point[0], point[1], point[2]));
    }
    MESSAGE("Initial array: %s\n", value.str().c_str());
    merlin::array::Array coeff(value.shape());

    // merlin::interpolant::calc_lagrange_coeffs_cpu(grid, value, coeff);
    merlin::interpolant::calc_newton_coeffs_cpu(grid, value, coeff);
    MESSAGE("Coefficient: %s\n", coeff.str().c_str());

    merlin::Vector<double> p1({0.0, 2.0, 1.0});  // on grid
    // double p1_eval = merlin::interpolant::eval_lagrange_cpu(grid, coeff, p1);
    double p1_eval = merlin::interpolant::eval_newton_cpu2(grid, coeff, p1);
    MESSAGE("Evaluated value: %f.\n", p1_eval);
    MESSAGE("Expected value: %f.\n", f(p1[0], p1[1], p1[2]));
    merlin::Vector<double> p2({1.0, 1.0, 1.2});  // 2nd dim on grid
    // double p2_eval = merlin::interpolant::eval_lagrange_cpu(grid, coeff, p2);
    double p2_eval = merlin::interpolant::eval_newton_cpu(grid, coeff, p2);
    MESSAGE("Evaluated value: %f.\n", p2_eval);
    MESSAGE("Expected value: %f.\n", f(p2[0], p2[1], p2[2]));
    merlin::Vector<double> p3({0.5, 0.25, 2.4});  // both dim off grid
    // double p3_eval = merlin::interpolant::eval_lagrange_cpu(grid, coeff, p3);
    double p3_eval = merlin::interpolant::eval_newton_cpu(grid, coeff, p3);
    MESSAGE("Evaluated value: %f.\n", p3_eval);
    MESSAGE("Expected value: %f.\n", f(p3[0], p3[1], p3[2]));

}
