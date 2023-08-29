#include <omp.h>

#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/array/slice.hpp"
#include "merlin/intpl/cartesian_grid.hpp"
#include "merlin/intpl/interpolant.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

double f(double x, double y, double z) {
    return (2.f*x + y*y + x*y) * z;
}

int main(void) {

    merlin::intvec dims = {2, 4, 3};
    merlin::array::Array value(dims);

    merlin::intpl::CartesianGrid grid({{0.0, 2.5}, {0.0, 1.0, 2.0, 3.0}, {-1.0, 0.0, 1.0}});
    MESSAGE("Grid: %s\n", grid.str().c_str());
    for (std::uint64_t i_point = 0; i_point < grid.size(); i_point++) {
        merlin::Vector<double> point = grid[i_point];
        value.set(i_point, f(point[0], point[1], point[2]));
    }
    MESSAGE("Initial array: %s\n", value.str().c_str());
    merlin::array::Array coeff(value.shape());

    merlin::intpl::PolynomialInterpolant pl_int(grid, value, merlin::intpl::Method::Lagrange);
    MESSAGE("Coefficient: %s\n", pl_int.get_coeff().str().c_str());


    merlin::Vector<double> p1({0.0, 2.0, 1.0});  // on grid
    MESSAGE("Evaluated value: %f.\n", pl_int(p1));
    MESSAGE("Expected value: %f.\n", f(p1[0], p1[1], p1[2]));

    merlin::Vector<double> p2({1.0, 1.0, 1.2});  // 2nd dim on grid
    MESSAGE("Evaluated value: %f.\n", pl_int(p2));
    MESSAGE("Expected value: %f.\n", f(p2[0], p2[1], p2[2]));

    merlin::Vector<double> p3({0.5, 0.25, 2.4});  // both dim off grid
    MESSAGE("Evaluated value: %f.\n", pl_int(p3));
    MESSAGE("Expected value: %f.\n", f(p3[0], p3[1], p3[2]));
}
