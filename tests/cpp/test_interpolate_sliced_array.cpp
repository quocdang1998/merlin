#include "merlin/array/array.hpp"
#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/splint/interpolator.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

// #include "merlin/intpl/cartesian_grid.hpp"
// #include "merlin/intpl/interpolant.hpp"

#include <cinttypes>

double foo(const merlin::floatvec & v) {
    return (2.f*v[0] + v[2])*v[2] + 3.f*v[1];
}

merlin::array::Array point_generator(std::uint64_t num_point, const merlin::grid::CartesianGrid & grid) {
    std::mt19937 gen;
    std::vector<std::uniform_real_distribution<double>> dists;
    dists.reserve(grid.ndim());
    for (std::uint64_t i_dim = 0; i_dim < grid.ndim(); i_dim++) {
        const merlin::floatvec grid_vector = grid.grid_vector(i_dim);
        const auto [it_min, it_max] = std::minmax_element(grid_vector.cbegin(), grid_vector.cend());
        dists.push_back(std::uniform_real_distribution<double>(*it_min, *it_max));
    }
    merlin::array::Array points({num_point, grid.ndim()});
    for (std::uint64_t i_point = 0; i_point < num_point; i_point++) {
        for (std::uint64_t i_dim = 0; i_dim < grid.ndim(); i_dim++) {
            points[{i_point, i_dim}] = dists[i_dim](gen);
        }
    }
    return points;
}

int main(void) {
    // initialize data and grid
    merlin::grid::CartesianGrid cart_gr({{0.1, 0.2, 0.3}, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.25, 0.5}});
    merlin::array::Array value(cart_gr.shape());
    for (std::uint64_t i = 0; i < cart_gr.size(); i++) {
        merlin::intvec index(merlin::contiguous_to_ndim_idx(i, cart_gr.shape()));
        value[index] = foo(cart_gr[index]);
    }

    // calculate Lagrange coefficients
    merlin::array::Array coeff(value);
    merlin::Vector<merlin::splint::Method> methods = {
        merlin::splint::Method::Lagrange,
        merlin::splint::Method::Newton,
        merlin::splint::Method::Newton
    };
    merlin::splint::Interpolator interp(cart_gr, coeff, methods, merlin::ProcessorType::Cpu);
    interp.build_coefficients(4);
    interp.synchronize();
    MESSAGE("Interpolation coefficients: %s\n", interp.get_coeff().str().c_str());

    // interpolation
    merlin::array::Array points = point_generator(1200, cart_gr);
    merlin::floatvec eval_values(1200);
    interp.evaluate(points, eval_values, 24);
    interp.synchronize();
    // interp.synchronize();
    MESSAGE("Evaluated values: %s.\n", eval_values.str().c_str());
    MESSAGE(
        "Function values: %f %f %f.\n",
        foo(merlin::floatvec(&points[0], &points[3])),
        foo(merlin::floatvec(&points[3], &points[6])),
        foo(merlin::floatvec(&points[6], &points[9]))
    );


/*
    merlin::splint::construct_coeff_cpu(coeff.data(), cart_gr, methods, 5);

    // print coefficients
    MESSAGE("Value: %s\n", value.str().c_str());
    MESSAGE("Lagrange coefficients: %s\n", coeff.str().c_str());
    merlin::intpl::CartesianGrid grid({{0.1, 0.2, 0.3}, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.25, 0.5}});
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
*/
}
