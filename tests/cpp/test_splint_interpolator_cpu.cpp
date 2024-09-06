#include <algorithm>
#include <cinttypes>
#include <random>
#include <vector>

#include "merlin/array/array.hpp"
#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/logger.hpp"
#include "merlin/splint/interpolator.hpp"
#include "merlin/synchronizer.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

using namespace merlin;


double foo(const DoubleVec & v) {
    return (2.f*v[0] + v[2])*v[2] + 3.f*v[1];
}

array::Array point_generator(std::uint64_t num_point, const grid::CartesianGrid & grid) {
    std::mt19937 gen;
    std::vector<std::uniform_real_distribution<double>> dists;
    dists.reserve(grid.ndim());
    for (std::uint64_t i_dim = 0; i_dim < grid.ndim(); i_dim++) {
        const DoubleVec grid_vector = grid.grid_vector(i_dim);
        const auto [it_min, it_max] = std::minmax_element(grid_vector.cbegin(), grid_vector.cend());
        dists.push_back(std::uniform_real_distribution<double>(*it_min, *it_max));
    }
    array::Array points({num_point, grid.ndim()});
    for (std::uint64_t i_point = 0; i_point < num_point; i_point++) {
        for (std::uint64_t i_dim = 0; i_dim < grid.ndim(); i_dim++) {
            points[{i_point, i_dim}] = dists[i_dim](gen);
        }
    }
    return points;
}

int main(void) {
    // initialize data and grid
    grid::CartesianGrid cart_gr({{0.1, 0.2, 0.3}, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.25, 0.5}});
    array::Array value(cart_gr.shape());
    for (std::uint64_t i = 0; i < cart_gr.size(); i++) {
        value.set(i, foo(cart_gr[i]));
    }
    Message("Grid: ") << cart_gr.str() << "\n";
    Message("Values: ") << value.str() << "\n";

    // calculate interpolation coefficients
    array::Array coeff(value);
    Vector<splint::Method> methods = {
        splint::Method::Lagrange,
        splint::Method::Newton,
        splint::Method::Newton
    };
    Synchronizer synch_stream(ProcessorType::Cpu);
    splint::Interpolator interp(cart_gr, coeff, methods.data(), synch_stream);
    interp.build_coefficients(10);
    synch_stream.synchronize();
    Message("Interpolation coefficients: ") << interp.get_coeff().str() << "\n";

    // interpolation
    std::uint64_t npoints = 16;
    array::Array points = point_generator(npoints, cart_gr);
    Message out_stream("Function values:");
    for (std::uint64_t i_point = 0; i_point < npoints; i_point++) {
        out_stream << " " << foo(DoubleVec(points.data() + 3 * i_point, 3));
    }
    out_stream << "\n";
    out_stream.emit();
    DoubleVec eval_values(npoints);
    interp.evaluate(points, eval_values, 32);
    synch_stream.synchronize();
    Message("Evaluated values: ") << eval_values.str() << ".\n";
}
