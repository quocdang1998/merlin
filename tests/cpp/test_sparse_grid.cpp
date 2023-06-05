#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "merlin/array/array.hpp"
#include "merlin/array/stock.hpp"
#include "merlin/interpolant/cartesian_grid.hpp"
#include "merlin/interpolant/lagrange.hpp"
#include "merlin/interpolant/newton.hpp"
#include "merlin/interpolant/sparse_grid.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

merlin::array::Array read_data(const std::string & filename) {
    merlin::array::Stock serialized_data(filename);
    merlin::array::Array result(serialized_data.shape());
    result.extract_data_from_file(serialized_data);
    return result;
}

int main(void) {
    merlin::Vector<double> br = {0, 300, 600, 900, 1200};  // l = 2
    merlin::Vector<double> pm = {157};  // l = 0
    merlin::Vector<double> tf = {500, 560, 600, 700, 800, 900, 1000, 1100, 1200};  // l = 3
    merlin::Vector<double> tm = {300, 400, 500, 560, 600};  // l = 2
    merlin::Vector<double> xe = {0.1, 0.4, 0.9};  // l = 1

    merlin::interpolant::SparseGrid grid({br, pm, tf, tm, xe}, 6, {1, 1, 1, 1, 1});
    merlin::interpolant::CartesianGrid validate_grid({br, pm, tf, tm, xe});

    merlin::array::Array raw_data = read_data("/home/catC/dn266595/These/x2vver_data/22UA/stock_data/Pu239_NuFission_FA_micro.txt");
    merlin::array::Array data(raw_data, {{0}, {0}, {0,5}, {}, {}, {0,5}, {0,3}, {}, {0}});
    data.remove_dim(8);
    data.remove_dim(7);
    data.remove_dim(1);
    data.remove_dim(0);

    merlin::array::Array data_newton(merlin::intvec({grid.size()}));
    std::cout << "Before calculation: " << data_newton.str() << "\n";
    merlin::interpolant::calc_newton_coeffs_cpu(grid, data, data_newton);
    std::cout << "After calculation: " << data_newton.str() << "\n";

    // eval interpoaltion time
    std::uint64_t n_point = validate_grid.size();
    merlin::intvec validate_shape = validate_grid.get_grid_shape();
    std::vector<double> error;
    std::vector<std::uint64_t> i_error;
    for (std::uint64_t i_point = 0; i_point < n_point; i_point++) {
        merlin::Vector<double> point = validate_grid[i_point];
        if (grid.contains(point)) {
            continue;
        }
        double interpolated_value = merlin::interpolant::eval_lagrange_cpu(grid, data_newton, point);
        double reference_value = data[merlin::contiguous_to_ndim_idx(i_point, validate_shape)];
        double err = 100.f * std::fabs((interpolated_value / reference_value) - 1.f);
        if (err == 100.f) {
            std::cout << interpolated_value << " " << reference_value << "\n";
        }
        error.push_back(err);
        i_error.push_back(i_point);
    }
}
