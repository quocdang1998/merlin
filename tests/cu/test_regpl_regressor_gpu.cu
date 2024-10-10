
#include <algorithm>
#include <random>

#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/logger.hpp"
#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/regpl/regressor.hpp"
#include "merlin/regpl/polynomial.hpp"
#include "merlin/synchronizer.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

array::Array point_generator(std::uint64_t num_point, const grid::CartesianGrid & grid) {
    std::mt19937 gen;
    std::vector<std::uniform_real_distribution<double>> dists;
    dists.reserve(grid.ndim());
    for (std::uint64_t i_dim = 0; i_dim < grid.ndim(); i_dim++) {
        DoubleView grid_vector = grid.grid_vector(i_dim);
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
    // initialize polynomial
    double coeff_simplified[] = {1.3, 2.4, 3.8, -6.2, -1.8, -3.5};
    UIntVec coef_idx = {9, 3, 11, 7, 16, 13};
    regpl::Polynomial p({2, 3, 3});
    p.set(coeff_simplified, coef_idx);
    Message("Polynomial: ") << p.str() << "\n";

    // intialize regressor on CPU and GPU
    Synchronizer cpu_sync(ProcessorType::Cpu);
    regpl::Regressor cpu_reg(regpl::Polynomial(p), cpu_sync);
    Synchronizer gpu_sync(ProcessorType::Gpu);
    regpl::Regressor gpu_reg(regpl::Polynomial(p), gpu_sync);

    // initialize data
    std::uint64_t npoint = 100;
    grid::CartesianGrid grid({
        {0, 0.5},
        {1, 1.5},
        {3, 6.1},
    });
    array::Array cpu_points = point_generator(npoint, grid);
    array::Parcel gpu_points(cpu_points.shape());
    gpu_points.transfer_data_to_gpu(cpu_points);

    // evaluate
    DoubleVec cpu_result(npoint), gpu_result(npoint);
    cpu_reg.evaluate(cpu_points, cpu_result);
    gpu_reg.evaluate(gpu_points, gpu_result);
    cpu_sync.synchronize();
    gpu_sync.synchronize();

    // print
    Message("CPU result: ") << cpu_result.str() << "\n";
    Message("GPU result: ") << gpu_result.str() << "\n";
}
