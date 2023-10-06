#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/env.hpp"
#include "merlin/intpl/cartesian_grid.hpp"
#include "merlin/intpl/interpolant.hpp"
#include "merlin/intpl/lagrange.hpp"
#include "merlin/intpl/newton.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

double f(double x, double y, double z) {
    return (2.f*x + y*y + x*y) * z;
}

int main(void) {
    merlin::intvec dims = {2, 3, 3};
    merlin::array::Array value_cpu(dims);

    merlin::intpl::CartesianGrid grid({{0, 4.8}, {0.0, 1.0, 1.5}, {0.0, 1.0, 2.0}});
    for (std::uint64_t i_point = 0; i_point < grid.size(); i_point++) {
        merlin::Vector<double> point = grid[i_point];
        value_cpu.set(i_point, f(point[0], point[1], point[2]));
    }
    MESSAGE("Initial array: %s\n", value_cpu.str().c_str());

    merlin::cuda::Stream stream(merlin::cuda::StreamSetting::Default);
    merlin::array::Parcel coeff(value_cpu.shape());
    coeff.transfer_data_to_gpu(value_cpu, stream);
    merlin::intpl::PolynomialInterpolant plm_int(grid, coeff, merlin::intpl::Method::Lagrange, stream, 32);
    stream.synchronize();
    MESSAGE("Result GPU: %s\n", plm_int.get_coeff().str().c_str());
    merlin::intpl::PolynomialInterpolant plm_int_cpu(grid, value_cpu, merlin::intpl::Method::Lagrange);
    MESSAGE("Result CPU: %s\n", plm_int_cpu.get_coeff().str().c_str());

    merlin::array::Array points(merlin::intvec({1, value_cpu.ndim()}));
    points[{0,0}] = 2.2; points[{0,1}] = 1.0; points[{0,2}] = 2.0;
    merlin::array::Parcel points_gpu(points.shape());
    points_gpu.transfer_data_to_gpu(points, stream);
    merlin::Vector<double> eval_value = plm_int(points_gpu, stream, 32);
    MESSAGE("Expected value: %f\n", f(points[{0,0}], points[{0,1}], points[{0,2}]));
    MESSAGE("Evaluated value GPU vs CPU: %f %f\n", eval_value[0], plm_int_cpu({2.2, 1.0, 2.0}));
}
