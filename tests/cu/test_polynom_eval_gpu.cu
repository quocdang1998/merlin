#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/cuda_interface.hpp"
#include "merlin/logger.hpp"
#include "merlin/regpl/regressor.hpp"
#include "merlin/regpl/polynomial.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

int main(void) {
    // initialize polynomial
    double coeff_simplified[6] = {1.3, 2.4, 3.8, -6.2, -1.8, -3.5};
    floatvec coeff_vec;
    coeff_vec.assign(coeff_simplified, 6);
    intvec coef_idx = {
        1, 0, 0,
        0, 1, 0,
        1, 0, 2,
        0, 2, 1,
        1, 2, 1,
        1, 1, 1,
    };
    regpl::Polynomial p(coeff_vec, {2, 3, 3}, coef_idx);
    MESSAGE("Polynomial: %s\n", p.str().c_str());

    // point
    double point_data[] = {1.8, 0.7, 1.5, 4.1, 2.1, 5.2};
    intvec shape = {2, 3};
    array::Array point_cpu(point_data, shape, array::contiguous_strides(shape, sizeof(double)), false);
    array::Parcel point_gpu(shape);
    point_gpu.transfer_data_to_gpu(point_cpu);

    // allocate data for result and cache mem
    floatvec result_cpu(2);
    floatvec cpu_cache_mem(3);
    floatvec result_gpu(2);
    double * result_gpu_ptr = reinterpret_cast<double *>(cuda_mem_alloc(2 * sizeof(double)));

    // calculate CPU result
    result_cpu[0] = p.eval(point_data, cpu_cache_mem.data());
    result_cpu[1] = p.eval(point_data + 3, cpu_cache_mem.data());

    // calculate GPU result
    cuda::Memory mem(0, p);
    regpl::eval_by_gpu(mem.get<0>(), point_gpu.data(), result_gpu_ptr, 2, 3,
                       p.sharedmem_size(), 1, cuda::Stream());
    cuda_mem_cpy_device_to_host(result_gpu.data(), result_gpu_ptr, 2 * sizeof(double));

    // print
    MESSAGE("CPU result: %s\n", result_cpu.str().c_str());
    MESSAGE("GPU result: %s\n", result_gpu.str().c_str());

    // free GPU data
    cuda_mem_free(result_gpu_ptr);
}
