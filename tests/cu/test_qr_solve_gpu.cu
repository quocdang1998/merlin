#include <cmath>

#include "merlin/cuda/memory.hpp"
#include "merlin/linalg/qr_solve.hpp"
#include "merlin/linalg/matrix.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

double * cpy_to_gpu(merlin::floatvec & vec) {
    double * result;
    ::cudaMalloc(&result, vec.size() * sizeof(double));
    ::cudaMemcpy(result, vec.data(), vec.size() * sizeof(double), cudaMemcpyHostToDevice);
    return result;
}

// Householder reflection
__global__ void test_householder_reflection(double * matrix_data, double * reflector_data) {
    extern __shared__ char shared_memory[];
    merlin::linalg::Matrix mat(matrix_data, {3, 3}, {3*sizeof(double), sizeof(double)});
    merlin::floatvec reflector;
    reflector.assign(reflector_data, 3);
    auto [buffer_, p_share_ref] = merlin::cuda::copy_objects(shared_memory, reflector);
    // CUDAOUT("Value from shared memory: %d %d %d.\n", p_share_ref[0][0], p_share_ref[0][1], p_share_ref[0][2]);
    merlin::linalg::householder_reflect(mat, *p_share_ref, 0, merlin::flatten_thread_index(), merlin::size_of_block());
}

// QR decomposition
__global__ void test_qr_solve(double * matrix_data, double * vector_data) {
    extern __shared__ char shared_memory[];
    __shared__ double norm;
    merlin::linalg::Matrix mat(matrix_data, {3, 3}, {3*sizeof(double), sizeof(double)});
    merlin::linalg::Matrix x(vector_data, {3, 1}, {sizeof(double), 0});
    double * buffer = reinterpret_cast<double *>(shared_memory);
    merlin::linalg::qr_solve_gpu(mat, x, buffer, norm, merlin::flatten_thread_index(), merlin::size_of_block());
}

int main(void) {
    merlin::floatvec matrix_data = {12, -51, 4, 6, 167, -68, -4, 24, -41};
    merlin::floatvec matrix_data_cpu(matrix_data);
    merlin::linalg::Matrix mat_cpu(matrix_data_cpu.data(), {3, 3}, {3*sizeof(double), sizeof(double)});
    double * matrix_data_gpu = cpy_to_gpu(matrix_data);
    merlin::floatvec x({126, -532, -175}), v({-1.f / std::sqrt(14), 3.f / std::sqrt(14), -2.f / std::sqrt(14)});
    double * x_gpu = cpy_to_gpu(x);
    double * v_gpu = cpy_to_gpu(v);

    // test_householder_reflection<<<1, 12, sizeof(mat_cpu) + v.sharedmem_size()>>>(matrix_data_gpu, v_gpu);
    // merlin::linalg::householder_reflect(mat_cpu, v, 0, 0, 1);

    test_qr_solve<<<1, 3, 3*sizeof(double)>>>(matrix_data_gpu, x_gpu);
    merlin::linalg::Matrix mat_x(x.data(), {3, 1}, {sizeof(double), 0});
    merlin::floatvec buffer(3);
    double norm;
    merlin::linalg::qr_solve_cpu(mat_cpu, mat_x, buffer.data(), norm, 0, 1);

    MESSAGE("After QR solve (CPU): %s\n", matrix_data_cpu.str().c_str());
    MESSAGE("Vector result (CPU): %s\n", x.str().c_str());
    ::cudaMemcpy(matrix_data.data(), matrix_data_gpu, matrix_data.size()*sizeof(double), ::cudaMemcpyDeviceToHost);
    ::cudaMemcpy(x.data(), x_gpu, x.size()*sizeof(double), ::cudaMemcpyDeviceToHost);
    MESSAGE("After QR solve (GPU): %s\n", matrix_data.str().c_str());
    MESSAGE("Vector result (GPU): %s\n", x.str().c_str());
    ::cudaFree(matrix_data_gpu);
    ::cudaFree(x_gpu);
    ::cudaFree(v_gpu);
}
