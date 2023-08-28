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
__global__ void test_householder_reflection(double * matrix_data, double * vector_data, double * reflector_data) {
    extern __shared__ char shared_memory[];
    merlin::linalg::Matrix mat(matrix_data, {3, 3}, {3*sizeof(double), sizeof(double)}, false);
    merlin::floatvec x, v;
    x.assign(vector_data, 3);
    v.assign(reflector_data, 3);
    auto [buffer_, p_share_x, p_share_v] = merlin::cuda::copy_class_to_shared_mem(shared_memory, x, v);
    double * buffer = reinterpret_cast<double *>(buffer_);
    merlin::linalg::householder_gpu(mat, *p_share_x, *p_share_v, 0, buffer, merlin::flatten_thread_index(), merlin::size_of_block());
    if (merlin::flatten_thread_index() == 0) {
        x[0] = (*p_share_x)[0];
        x[1] = (*p_share_x)[1];
        x[2] = (*p_share_x)[2];
    }
    __syncthreads();
}

// QR decomposition
__global__ void test_qr_solve(double * matrix_data, double * vector_data) {
    extern __shared__ char shared_memory[];
    merlin::linalg::Matrix mat(matrix_data, {3, 3}, {3*sizeof(double), sizeof(double)}, false);
    merlin::floatvec x;
    x.assign(vector_data, 3);
    auto [buffer_, p_share_x] = merlin::cuda::copy_class_to_shared_mem(shared_memory, x);
    double * buffer = reinterpret_cast<double *>(buffer_);
    merlin::linalg::qr_decomposition_gpu(mat, *p_share_x, buffer, merlin::flatten_thread_index(), merlin::size_of_block());
    merlin::linalg::upright_solver_gpu(mat, *p_share_x, merlin::flatten_thread_index(), merlin::size_of_block());
    if (merlin::flatten_thread_index() == 0) {
        x[0] = (*p_share_x)[0];
        x[1] = (*p_share_x)[1];
        x[2] = (*p_share_x)[2];
    }
    __syncthreads();
}


int main(void) {
    merlin::floatvec matrix_data = {12, -51, 4, 6, 167, -68, -4, 24, -41};
    merlin::floatvec matrix_data_cpu(matrix_data);
    merlin::linalg::Matrix mat_cpu(matrix_data_cpu.data(), {3, 3}, {3*sizeof(double), sizeof(double)}, false);
    double * matrix_data_gpu = cpy_to_gpu(matrix_data);
    merlin::floatvec x({126, -532, -175}), v({-1.f / std::sqrt(14), 3.f / std::sqrt(14), -2.f / std::sqrt(14)});
    double * x_gpu = cpy_to_gpu(x);
    double * v_gpu = cpy_to_gpu(v);

    // test_householder_reflection<<<1, 1, sizeof(double)*3 + x.shared_mem_size() + v.shared_mem_size()>>>(matrix_data_gpu, x_gpu, v_gpu);
    // merlin::linalg::householder_cpu(mat_cpu, x, v);

    test_qr_solve<<<1, 3, sizeof(double)*(3+3) + x.sharedmem_size()>>>(matrix_data_gpu, x_gpu);
    merlin::linalg::qr_decomposition_cpu(mat_cpu, x);
    merlin::linalg::upright_solver_cpu(mat_cpu, x);

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
