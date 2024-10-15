#include <cstring>

#include "merlin/linalg/matrix.hpp"
#include "merlin/linalg/tri_solve.hpp"
#include "merlin/linalg/qrp_decomp.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

void test_triu_one_solve(void) {
    // solve a triangular system
    double data[100] = {
        1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        2.5, 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        3.6, 9.5, 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        4.7, 3.2, 7.1, 1. , 0. , 0. , 0. , 0. , 0. , 0. ,
        1.3, 1.6, 5.3, 4.8, 1. , 0. , 0. , 0. , 0. , 0. ,
        7.4, 2.6, 6.2, 9.5, 1.2, 1. , 0. , 0. , 0. , 0. ,
        1.8, 7.4, 1.9, 3.6, 5.4, 3.6, 1. , 0. , 0. , 0. ,
        9.1, 1.8, 2.5, 4.9, 6.5, 1.5, 2.1, 1. , 0. , 0. ,
        0.2, 3.7, 8.1, 2.6, 7.1, 7.5, 4.3, 5.4, 1. , 0. ,
        4.6, 5. , 9.8, 1.4, 8.9, 5.9, 8.4, 3.8, 2.9, 1. ,
    };
    linalg::Matrix mat(10, 10);
    std::memcpy(mat.data(), data, 100 * sizeof(double));
    Message("Matrix: ") << mat.str() << "\n";
    DoubleVec vec = {219.7, 216.4, 299.3, 186.8, 254.9, 169.7, 146.5,  94.6,  38. , 10.};
    Message("Vector: ") << vec.str() << "\n";
    linalg::triu_one_solve(mat, vec.data());
    // expected solution [1.0, 2.0, 3.0, ..., 9.0, 10.0]
    Message("Solution: ") << vec.str() << "\n";
}

void test_triu_solve(void) {
    // solve a triangular system
    double data[100] = {
        4.1, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        2.5, 5.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        3.6, 9.5, 8.3, 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        4.7, 3.2, 7.1, 6.4, 0. , 0. , 0. , 0. , 0. , 0. ,
        1.3, 1.6, 5.3, 4.8, 5.5, 0. , 0. , 0. , 0. , 0. ,
        7.4, 2.6, 6.2, 9.5, 1.2, 4. , 0. , 0. , 0. , 0. ,
        1.8, 7.4, 1.9, 3.6, 5.4, 3.6, 2.8, 0. , 0. , 0. ,
        9.1, 1.8, 2.5, 4.9, 6.5, 1.5, 2.1, 1.1, 0. , 0. ,
        0.2, 3.7, 8.1, 2.6, 7.1, 7.5, 4.3, 5.4, 3.7, 0. ,
        4.6, 5. , 9.8, 1.4, 8.9, 5.9, 8.4, 3.8, 2.9, 7.3
    };
    linalg::Matrix mat(10, 10);
    std::memcpy(mat.data(), data, 100 * sizeof(double));
    Message("Matrix: ") << mat.str() << "\n";
    DoubleVec vec = {222.8, 224.8, 321.2, 208.4, 277.4, 187.7, 159.1,  95.4,  62.3, 73.};
    Message("Vector: ") << vec.str() << "\n";
    linalg::triu_solve(mat, vec.data());
    // expected solution [1.0, 2.0, 3.0, ..., 9.0, 10.0]
    Message("Solution: ") << vec.str() << "\n";
}

void test_qr_solve(void) {
    // data (20x4)
    double data[80] = {
        4.4, 6.7, 3.6, 8.8, 3.9, 8.1, 7.2, 6.9, 8.2, 2.9, 3.9, 5.7, 2.3, 2.8, 3.6, 1.7, 5.8, 4.1, 4.6, 1.4,
        4.7, 0.9, 8.7, 1.2, 8.7, 3.7, 0.9, 7.9, 9.9, 1.9, 3.2, 3.2, 3.5, 3.4, 5.3, 7.9, 3.1, 5.7, 8.2, 9.9,
        6.4, 8.3, 7. , 5.8, 4.6, 2.5, 2. , 4.7, 8.8, 1.9, 6.5, 3.1, 7.5, 0. , 0.5, 0.4, 0.1, 3.5, 9.1, 5.3,
        6.7, 2.1, 8.8, 6.5, 8.8, 7.7, 8. , 6.4, 4.9, 1.4, 0.9, 7.4, 5.5, 0. , 3.8, 4.2, 6.5, 1.1, 0. , 1.2,
    };
    linalg::QRPDecomp qrp(20, 4);
    std::memcpy(qrp.core().data(), data, 80 * sizeof(double));
    Message("Matrix: ") << qrp.core().str() << "\n";
    qrp.decompose();
    double problem[20] = {
        59.8, 41.8, 77.2, 54.6, 70.3, 53.8, 47. , 62.4, 74. , 18. , 33.4, 51. ,
        53.8,  9.6, 30.9, 35.5, 38.3, 30.4, 48.3, 41.9,
    };
    double residual = qrp.solve(problem);
    DoubleVec solution(problem, 4, true);
    Message("Solution: ") << solution.str() << "\n";
    Message("Residual: ") << residual << "\n";
}

void test_permutation(void) {
    Permutation p({1, 5, 2, 4, 3, 0});
    Permutation p_1 = p.inv();
    Message("Inverse: {}\n", p_1.str());
    double a[6] = {1.8, 2.1, 3.2, 4.5, 5.7, 6.2};
    DoubleVec vec_b(6);
    p.inv_permute(a, vec_b.data());
    Message("Vec_b: {}\n", vec_b.str());
    p.inplace_inv_permute(a);
    DoubleVec vec_a(a, 6, true);
    Message("Vec_a: {}\n", vec_a.str());
}

int main(void) {
    test_qr_solve();
    // test_permutation();
}
