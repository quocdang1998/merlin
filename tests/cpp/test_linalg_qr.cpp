#include <cstring>

#include "merlin/linalg/matrix.hpp"
#include "merlin/linalg/tri_solve.hpp"
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
    linalg::Matrix mat({10, 10});
    std::memcpy(mat.data(), data, 100 * sizeof(double));
    MESSAGE("Matrix: %s\n", mat.str().c_str());
    DoubleVec vec = {219.7, 216.4, 299.3, 186.8, 254.9, 169.7, 146.5,  94.6,  38. , 10.};
    MESSAGE("Vector: %s\n", vec.str().c_str());
    linalg::triu_one_solve(mat, vec.data());
    // expected solution [1.0, 2.0, 3.0, ..., 9.0, 10.0]
    MESSAGE("Solution: %s\n", vec.str().c_str());
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
    linalg::Matrix mat({10, 10});
    std::memcpy(mat.data(), data, 100 * sizeof(double));
    MESSAGE("Matrix: %s\n", mat.str().c_str());
    DoubleVec vec = {222.8, 224.8, 321.2, 208.4, 277.4, 187.7, 159.1,  95.4,  62.3, 73.};
    MESSAGE("Vector: %s\n", vec.str().c_str());
    linalg::triu_solve(mat, vec.data());
    // expected solution [1.0, 2.0, 3.0, ..., 9.0, 10.0]
    MESSAGE("Solution: %s\n", vec.str().c_str());
}

int main(void) {
    test_triu_solve();
}
