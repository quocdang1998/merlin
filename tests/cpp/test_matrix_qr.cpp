#include <cmath>

#include "merlin/linalg/matrix.hpp"
#include "merlin/linalg/qr_solve.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

// Example on https://fr.wikipedia.org/wiki/D%C3%A9composition_QR
void test_householder(void) {
    double matrix_data[9] = {12, -51, 4, 6, 167, -68, -4, 24, -41};
    merlin::linalg::Matrix mat(matrix_data, {3, 3}, {3*sizeof(double), sizeof(double)}, false);
    MESSAGE("Initial matrix: %s\n", mat.str().c_str());

    // test 1st Householder reflection
    merlin::floatvec reflector({-1.f / std::sqrt(14), 3.f / std::sqrt(14), -2.f / std::sqrt(14)});
    merlin::floatvec test_vector({126, -532, -175});  // a dummy vector
    MESSAGE("Apply Householder refection by the vector: %s\n", reflector.str().c_str());
    merlin::linalg::householder_cpu(mat, test_vector, reflector);
    MESSAGE("Matrix after 1st Householder refection: %s\n", mat.str().c_str());
    MESSAGE("Test vector after Householder refection: %s\n", test_vector.str().c_str());

    // test 2nd Householder reflection
    reflector[1] = -4.f/5.f;
    reflector[2] = 3.f/5.f;
    merlin::linalg::householder_cpu(mat, test_vector, reflector, 1);
    MESSAGE("Matrix after 2nd Householder refection: %s\n", mat.str().c_str());
}

void test_qr_solve(void) {
    double matrix_data[9] = {12, -51, 4, 6, 167, -68, -4, 24, -41};
    merlin::linalg::Matrix mat(matrix_data, {3, 3}, {3*sizeof(double), sizeof(double)}, false);
    MESSAGE("Initial matrix: %s\n", mat.str().c_str());
    merlin::floatvec test_vector({126, -532, -175});  // 1st column
    MESSAGE("Vector to solve: %s\n", test_vector.str().c_str());

    merlin::linalg::qr_decomposition_cpu(mat, test_vector);
    MESSAGE("Matrix after QR: %s\n", mat.str().c_str());
    MESSAGE("Vector after QR: %s\n", test_vector.str().c_str());

    merlin::linalg::upright_solver_cpu(mat, test_vector);
    MESSAGE("Matrix after solve: %s\n", mat.str().c_str());
    MESSAGE("Vector after solve: %s\n", test_vector.str().c_str());
}

int main(void) {
    // test_householder();
    test_qr_solve();
}
