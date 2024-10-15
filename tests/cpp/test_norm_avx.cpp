#include "merlin/linalg/dot.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

int main(void) {
    // data
    DoubleVec a = {2.1, 1.2, 5.7, 2.8, 6.3, 8.7, 3.3, 5.9, 8.8, 1.5, 3.8, 1.7, 5.5, 6.8, 8.8, 9.4, 6.3, 3.8, 6.4};
    Message("a = {}\n", a.str(", "));
    DoubleVec b = {9.3, 3. , 2.2, 0.7, 6.3, 4.5, 8.1, 5.7, 6.2, 6.8, 4.2, 5. , 9.4, 8. , 9.4, 9.7, 4.5, 3.1, 8.8};
    Message("b = {}\n", b.str(", "));

    // norm
    double norm_a;
    linalg::norm(a.data(), a.size(), norm_a);
    Message("Norm of a: {} (expected: 641.66)\n", norm_a);

    // dot
    double dot_ab;
    linalg::dot(a.data(), b.data(), a.size(), dot_ab);
    Message("Dot product of a and b: {} (expected: 642.50)\n", dot_ab);

    // saxpy
    double c = 1.4;
    linalg::saxpy(c, b.data(), a.data(), a.size());
    Message("a <- 1.4 * b + a: {}\n", a.str(", "));
    DoubleVec saxpy = {
        15.12,  5.4 ,  8.78,  3.78, 15.12, 15.  , 14.64, 13.88, 17.48, 11.02,  9.68,  8.7 , 18.66, 18.  , 21.96, 22.98,
        12.6 ,  8.14, 18.72,
    };
    Message("Expected: {}\n", saxpy.str());
}
