#include "merlin/regpl/polynomial.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

double reference(double * point) {
    double result = 1.3*point[0] + 2.4*point[1] + 3.8*point[0]*point[2]*point[2];
    result += (-6.2)*point[1]*point[1]*point[2] + (-1.8)*point[0]*point[1]*point[1]*point[2] + (-3.5)*point[0]*point[1]*point[2];
    return result;
}

int main(void) {
    double coeff[18] = {
        0.0, 0.0, 0.0, 2.4, 0.0, 0.0, 0.0, -6.2, 0.0,
        1.3, 0.0, 3.8, 0.0, -3.5, 0.0, 0.0, -1.8, 0.0
    };  // polynomial 1.3x + 2.4y + 3.8 xz^2 - 6.2y^2z - 1.8xy^2z - 3.5xyz
    regpl::Polynomial p(coeff, {2, 3, 3});
    MESSAGE("Polynomial: %s\n", p.str().c_str());

    floatvec buffer(3);
    floatvec point = {1.2, 0.4, 1.1};
    MESSAGE("Evaluation at point %s is %f\n", point.str().c_str(), p.eval(point.data(), buffer.data()));
    MESSAGE("Reference  of point %s is %f\n", point.str().c_str(), reference(point.data()));

}
