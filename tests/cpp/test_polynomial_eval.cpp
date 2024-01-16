#include "merlin/regpl/polynomial.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

double reference(double * point) {
    return 1 + 2*point[1] + 3*point[1]*point[1] + 5*point[0]*point[1] + 7*point[0]*point[1]*point[1];
}

int main(void) {
    double coeff[6] = {1, 2, 3, 0, 5, 7};  // polynomial 1 + 2y + 3y^2 + 0x + 5xy + 7xy^2
    regpl::Polynomial p(coeff, {2, 3});
    MESSAGE("Polynomial: %s\n", p.str().c_str());

    floatvec buffer(2);
    floatvec point = {1.2, 0.4};
    MESSAGE("Evaluation at point %s is %f\n", point.str().c_str(), p.eval(point.data(), buffer.data()));
    MESSAGE("Reference  of point %s is %f\n", point.str().c_str(), reference(point.data()));


}
