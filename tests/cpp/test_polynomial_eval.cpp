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
    /*
    double coeff[18] = {
        0.0, 0.0, 0.0, 2.4, 0.0, 0.0, 0.0, -6.2, 0.0,
        1.3, 0.0, 3.8, 0.0, -3.5, 0.0, 0.0, -1.8, 0.0
    };  // polynomial 1.3x + 2.4y + 3.8 xz^2 - 6.2y^2z - 1.8xy^2z - 3.5xyz
    floatvec coeff_vec;
    coeff_vec.assign(coeff, 18);
    regpl::Polynomial p(coeff_vec, {2, 3, 3});
    MESSAGE("Polynomial: %s\n", p.str().c_str());
    */

    double coeff_simplified[] = {1.3, 2.4, 3.8, -6.2, -1.8, -3.5};
    DoubleVec coeff_vec;
    coeff_vec.assign(coeff_simplified, 6);
    UIntVec coef_idx = {9, 3, 11, 7, 16, 13};
    regpl::Polynomial p(coeff_vec, make_array<std::uint64_t>({2, 3, 3}), coef_idx);
    MESSAGE("Polynomial: %s\n", p.str().c_str());

    Point buffer;
    Point point = make_array<double>({1.8, 0.7, 1.5});
    DoubleVec point_vector;
    point_vector.assign(point.data(), 3);
    MESSAGE("Evaluation at point %s is %f\n", point_vector.str().c_str(), p.eval(point, buffer));
    MESSAGE("Reference  of point %s is %f\n", point_vector.str().c_str(), reference(point.data()));

    p.save("polynom.txt", true);
    regpl::Polynomial p_read;
    p_read.load("polynom.txt");
    MESSAGE("Read polynomial: %s\n", p_read.str().c_str());

}
