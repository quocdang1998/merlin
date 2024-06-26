#include "merlin/env.hpp"
#include "merlin/logger.hpp"
#include "merlin/regpl/polynomial.hpp"
#include "merlin/vector.hpp"

double reference(double * point) {
    double result = 1.3*point[0] + 2.4*point[1] + 3.8*point[0]*point[2]*point[2];
    result += (-6.2)*point[1]*point[1]*point[2] + (-1.8)*point[0]*point[1]*point[1]*point[2] + (-3.5)*point[0]*point[1]*point[2];
    return result;
}

using namespace merlin;

int main(void) {

    Environment::init_cuda(0);

    double coeff_simplified[] = {1.3, 2.4, 3.8, -6.2, -1.8, -3.5};
    UIntVec coef_idx = {9, 3, 11, 7, 16, 13};
    regpl::Polynomial p(make_array<std::uint64_t>({2, 3, 3}));
    p.set(coeff_simplified, coef_idx);
    Message("Polynomial: %s\n", p.str().c_str());

    Point buffer;
    Point point = make_array<double>({1.8, 0.7, 1.5});
    DoubleVec point_vector;
    point_vector.assign(point.data(), 3);
    Message("Evaluation at point %s is %f\n", point_vector.str().c_str(), p.eval(point, buffer));
    Message("Reference  of point %s is %f\n", point_vector.str().c_str(), reference(point.data()));

    p.save("polynom.txt", true);
    regpl::Polynomial p_read;
    p_read.load("polynom.txt");
    Message("Read polynomial: %s\n", p_read.str().c_str());
}
