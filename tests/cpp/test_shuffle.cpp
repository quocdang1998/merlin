#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/shuffle.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

int main(void) {
    merlin::intvec shape = {7, 8, 3, 5};
    merlin::Shuffle::set_random_seed(5);
    merlin::Shuffle sffle(shape);
    merlin::Shuffle inv_sffle = sffle.inverse();

    MESSAGE("Shuffle object: %s\n", sffle.str().c_str());
    MESSAGE("Inverse object: %s\n", inv_sffle.str().c_str());
    std::printf("\n");

    double A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    merlin::intvec dims({2, 5});
    merlin::intvec strides = merlin::array::contiguous_strides(dims, sizeof(double));
    merlin::array::Array data(A, dims, strides, false);
    MESSAGE("Initial array: %s\n", data.str().c_str());

    merlin::Shuffle sffle_array(data.shape());
    MESSAGE("Shuffle object: %s\n", sffle_array.str().c_str());
    merlin::array::Array shuffled_data = merlin::array::shuffle_array(data, sffle_array);
    MESSAGE("Array after shuffled: %s\n", shuffled_data.str().c_str());
    merlin::array::Array inverted_data = merlin::array::shuffle_array(shuffled_data, sffle_array.inverse());
    MESSAGE("Array after inverse shuffle: %s\n", inverted_data.str().c_str());
}
