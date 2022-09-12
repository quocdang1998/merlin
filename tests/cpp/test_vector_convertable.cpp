#include "merlin/vector.hpp"
#include "merlin/logger.hpp"

int main(void) {
    float src[3] = {1.2, 2.3, 5.6};
    merlin::intvec v(src, 3);

    MESSAGE("Original array    : %f %f %f.\n", src[0], src[1], src[2]);
    MESSAGE("Constructed vector: %lu %lu %lu.\n", v[0], v[1], v[2]);
}
