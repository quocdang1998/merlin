#include "merlin/vector.hpp"
#include "merlin/logger.hpp"

#include <cinttypes>

int main(void) {
    float src[3] = {1.2, 2.3, 5.6};
    merlin::intvec v(src, 3);
    std::string s[2] = {"abc", "defg"};
    // merlin::intvec v2(s, s+1);  // error

    MESSAGE("Original array    : %f %f %f.\n", src[0], src[1], src[2]);
    MESSAGE("Constructed vector: %" PRIu64 " %" PRIu64 " %" PRIu64 ".\n", v[0], v[1], v[2]);
}
