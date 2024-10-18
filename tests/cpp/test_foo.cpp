#include "merlin/linalg/aligned_vector.hpp"
#include "merlin/linalg/level1.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

int main(void) {
    linalg::AlignedVector v1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
    Message("Aligned vector is:") << v1.str() << "\n";

    linalg::subtract_vectors(2.5, v1.data(), v1.data(), v1.capacity());
    Message("Multiplied by 2.5 vector is:") << v1.str() << "\n";
}
