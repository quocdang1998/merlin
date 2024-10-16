#include "merlin/linalg/vector.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"


using namespace merlin;

int main(void) {
    linalg::Vector v1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    Message("Aligned vector is:") << v1.str() << "\n";
}
