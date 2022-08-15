#include "merlin/logger.hpp"

void foo (void) {
    MESSAGE("Status message %d.", 1);
}

int main (void) {
    // note: no newline at the end of the message !
    MESSAGE("Status message %d.", 0);
    WARNING("Warning message %.3f.", 1.0/3.0);
    foo();
    try {
        FAILURE(std::runtime_error, "Fatal error: %s.", "Runtime error");
    } catch (std::exception & e) {}
}
