#include "merlin/logger.hpp"

#include <iostream>

void foo3 (void) {
    try {
        FAILURE(std::runtime_error, "Foo error.\n");
    } catch (std::exception & e) {};
}

void foo2 (int i = 1) {
    foo3();
}

void foo1 (void) {
    MESSAGE("Print stacktrace from foo1.\n");
    foo2();
}

void foo (void) {
    MESSAGE("Status message %d.\n", 1);
}

int main (void) {
    // note: no newline at the end of the message !
    MESSAGE("Status message %d.\n", 0);
    WARNING("Warning message %.3f.\n", 1.0/3.0);
    foo();
    try {
        FAILURE(std::runtime_error, "Fatal error: %s.\n", "Runtime error");
    } catch (std::exception & e) {}
    try {
        FAILURE(merlin::cuda_compile_error, "CUDA compile error.\n");
    } catch (std::exception & e) {}
    foo1();
}
