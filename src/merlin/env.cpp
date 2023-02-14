// Copyright quocdang1998
#include "merlin/env.hpp"

#include <iostream>

namespace merlin {

// Default constructor
Environment::Environment(void) {}

void Environment::set_inited(bool value) {
    Environment::is_inited = value;
}

void Environment::print_inited(void) {
    std::cout << "Inited variable is " << ((Environment::is_inited) ? "true" : "false") << ".\n";
}

// Check if environment is initialized
bool Environment::is_inited = false;

}  // namespace merlin


