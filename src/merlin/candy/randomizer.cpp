// Copyright 2024 quocdang1998
#include "merlin/candy/randomizer.hpp"

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Randomizer
// ---------------------------------------------------------------------------------------------------------------------

// Change mean and standard deviation of the distribution
void candy::set_params(candy::Randomizer & initializer, double mean, double std_dev) {
    switch (initializer.index()) {
        case 0 : {  // gaussian
            std::get<0>(initializer).set_params(mean, std_dev);
            break;
        }
        case 1 : {  // uniform
            std::get<1>(initializer).set_params(mean);
            break;
        }
    }
}

// Sample a value for initializing CP model
double candy::sample(candy::Randomizer & initializer) {
    double result;
    switch (initializer.index()) {
        case 0 : {  // gaussian
            result = std::get<0>(initializer).sample();
            break;
        }
        case 1 : {  // uniform
            result = std::get<1>(initializer).sample();
            break;
        }
    }
    return result;
}

}  // namespace merlin
