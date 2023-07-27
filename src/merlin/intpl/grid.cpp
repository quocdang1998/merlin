// Copyright 2022 quocdang1998
#include "merlin/intpl/grid.hpp"

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Grid
// --------------------------------------------------------------------------------------------------------------------

// Destructor
intpl::Grid::~Grid(void) {
    if (this->points_ != nullptr) {
        delete this->points_;
    }
}

}  // namespace merlin
