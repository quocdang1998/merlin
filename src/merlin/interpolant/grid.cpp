// Copyright 2022 quocdang1998
#include "merlin/interpolant/grid.hpp"

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Grid
// --------------------------------------------------------------------------------------------------------------------

// Destructor
interpolant::Grid::~Grid(void) {
    if (this->points_ != nullptr) {
        delete this->points_;
    }
}

}  // namespace merlin
