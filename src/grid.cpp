// Copyright 2022 quocdang1998
#include "merlin/grid.hpp"

#include <vector>

namespace merlin {

Grid::Grid(unsigned int ndim, unsigned int npoint) : ndim_(ndim), npoint_(npoint) {
    // calculate capacity
    if (npoint != 0) {
        this->capacity_ = 1;
        while(this->capacity_ < npoint) {
            this->capacity_ <<= 1;
        }
    } else {
        this->capacity_ = 0;
    }

    // allocate data
    this->grid_points_ = Array(std::vector<unsigned int>({this->capacity_, ndim}));
}



std::vector<float> Grid::operator[] (unsigned int index) {
    std::vector<unsigned int> index_grid = {index, 0};
    float* target = &(this->grid_points_[index_grid]);
    return std::vector<float>(target, target + this->ndim_);
}

}  // namespace merlin