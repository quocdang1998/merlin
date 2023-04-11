// Copyright 2022 quocdang1998
#include "merlin/interpolant/sparse_grid.hpp"

#include <algorithm>  // std::stable_sort
#include <cinttypes>  // PRIu64
#include <cstdint>  // std::uint64_t
#include <cstring>  // std::memcpy
#include <numeric>  // std::iota
#include <sstream>  // std::ostringstream
#include <utility>  // std::move
#include <vector>  // std::vector

#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx, merlin::calc_subgrid_index

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Utils
// --------------------------------------------------------------------------------------------------------------------

// The size of grid vector is valid
static inline bool is_valid_size(std::uint64_t size) {
    return ((size != 2) && ((size-1) & (size-2)) == 0);
}

// Check the validity of a set of grid vector
static inline void check_validity(const Vector<Vector<double>> & grid_vectors) {
    const Vector<double> * data = grid_vectors.cbegin();
    for (std::uint64_t i = 0; i < grid_vectors.size(); i++) {
        // check size
        if (!is_valid_size(data[i].size())) {
            FAILURE(std::invalid_argument, "Size of each grid vector must be 2^n + 1, dimension %" PRIu64 " got %"
                    PRIu64 ".\n", i, data[i].size());
        }
    }
}

// Get the level from a given valid size of a 1D grid
static inline std::uint64_t get_level_from_valid_size(std::uint64_t size) noexcept {
    size -= 1;
    std::uint64_t level = 0;
    while (size >>= 1) {
        ++level;
    }
    return level;
}

// Get index of nodes in a given level of a 1D grid
static intvec hiearchical_index(std::uint64_t level, std::uint64_t size) {
    // check level validity
    std::uint64_t max_level = get_level_from_valid_size(size);
    if (level > max_level) {
        CUHDERR(std::invalid_argument, "Expected level less than %" PRIu64 ", got %" PRIu64 ".\n", max_level, level);
    }
    // trivial cases
    if (level == 0) {
        return intvec{(size - 1) / 2};
    } else if (level == 1) {
        return intvec{0, (size - 1)};
    }
    // normal cases: calculate the jump and loop over each odd number
    std::uint64_t jump = 1 << (max_level - level);
    intvec indices(1 << (level - 1));
    for (std::uint64_t i_node = 0; i_node < indices.size(); i_node++) {
        indices[i_node] = jump * (2*i_node + 1);
    }
    return indices;
}

// --------------------------------------------------------------------------------------------------------------------
// SparseGrid
// --------------------------------------------------------------------------------------------------------------------

// Constructor isotropic grid from grid vectors
interpolant::SparseGrid::SparseGrid(std::initializer_list<Vector<double>> grid_vectors) : grid_vectors_(grid_vectors) {
    // check validity
    check_validity(this->grid_vectors_);
    // calculate number of level in full grid
    intvec maxlevel = this->max_levels();
    std::uint64_t num_subgrid = 1;
    for (std::uint64_t i_dim = 0; i_dim < maxlevel.size(); i_dim++) {
        num_subgrid *= ++maxlevel[i_dim];
    }
    // initialize level index vector
    this->level_index_ = intvec(num_subgrid*this->ndim(), 0);
    this->sub_grid_start_index_ = intvec(num_subgrid+1, 0);
    for (std::uint64_t i_subgrid = 0; i_subgrid < num_subgrid; i_subgrid++) {
        intvec subgrid_index = contiguous_to_ndim_idx(i_subgrid, maxlevel);
        intvec dummy_level = this->level_index(i_subgrid);
        for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
            dummy_level[i_dim] = subgrid_index[i_dim];
        }
        std::uint64_t subgrid_size = calc_subgrid_size(subgrid_index);
        this->sub_grid_start_index_[i_subgrid+1] = this->sub_grid_start_index_[i_subgrid] + subgrid_size;
    }
}

// Constructor anisotropic grid from grid vectors
interpolant::SparseGrid::SparseGrid(std::initializer_list<Vector<double>> grid_vectors,
                                    std::uint64_t max, const intvec & weight) : grid_vectors_(grid_vectors) {
    // check validity
    check_validity(this->grid_vectors_);
    // calculate number of level in full grid
    intvec maxlevel = this->max_levels();
    std::uint64_t num_subgrid_in_fullgrid = 1;
    for (std::uint64_t i_dim = 0; i_dim < maxlevel.size(); i_dim++) {
        num_subgrid_in_fullgrid *= ++maxlevel[i_dim];
    }
    // loop over each subgrid index and see if it is in the sparse grid
    std::vector<intvec> level_vector_storage;
    std::vector<std::uint64_t> alpha_level;
    for (std::uint64_t i_subgrid = 0; i_subgrid < num_subgrid_in_fullgrid; i_subgrid++) {
        intvec subgrid_index = contiguous_to_ndim_idx(i_subgrid, maxlevel);
        std::uint64_t alpha_l = inner_prod(subgrid_index, weight);
        if (alpha_l > max) {
            continue;  // exclude if alpha_l > max
        }
        alpha_level.push_back(alpha_l);
        level_vector_storage.push_back(std::move(subgrid_index));
    }
    // sort according to its alpha_l
    intvec sorted_index(level_vector_storage.size());
    std::iota(sorted_index.begin(), sorted_index.end(), 0);
    auto sort_policy = [&alpha_level] (const std::uint64_t & i1, const std::uint64_t & i2) {
        return alpha_level[i1] < alpha_level[i2];
    };
    std::stable_sort(sorted_index.begin(), sorted_index.end(), sort_policy);
    // copying value to level_vectors and calculate start index
    this->level_index_ = intvec(level_vector_storage.size()*this->ndim(), 0);
    this->sub_grid_start_index_ = intvec(level_vector_storage.size()+1, 0);
    for (std::uint64_t i = 0; i < sorted_index.size(); i++) {
        const std::uint64_t & i_subgrid = sorted_index[i];
        intvec dummy_level = this->level_index(i);
        std::memcpy(dummy_level.data(), level_vector_storage[i_subgrid].data(), this->ndim()*sizeof(std::uint64_t));
        std::uint64_t subgrid_size = calc_subgrid_size(dummy_level);
        this->sub_grid_start_index_[i+1] = this->sub_grid_start_index_[i] + subgrid_size;
    }
}

// Construct sparse grid from vector of components and level index
interpolant::SparseGrid::SparseGrid(std::initializer_list<Vector<double>> grid_vectors,
                                    const Vector<intvec> & level_vectors) : grid_vectors_(grid_vectors) {
    // check validity
    check_validity(this->grid_vectors_);
    intvec max_level = this->max_levels();
    for (const intvec & level_vector : level_vectors) {
        if (level_vector.size() != this->ndim()) {
            FAILURE(std::invalid_argument, "Expected level index vector of size %" PRIu64 ", got size %" PRIu64 ".\n",
                    this->ndim(), level_vector.size());
        }
        for (std::uint64_t i_dim = 0; i_dim < level_vectors.size(); i_dim++) {
            if (level_vector[i_dim] > max_level[i_dim]) {
                FAILURE(std::invalid_argument, "Index of level vector cannot be bigger than max level.\n");
            }
        }
    }
    // copy level vectors
    this->level_index_ = intvec(level_vectors.size()*this->ndim(), 0);
    this->sub_grid_start_index_ = intvec(level_vectors.size()+1, 0);
    for (std::uint64_t i_level = 0; i_level < level_vectors.size(); i_level++) {
        intvec dummy_level = this->level_index(i_level);
        for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
            dummy_level[i_dim] = level_vectors[i_level][i_dim];
        }
        std::uint64_t subgrid_size = calc_subgrid_size(dummy_level);
        this->sub_grid_start_index_[i_level+1] = this->sub_grid_start_index_[i_level] + subgrid_size;
    }
}

// Copy constructor
interpolant::SparseGrid::SparseGrid(const interpolant::SparseGrid & src) :
grid_vectors_(src.grid_vectors_), level_index_(src.level_index_), sub_grid_start_index_(src.sub_grid_start_index_) {}

// Copy assignment
interpolant::SparseGrid & interpolant::SparseGrid::operator=(const interpolant::SparseGrid & src) {
    this->grid_vectors_ = src.grid_vectors_;
    this->level_index_ = src.level_index_;
    this->sub_grid_start_index_ = src.sub_grid_start_index_;
    return *this;
}

// Move constructor
interpolant::SparseGrid::SparseGrid(interpolant::SparseGrid && src) :
grid_vectors_(src.grid_vectors_), level_index_(src.level_index_), sub_grid_start_index_(src.sub_grid_start_index_) {}

// Move assignment
interpolant::SparseGrid & interpolant::SparseGrid::operator=(interpolant::SparseGrid && src) {
    this->grid_vectors_ = src.grid_vectors_;
    this->level_index_ = src.level_index_;
    this->sub_grid_start_index_ = src.sub_grid_start_index_;
    return *this;
}

// Get shape of the grid
intvec interpolant::SparseGrid::get_grid_shape(void) const noexcept {
    intvec grid_shape(this->ndim());
    for (std::uint64_t i_dim = 0; i_dim < grid_shape.size(); i_dim++) {
        grid_shape[i_dim] = this->grid_vectors_[i_dim].size();
    }
    return grid_shape;
}

// --------------------------------------------------------------------------------------------------------------------
// SparseGrid levels
// --------------------------------------------------------------------------------------------------------------------

// Max level per dimension
intvec interpolant::SparseGrid::max_levels(void) const {
    intvec result(this->ndim(), 0);
    for (std::uint64_t i = 0; i < result.size(); i++) {
        result[i] = get_level_from_valid_size(this->grid_vectors_[i].size());
    }
    return result;
}

// Number of points in grid
std::uint64_t interpolant::SparseGrid::size(void) const {
    std::uint64_t num_level = this->num_level();
    std::uint64_t result = 0;
    for (std::uint64_t i_level = 0; i_level < num_level; i_level++) {
        intvec dummy_level;
        dummy_level.assign(const_cast<std::uint64_t *>(&(this->level_index_[i_level*this->ndim()])), this->ndim());
        std::uint64_t subgrid_size = calc_subgrid_size(dummy_level);
        result += subgrid_size;
    }
    return result;
}

// Index of point in grid given its contiguous order
intvec interpolant::SparseGrid::index_from_contiguous(std::uint64_t contiguous_index) const {
    // determine i_level
    std::uint64_t i_level;
    for (i_level = 0; i_level < this->num_level(); i_level++) {
        if (contiguous_index < this->sub_grid_start_index_[i_level+1]) {
            break;
        }
    }
    std::uint64_t contiguous_index_in_level = contiguous_index - this->sub_grid_start_index_[i_level];
    // calculate level shape
    intvec level_vector;
    level_vector.assign(const_cast<std::uint64_t *>(&(this->level_index_[i_level*this->ndim()])), this->ndim());
    intvec level_shape = get_level_shape(level_vector);
    // get index wrt full grid
    intvec ndim_index_in_level = contiguous_to_ndim_idx(contiguous_index_in_level, level_shape);
    intvec result(this->ndim());
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        intvec full_grid_index = hiearchical_index(level_vector[i_dim], this->grid_vectors_[i_dim].size());
        result[i_dim] = full_grid_index[ndim_index_in_level[i_dim]];
    }
    return result;
}

// Point at a give multi-dimensional index.
Vector<double> interpolant::SparseGrid::point_at_index(const intvec & index) const {
    if (index.size() != this->ndim()) {
        CUHDERR(std::invalid_argument, "Index must have the same ndim as the grid.\n");
    }
    Vector<double> result(this->ndim());
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        result[i_dim] = this->grid_vectors_[i_dim][index[i_dim]];
    }
    return result;
}

bool interpolant::SparseGrid::contains(const Vector<double> & point) const {
    if (point.size() != this->ndim()) {
        return false;
    }
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        const Vector<double> & grid_vector = this->grid_vectors_[i_dim];
        if (std::find(grid_vector.cbegin(), grid_vector.cend(), point[i_dim]) == grid_vector.cend()) {
            return false;
        }
    }
    std::uint64_t num_level = this->num_level();
    for (std::uint64_t i_level = 0; i_level < num_level; i_level++) {
        interpolant::CartesianGrid cart_grid_level(interpolant::get_cartesian_grid(*this, i_level));
        if (cart_grid_level.contains(point)) {
            return true;
        }
    }
    return false;
}

// Increase number of grid values on a given dimension.
void interpolant::SparseGrid::add_points_to_grid(const Vector<double> & new_points, std::uint64_t dimension) {
    // check for coherent number of points
    Vector<double> & current_grid_points = this->grid_vectors_[dimension];
    if (current_grid_points.size() == 1 && new_points.size() != 2) {
        FAILURE(std::invalid_argument, "Expect number of added points to be 2, got " PRIu64 ".\n", new_points.size());
    } else if (new_points.size() != current_grid_points.size()-1) {
        FAILURE(std::invalid_argument, "Invalid number of added points.\n");
    }
    // copy old array to new array
    Vector<double> new_grid_points(new_points.size() + current_grid_points.size());
    if (current_grid_points.size() == 1) {
        new_grid_points[1] = current_grid_points[0];
        new_grid_points[0] = new_points[0];
        new_grid_points[2] = new_points[1];
    } else {
        for (std::uint64_t i_point = 0; i_point < current_grid_points.size(); i_point++) {
            new_grid_points[2*i_point] = current_grid_points[i_point];
        }
        for (std::uint64_t i_point = 0; i_point < new_points.size(); i_point++) {
            new_grid_points[2*i_point+1] = new_points[i_point];
        }
    }
    this->grid_vectors_[dimension] = std::move(new_grid_points);
}

// Representation
std::string interpolant::SparseGrid::str(void) const {
    std::ostringstream os;
    os << "Sparse grid:\n";
    os << "  Grid vectors:\n";
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        os << "    Dimension " << i_dim << ": " << this->grid_vectors_[i_dim].str() << "\n";
    }
    os << "  Registered level vectors:\n";
    std::uint64_t num_level = this->num_level();
    for (std::uint64_t i_level = 0; i_level < num_level; i_level++) {
        intvec level_index;
        level_index.assign(const_cast<std::uint64_t *>(&(this->level_index_[i_level*this->ndim()])), this->ndim());
        os << "    Level vector " << i_level << ": " << level_index.str() << "\n";
    }
    os << "  Starting index of subgrid: " << this->sub_grid_start_index_.str();
    return os.str();
}

// Destructor
interpolant::SparseGrid::~SparseGrid(void) {}

// --------------------------------------------------------------------------------------------------------------------
// SparseGrid utils
// --------------------------------------------------------------------------------------------------------------------

// Retrieve values of points in the sparse grid from a full Cartesian array of value
void interpolant::copy_value_from_cartesian_array(array::NdData & dest, const array::NdData & src,
                                                  const interpolant::SparseGrid & grid) {
    // check size
    if (grid.ndim() != src.ndim()) {
        FAILURE(std::invalid_argument, "Expected grid and value array have equal number of dimension.\n");
    }
    std::uint64_t ndim = grid.ndim();
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        if (grid.grid_vectors()[i_dim].size() != src.shape()[i_dim]) {
            FAILURE(std::invalid_argument, "The grid and the value array should have the same shape.\n");
        }
    }
    if (dest.ndim() != 1) {
        FAILURE(std::invalid_argument, "Expected ndim of destination array is 1.\n");
    }
    std::uint64_t grid_size = grid.size();
    if (dest.shape()[0] != grid_size) {
        FAILURE(std::invalid_argument, "Expected length of destination array is %" PRIu64 ".\n", grid_size);
    }
    // get element by index
    for (std::uint64_t i_point = 0; i_point < grid_size; i_point++) {
        intvec index_in_full_grid = grid.index_from_contiguous(i_point);
        dest.set(intvec({i_point}), src.get(index_in_full_grid));
    }
}

// Get Cartesian Grid corresponding to a given level vector
interpolant::CartesianGrid interpolant::get_cartesian_grid(const SparseGrid & grid, std::uint64_t subgrid_index) {
    // check for valid level index
    std::uint64_t num_level = grid.num_level();
    if (subgrid_index > num_level) {
        FAILURE(std::invalid_argument, "Expected subgrid index less than %" PRIu64 ", got %" PRIu64 ".\n",
                num_level, subgrid_index);
    }
    // initialize cartesian grid vector
    Vector<Vector<double>> cart_grid_vectors(grid.ndim());
    const intvec level_vector = grid.level_index(subgrid_index);
    for (std::uint64_t i = 0; i < grid.ndim(); i++) {
        intvec dim_index = hiearchical_index(level_vector[i], grid.grid_vectors()[i].size());
        Vector<double> points(dim_index.size());
        for (std::uint64_t j = 0; j < points.size(); j++) {
            points[j] = grid.grid_vectors()[i][dim_index[j]];
        }
        cart_grid_vectors[i] = points;
    }
    return interpolant::CartesianGrid(std::move(cart_grid_vectors));
}

}  // namespace merlin
