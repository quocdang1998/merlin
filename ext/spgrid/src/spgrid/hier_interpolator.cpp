// Copyright 2022 quocdang1998
#include "spgrid/hier_interpolator.hpp"

#include <cstring>  // std::memset
#include <sstream>  // std::ostringstream

#include "merlin/array/array.hpp"   // merlin::array::Array
#include "merlin/logger.hpp"        // Fatal
#include "merlin/splint/tools.hpp"  // merlin::splint::construct_coeff_cpu, merlin::splint::eval_intpl_cpu
#include "merlin/utils.hpp"         // merlin::ptr_to_subsequence

#include "spgrid/sparse_grid.hpp"  // spgrid::SparseGrid
#include "spgrid/utils.hpp"        // spgrid::get_grid, spgrid::get_array

namespace spgrid {

// ---------------------------------------------------------------------------------------------------------------------
// Interpolator
// ---------------------------------------------------------------------------------------------------------------------

// Construct from a hierarchical grid and a full Cartesian data
HierInterpolator::HierInterpolator(const SparseGrid & grid, const mln::array::Array & full_data,
                                   const mln::splint::Method * p_method, mln::Synchronizer & synchronizer) :
p_synch_(&synchronizer) {
    // allocate memory
    this->intpl.reserve(grid.nlevel());
    // loop over each level and construct the interpolator
    for (LevelIterator it = grid.begin(); it != grid.end(); ++it) {
        // std::cout << "Level i: " << it.level << "\n";
        // get current grid and data
        mln::grid::CartesianGrid level_grid = get_grid(grid.fullgrid(), it.cum_idx);
        mln::array::Array level_data = get_data(full_data, it.cum_idx);
        // subtract data by the evaluation of previous grids
        mln::array::Array level_grid_points = level_grid.get_points();
        mln::DoubleVec level_evaluation(level_data.size());
        for (LevelIterator jt = grid.begin(); jt != it; ++jt) {
            // std::cout << "    Level j: " << jt.level << "\n";
            this->intpl[jt.level].evaluate(level_grid_points, level_evaluation, 1);
            synchronizer.synchronize();
            for (std::uint64_t i = 0; i < level_evaluation.size(); i++) {
                level_data[i] -= level_evaluation[i];
            }
        }
        // add interpolator
        this->intpl.emplace_back(level_grid, level_data, p_method, synchronizer);
        this->intpl.back().build_coefficients();
        synchronizer.synchronize();
        // std::cout << "    Coeff: " << this->intpl.back().get_coeff().str() << "\n";
    }
}

// Evaluate interpolation
void HierInterpolator::evaluate(const mln::array::Array & points, mln::DoubleVec & result) {
    // check if initialized
    if ((this->intpl.size() == 0) || (this->p_synch_ == nullptr)) {
        mln::Fatal<std::runtime_error>("Interpolator not initialized.\n");
    }
    // initialize
    std::memset(result.data(), 0, result.size() * sizeof(double));
    mln::DoubleVec level_result(result.size());
    // loop on each level
    for (mln::splint::Interpolator & interpolator : this->intpl) {
        // evaluate by current grid
        interpolator.evaluate(points, level_result, 1);
        this->p_synch_->synchronize();
        // add up the rest
        for (std::uint64_t i = 0; i < result.size(); i++) {
            result[i] += level_result[i];
        }
    }
}

// String representation
std::string HierInterpolator::str(void) const {
    std::ostringstream os;
    os << "<HierInterpolator(";
    for (const mln::splint::Interpolator & interp : this->intpl) {
        os << interp.str();
    }
    os << ")>";
    return os.str();
}

}  // namespace spgrid
