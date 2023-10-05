// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_CARTESIAN_GRID_HPP_
#define MERLIN_SPLINT_CARTESIAN_GRID_HPP_

#include <cstdint>           // std::uint64_t
#include <initializer_list>  // std::initializer_list
#include <string>            // std::string

#include "merlin/array/slice.hpp"         // merlin::slicevec
#include "merlin/cuda_interface.hpp"      // __cuhostdev__
#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/splint/declaration.hpp"  // merlin::splint::CartesianGrid
#include "merlin/vector.hpp"              // merlin::floatvec, merlin::intvec, merlin::Vector

namespace merlin {

/** @brief Multi-dimensional Cartesian grid.
 *  @details The i-th coordinate of each point in the grid is an element derived from the i-th vector containing real
 *  values (each element within this vector is called grid node). A Cartesian grid is formed by taking the Cartesian
 *  product over a set of vectors of nodes, representing the set of all possible distinct points that can constructed
 *  from the set of vectors.
 */
class splint::CartesianGrid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    CartesianGrid(void) = default;
    /** @brief Constructor from list of initializer lists.*/
    MERLIN_EXPORTS CartesianGrid(const Vector<floatvec> & grid_vectors);
    /** @brief Constructor from list of nodes and shape (meant to be used in the Python interface).*/
    MERLIN_EXPORTS CartesianGrid(floatvec && grid_nodes, intvec && shape);
    /** @brief Constructor as a sub-grid from a larger grid.*/
    MERLIN_EXPORTS CartesianGrid(const splint::CartesianGrid & whole, const slicevec & slices);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    MERLIN_EXPORTS CartesianGrid(const splint::CartesianGrid & src);
    /** @brief Copy assignment.*/
    MERLIN_EXPORTS splint::CartesianGrid & operator=(const splint::CartesianGrid & src);
    /** @brief Move constructor.*/
    CartesianGrid(splint::CartesianGrid && src) = default;
    /** @brief Move assignment.*/
    splint::CartesianGrid & operator=(splint::CartesianGrid && src) = default;
    /// @}

    /// @name Get members and attributes
    /// @{
    /** @brief Get grid vector of a given dimension.*/
    __cuhostdev__ const floatvec grid_vector(std::uint64_t i_dim) const noexcept {
        floatvec grid_vector;
        grid_vector.assign(const_cast<double *>(this->node_per_dim_ptr_[i_dim]), this->grid_shape_[i_dim]);
        return grid_vector;
    }
    /** @brief Get dimensions of the grid.*/
    __cuhostdev__ constexpr std::uint64_t ndim(void) const noexcept { return this->grid_shape_.size(); }
    /** @brief Get shape of the grid.*/
    __cuhostdev__ constexpr const intvec & shape(void) const noexcept { return this->grid_shape_; }
    /** @brief Get total number of points in the grid.*/
    __cuhostdev__ std::uint64_t size(void) const;
    /** @brief Get total number of nodes on all dimension.*/
    __cuhostdev__ std::uint64_t num_nodes(void) const noexcept { return this->grid_nodes_.size(); }
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    MERLIN_EXPORTS ~CartesianGrid(void);
    /// @}

  protected:
    /** @brief Vector of contiguous grid nodes per dimension.*/
    floatvec grid_nodes_;
    /** @brief Shape of the grid.*/
    intvec grid_shape_;

  private:
    /** @brief Pointer to first node in each dimension.*/
    Vector<double *> node_per_dim_ptr_;
};

}  // namespace merlin

#endif  // MERLIN_SPLINT_CARTESIAN_GRID_HPP_
