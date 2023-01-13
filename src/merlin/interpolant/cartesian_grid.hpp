// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_CARTESIAN_GRID_HPP_
#define MERLIN_INTERPOLANT_CARTESIAN_GRID_HPP_

#include <cstdint>  // std::uint64_t
#include <initializer_list>  // std::initializer_list

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/cuda_decorator.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/interpolant/grid.hpp"  //  merlin::interpolant::Grid
#include "merlin/iterator.hpp"  // merlin::Iterator
#include "merlin/vector.hpp"  // merlin::Vector, merlin::intvec, merlin::floatvec

namespace merlin {

/** @brief Multi-dimensional Cartesian grid.*/
class MERLIN_EXPORTS interpolant::CartesianGrid : public interpolant::Grid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    CartesianGrid(void) {}
    /** @brief Constructor from a list of vector of values.*/
    CartesianGrid(std::initializer_list<floatvec> grid_vectors);
    /** @brief Constructor from a vector of values.*/
    CartesianGrid(const Vector<floatvec> & grid_vectors);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    __cuhostdev__ CartesianGrid(const interpolant::CartesianGrid & src) : grid_vectors_(src.grid_vectors_) {}
    /** @brief Copy assignment.*/
    __cuhostdev__ interpolant::CartesianGrid & operator=(const interpolant::CartesianGrid & src) {
        this->grid_vectors_ = src.grid_vectors_;
        return *this;
    }
    /** @brief Move constructor.*/
    __cuhostdev__ CartesianGrid(interpolant::CartesianGrid && src) : grid_vectors_(src.grid_vectors_) {}
    /** @brief Move assignment.*/
    __cuhostdev__ interpolant::CartesianGrid & operator=(interpolant::CartesianGrid && src) {
        this->grid_vectors_ = src.grid_vectors_;
        return *this;
    }
    /// @}

    /// @name Get members and attributes
    /// @{
    /** @brief Get grid vectors.*/
    __cuhostdev__ const Vector<floatvec> & grid_vectors(void) const {return this->grid_vectors_;}
    /** @brief Shape of the grid.*/
    const intvec & grid_shape(void) const {return this->grid_shape_;}
    /** @brief Full tensor of each point in the CartesianGrid in form of 2D table.*/
    array::Array grid_points(void);
    /** @brief Number of dimension of the CartesianGrid.*/
    __cuhostdev__ std::uint64_t ndim(void) const {return this->grid_vectors_.size();}
    /** @brief Number of points in the CartesianGrid.*/
    __cuhostdev__ std::uint64_t size(void) const;
    /// @}

    /// @name Iterator
    /// @{
    using iterator = Iterator;
    /** @brief Begin iterator.*/
    interpolant::CartesianGrid::iterator begin(void);
    /** @brief End iterator.*/
    interpolant::CartesianGrid::iterator end(void);
    /// @}

    /// @name Slicing operator
    /// @{
    /** @brief Get element at a given index.
     *  @param index Index of point in the CartesianGrid::grid_points table.
     */
    floatvec operator[](std::uint64_t index);
    /** @brief Get element at a given index vector.
     *  @param index Vector of index on each dimension.
     */
    floatvec operator[](const intvec & index);
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the grid and its data.*/
    std::uint64_t malloc_size(void) const;
    /** @brief Copy the grid from CPU to a pre-allocated memory on GPU.
     *  @details Values of vectors should be copied to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param grid_vector_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors.
     */
    void copy_to_gpu(interpolant::CartesianGrid * gpu_ptr, void * grid_vector_data_ptr) const;
    #ifdef __NVCC__
    /** @brief Copy meta-data from GPU global memory to shared memory of a kernel.
     *  @note This operation is single-threaded.
     *  @param share_ptr Dynamically allocated shared pointer on GPU.
     *  @param grid_vector_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors.
     */
    __cudevice__ void copy_to_shared_mem(interpolant::CartesianGrid * share_ptr, void * grid_vector_data_ptr);
    #endif  // __NVCC__
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    __cuhostdev__ ~CartesianGrid(void);
    /// @}

  protected:
    /** @brief List of vector of values.*/
    Vector<floatvec> grid_vectors_;
    /** @brief Shape of the grid (number of points on each dimension).*/
    intvec grid_shape_;
    /** @brief Begin iterator.*/
    intvec begin_;
    /** @brief End iterator.*/
    intvec end_;

  private:
    void calc_grid_shape(void);
};

}  // namespace merlin

#endif  // MERLIN_INTERPOLANT_CARTESIAN_GRID_HPP_
