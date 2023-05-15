// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_CARTESIAN_GRID_HPP_
#define MERLIN_INTERPOLANT_CARTESIAN_GRID_HPP_

#include <cstdint>  // std::uint64_t
#include <initializer_list>  // std::initializer_list
#include <string>  // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::Array
#include "merlin/cuda_decorator.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/interpolant/grid.hpp"  //  merlin::interpolant::Grid
#include "merlin/iterator.hpp"  // merlin::Iterator
#include "merlin/vector.hpp"  // merlin::Vector, merlin::intvec

namespace merlin {

/** @brief Multi-dimensional Cartesian grid.*/
class interpolant::CartesianGrid : public interpolant::Grid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    CartesianGrid(void) = default;
    /** @brief Constructor from a vector of values.*/
    MERLIN_EXPORTS CartesianGrid(const Vector<Vector<double>> & grid_vectors);
    /** @brief Constructor from an r-value reference to a vector of values.*/
    MERLIN_EXPORTS CartesianGrid(Vector<Vector<double>> && grid_vectors);
    /** @brief Get a subgrid from original grid.*/
    MERLIN_EXPORTS CartesianGrid(const interpolant::CartesianGrid & whole, const Vector<array::Slice> & slices);
    /** @brief Constructor from the number of dimension.*/
    CartesianGrid(std::uint64_t ndim) : grid_vectors_(ndim) {}
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
    /** @brief Get reference to grid vectors.*/
    __cuhostdev__ constexpr Vector<Vector<double>> & grid_vectors(void) noexcept {return this->grid_vectors_;}
    /** @brief Get constant reference to grid vectors.*/
    __cuhostdev__ constexpr const Vector<Vector<double>> & grid_vectors(void) const noexcept {
        return this->grid_vectors_;
    }
    /** @brief Get shape of the grid.*/
    __cuhostdev__ intvec get_grid_shape(std::uint64_t * data_ptr = nullptr) const noexcept;
    /** @brief Full tensor of each point in the CartesianGrid in form of 2D table.*/
    MERLIN_EXPORTS array::Array grid_points(void) const;
    /** @brief Number of dimension of the CartesianGrid.*/
    __cuhostdev__ std::uint64_t ndim(void) const {return this->grid_vectors_.size();}
    /** @brief Number of points in the CartesianGrid.*/
    __cuhostdev__ std::uint64_t size(void) const;
    /// @}

    /// @name Iterator
    /// @{
    using iterator = Iterator;
    /** @brief Begin iterator.*/
    MERLIN_EXPORTS interpolant::CartesianGrid::iterator begin(void);
    /** @brief End iterator.*/
    MERLIN_EXPORTS interpolant::CartesianGrid::iterator end(void);
    /// @}

    /// @name Slicing operator
    /// @{
    /** @brief Get element at a given index.
     *  @param index Index of point in the CartesianGrid::grid_points table.
     */
    __cuhostdev__ Vector<double> operator[](std::uint64_t index) const noexcept;
    /** @brief Get element at a given index vector.
     *  @param index Vector of index on each dimension.
     */
    __cuhostdev__ Vector<double> operator[](const intvec & index) const noexcept;
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the grid and its data.*/
    MERLIN_EXPORTS std::uint64_t malloc_size(void) const;
    /** @brief Copy the grid from CPU to a pre-allocated memory on GPU.
     *  @details Values of vectors should be copied to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param grid_vector_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors.
     *  @param stream_ptr Pointer to CUDA stream for asynchronous copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(interpolant::CartesianGrid * gpu_ptr, void * grid_vector_data_ptr,
                                      std::uintptr_t stream_ptr = 0) const;
    #ifdef __NVCC__
    /** @brief Copy meta-data from GPU global memory to shared memory of a kernel.
     *  @note This operation is single-threaded.
     *  @param share_ptr Dynamically allocated shared pointer on GPU.
     *  @param grid_vector_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors.
     */
    __cudevice__ void * copy_to_shared_mem(interpolant::CartesianGrid * share_ptr, void * grid_vector_data_ptr) const;
    /** @brief Copy meta-data from GPU global memory to shared memory of a kernel.
     *  @param share_ptr Dynamically allocated shared pointer on GPU.
     *  @param grid_vector_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors.
     */
    __cudevice__ void * copy_to_shared_mem_single(interpolant::CartesianGrid * share_ptr,
                                                  void * grid_vector_data_ptr) const;
    #endif  // __NVCC__
    /// @}

    /// @name Grid merge operator
    /// @{
    /** @brief Union assignment of 2 Cartesian grids.*/
    MERLIN_EXPORTS interpolant::CartesianGrid & operator+=(const interpolant::CartesianGrid & grid);
    MERLIN_EXPORTS friend interpolant::CartesianGrid operator+(const interpolant::CartesianGrid & grid_1,
                                                               const interpolant::CartesianGrid & grid_2);
    MERLIN_EXPORTS friend double exclusion_grid(const interpolant::CartesianGrid & grid_parent,
                                                const interpolant::CartesianGrid & grid_child,
                                                const Vector<double> & x);
    /// @}

    /// @name Query
    /// @{
    /** @brief Check if point in the grid.*/
    MERLIN_EXPORTS bool contains(const Vector<double> & point) const;
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
    /** @brief List of vector of values.*/
    Vector<Vector<double>> grid_vectors_;
    /** @brief Begin iterator.*/
    interpolant::CartesianGrid::iterator begin_;
    /** @brief End iterator.*/
    interpolant::CartesianGrid::iterator end_;
};

namespace interpolant {

/** @brief Union of 2 Cartesian grids.*/
MERLIN_EXPORTS interpolant::CartesianGrid operator+(const interpolant::CartesianGrid & grid_1,
                                                    const interpolant::CartesianGrid & grid_2);

/** @brief Exclusion on each dimension of 2 Cartesian grids.*/
MERLIN_EXPORTS double exclusion_grid(const interpolant::CartesianGrid & grid_parent,
                                     const interpolant::CartesianGrid & grid_child, const Vector<double> & x);

}  // namespace interpolant

}  // namespace merlin

#endif  // MERLIN_INTERPOLANT_CARTESIAN_GRID_HPP_
