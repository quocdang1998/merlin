// Copyright 2022 quocdang1998
#ifndef MERLIN_INTPL_CARTESIAN_GRID_HPP_
#define MERLIN_INTPL_CARTESIAN_GRID_HPP_

#include <cstdint>           // std::uint64_t
#include <initializer_list>  // std::initializer_list
#include <string>            // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::Array
#include "merlin/cuda_interface.hpp"     // __cuhostdev__
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/intpl/grid.hpp"         //  merlin::intpl::Grid
#include "merlin/iterator.hpp"           // merlin::Iterator
#include "merlin/vector.hpp"             // merlin::Vector, merlin::intvec, merlin::floatvec

namespace merlin {

/** @brief Multi-dimensional Cartesian grid.*/
class intpl::CartesianGrid : public intpl::Grid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    CartesianGrid(void) = default;
    /** @brief Constructor from a vector of values.*/
    MERLIN_EXPORTS CartesianGrid(const Vector<floatvec> & grid_vectors);
    /** @brief Constructor from an r-value reference to a vector of values.*/
    MERLIN_EXPORTS CartesianGrid(Vector<floatvec> && grid_vectors);
    /** @brief Get a subgrid from original grid.*/
    MERLIN_EXPORTS CartesianGrid(const intpl::CartesianGrid & whole, const Vector<array::Slice> & slices);
    /** @brief Constructor from the number of dimension.*/
    CartesianGrid(std::uint64_t ndim) : grid_vectors_(ndim) {}
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    __cuhostdev__ CartesianGrid(const intpl::CartesianGrid & src) : grid_vectors_(src.grid_vectors_) {}
    /** @brief Copy assignment.*/
    __cuhostdev__ intpl::CartesianGrid & operator=(const intpl::CartesianGrid & src) {
        this->grid_vectors_ = src.grid_vectors_;
        return *this;
    }
    /** @brief Move constructor.*/
    __cuhostdev__ CartesianGrid(intpl::CartesianGrid && src) : grid_vectors_(src.grid_vectors_) {}
    /** @brief Move assignment.*/
    __cuhostdev__ intpl::CartesianGrid & operator=(intpl::CartesianGrid && src) {
        this->grid_vectors_ = src.grid_vectors_;
        return *this;
    }
    /// @}

    /// @name Get members and attributes
    /// @{
    /** @brief Get reference to grid vectors.*/
    __cuhostdev__ constexpr Vector<floatvec> & grid_vectors(void) noexcept { return this->grid_vectors_; }
    /** @brief Get constant reference to grid vectors.*/
    __cuhostdev__ constexpr const Vector<floatvec> & grid_vectors(void) const noexcept { return this->grid_vectors_; }
    /** @brief Get shape of the grid.*/
    __cuhostdev__ intvec get_grid_shape(std::uint64_t * data_ptr = nullptr) const noexcept;
    /** @brief Full tensor of each point in the CartesianGrid in form of 2D table.*/
    MERLIN_EXPORTS array::Array grid_points(void) const;
    /** @brief Number of dimension of the CartesianGrid.*/
    __cuhostdev__ std::uint64_t ndim(void) const { return this->grid_vectors_.size(); }
    /** @brief Number of points in the CartesianGrid.*/
    __cuhostdev__ std::uint64_t size(void) const;
    /// @}

    /// @name Iterator
    /// @{
    using iterator = Iterator;
    /** @brief Begin iterator.*/
    MERLIN_EXPORTS intpl::CartesianGrid::iterator begin(void);
    /** @brief End iterator.*/
    MERLIN_EXPORTS intpl::CartesianGrid::iterator end(void);
    /// @}

    /// @name Slicing operator
    /// @{
    /** @brief Get element at a given index.
     *  @param index Index of point in the CartesianGrid::grid_points table.
     */
    __cuhostdev__ floatvec operator[](std::uint64_t index) const noexcept;
    /** @brief Get element at a given index vector.
     *  @param index Vector of index on each dimension.
     */
    __cuhostdev__ floatvec operator[](const intvec & index) const noexcept;
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the grid and its data.*/
    MERLIN_EXPORTS std::uint64_t cumalloc_size(void) const noexcept;
    /** @brief Copy the grid from CPU to a pre-allocated memory on GPU.
     *  @details Values of vectors should be copied to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param grid_vector_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors.
     *  @param stream_ptr Pointer to CUDA stream for asynchronous copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(intpl::CartesianGrid * gpu_ptr, void * grid_vector_data_ptr,
                                      std::uintptr_t stream_ptr = 0) const;
    /** @brief Calculate the minimum number of bytes to allocate in CUDA shared memory to store the grid.*/
    std::uint64_t sharedmem_size(void) const noexcept { return this->cumalloc_size(); }
#ifdef __NVCC__
    /** @brief Copy grid to a pre-allocated memory region by a GPU block of threads.
     *  @details The copy action is performed by the whole CUDA thread block.
     *  @param dest_ptr Memory region where the grid is copied to.
     *  @param grid_vector_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors, size of
     *  ``floatvec[this->ndim()] + double[this->size()]``.
     *  @param thread_idx Flatten ID of the current CUDA thread in the block.
     *  @param block_size Number of threads in the current CUDA block.
     */
    __cudevice__ void * copy_by_block(intpl::CartesianGrid * dest_ptr, void * grid_vector_data_ptr,
                                      std::uint64_t thread_idx, std::uint64_t block_size) const;
    /** @brief Copy grid to a pre-allocated memory region by a single GPU threads.
     *  @param dest_ptr Memory region where the grid is copied to.
     *  @param grid_vector_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors, size of
     *  ``floatvec[this->ndim()] + double[this->size()]``.
     */
    __cudevice__ void * copy_by_thread(intpl::CartesianGrid * dest_ptr, void * grid_vector_data_ptr) const;
#endif  // __NVCC__
    /// @}

    /// @name Grid merge operator
    /// @{
    /** @brief Union assignment of 2 Cartesian grids.*/
    MERLIN_EXPORTS intpl::CartesianGrid & operator+=(const intpl::CartesianGrid & grid);
    MERLIN_EXPORTS friend intpl::CartesianGrid operator+(const intpl::CartesianGrid & grid_1,
                                                         const intpl::CartesianGrid & grid_2);
    MERLIN_EXPORTS friend double exclusion_grid(const intpl::CartesianGrid & grid_parent,
                                                const intpl::CartesianGrid & grid_child, const floatvec & x);
    /// @}

    /// @name Query
    /// @{
    /** @brief Check if point in the grid.*/
    MERLIN_EXPORTS bool contains(const floatvec & point) const;
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
    Vector<floatvec> grid_vectors_;
    /** @brief Begin iterator.*/
    intpl::CartesianGrid::iterator begin_;
    /** @brief End iterator.*/
    intpl::CartesianGrid::iterator end_;
};

namespace intpl {

/** @brief Union of 2 Cartesian grids.*/
MERLIN_EXPORTS intpl::CartesianGrid operator+(const intpl::CartesianGrid & grid_1, const intpl::CartesianGrid & grid_2);

/** @brief Exclusion on each dimension of 2 Cartesian grids.*/
MERLIN_EXPORTS double exclusion_grid(const intpl::CartesianGrid & grid_parent, const intpl::CartesianGrid & grid_child,
                                     const floatvec & x);

}  // namespace intpl

}  // namespace merlin

#endif  // MERLIN_INTPL_CARTESIAN_GRID_HPP_
