// Copyright 2022 quocdang1998
#ifndef MERLIN_GRID_CARTESIAN_GRID_HPP_
#define MERLIN_GRID_CARTESIAN_GRID_HPP_

#include <cstdint>           // std::uint64_t
#include <initializer_list>  // std::initializer_list
#include <string>            // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::Array
#include "merlin/cuda_interface.hpp"     // __cuhostdev__
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/slice.hpp"              // merlin::slicevec
#include "merlin/grid/declaration.hpp"   // merlin::grid::CartesianGrid
#include "merlin/vector.hpp"             // merlin::floatvec, merlin::intvec, merlin::Vector

namespace merlin {

/** @brief Multi-dimensional Cartesian grid.
 *  @details The i-th coordinate of each point in the grid is an element derived from the i-th vector containing real
 *  values (each element within this vector is called grid node). A Cartesian grid is formed by taking the Cartesian
 *  product over a set of vectors of nodes, representing the set of all possible distinct points that can constructed
 *  from the set of vectors.
 */
class grid::CartesianGrid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    CartesianGrid(void) = default;
    /** @brief Constructor from list of initializer lists.*/
    MERLIN_EXPORTS CartesianGrid(const Vector<floatvec> & grid_vectors);
    /** @brief Constructor as a sub-grid from a larger grid.*/
    MERLIN_EXPORTS CartesianGrid(const grid::CartesianGrid & whole, const slicevec & slices);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    MERLIN_EXPORTS CartesianGrid(const grid::CartesianGrid & src);
    /** @brief Copy assignment.*/
    MERLIN_EXPORTS grid::CartesianGrid & operator=(const grid::CartesianGrid & src);
    /** @brief Move constructor.*/
    CartesianGrid(grid::CartesianGrid && src) = default;
    /** @brief Move assignment.*/
    grid::CartesianGrid & operator=(grid::CartesianGrid && src) = default;
    /// @}

    /// @name Get members and attributes
    /// @{
    /** @brief Get grid vector of a given dimension.*/
    __cuhostdev__ const floatvec grid_vector(std::uint64_t i_dim) const noexcept {
        floatvec grid_vector;
        grid_vector.assign(const_cast<double *>(this->grid_vectors_[i_dim]), this->grid_shape_[i_dim]);
        return grid_vector;
    }
    /** @brief Get constant reference to grid vector pointers.*/
    __cuhostdev__ constexpr const Vector<double *> & grid_vectors(void) const noexcept { return this->grid_vectors_; }
    /** @brief Get dimensions of the grid.*/
    __cuhostdev__ constexpr std::uint64_t ndim(void) const noexcept { return this->grid_shape_.size(); }
    /** @brief Get shape of the grid.*/
    __cuhostdev__ constexpr const intvec & shape(void) const noexcept { return this->grid_shape_; }
    /** @brief Get total number of points in the grid.*/
    __cuhostdev__ constexpr std::uint64_t size(void) const noexcept { return this->size_; }
    /** @brief Get total number of nodes on all dimension.*/
    __cuhostdev__ constexpr std::uint64_t num_nodes(void) const noexcept { return this->grid_nodes_.size(); }
    /// @}

    /// @name Slicing operator
    /// @{
    /** @brief Get element at a given flatten index.
     *  @param index Flatten index of point in the grid (in C order).
     */
    MERLIN_EXPORTS floatvec operator[](std::uint64_t index) const noexcept;
    /** @brief Get element at a given index vector.
     *  @param index Vector of index on each dimension.
     */
    MERLIN_EXPORTS floatvec operator[](const intvec & index) const noexcept;
    /// @}

    /// @name Get points
    /// @{
    /** @brief Get all points in the grid.*/
    MERLIN_EXPORTS array::Array get_points(void) const;
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the grid and its data.*/
    std::uint64_t cumalloc_size(void) const noexcept {
        std::uint64_t size = sizeof(grid::CartesianGrid);
        size += this->num_nodes() * sizeof(double) + this->ndim() * (sizeof(std::uint64_t) + sizeof(double *));
        return size;
    }
    /** @brief Copy the grid from CPU to a pre-allocated memory on GPU.
     *  @details Values of vectors should be copied to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param grid_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors.
     *  @param stream_ptr Pointer to CUDA stream for asynchronous copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(grid::CartesianGrid * gpu_ptr, void * grid_data_ptr,
                                      std::uintptr_t stream_ptr = 0) const;
    /** @brief Calculate the minimum number of bytes to allocate in CUDA shared memory to store the grid.*/
    std::uint64_t sharedmem_size(void) const noexcept { return this->cumalloc_size(); }
#ifdef __NVCC__
    /** @brief Copy grid to a pre-allocated memory region by a GPU block of threads.
     *  @details The copy action is performed by the whole CUDA thread block.
     *  @param dest_ptr Memory region where the grid is copied to.
     *  @param grid_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors, size of
     *  ``floatvec[this->ndim()] + double[this->size()]``.
     *  @param thread_idx Flatten ID of the current CUDA thread in the block.
     *  @param block_size Number of threads in the current CUDA block.
     */
    __cudevice__ void * copy_by_block(grid::CartesianGrid * dest_ptr, void * grid_data_ptr, std::uint64_t thread_idx,
                                      std::uint64_t block_size) const;
    /** @brief Copy grid to a pre-allocated memory region by a single GPU threads.
     *  @param dest_ptr Memory region where the grid is copied to.
     *  @param grid_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors, size of
     *  ``floatvec[this->ndim()] + double[this->size()]``.
     */
    __cudevice__ void * copy_by_thread(grid::CartesianGrid * dest_ptr, void * grid_data_ptr) const;
#endif  // __NVCC__
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
    /** @brief Number of points in the grid.*/
    std::uint64_t size_ = 0;

  private:
    /** @brief Pointer to first node in each dimension.*/
    Vector<double *> grid_vectors_;
};

}  // namespace merlin

#endif  // MERLIN_GRID_CARTESIAN_GRID_HPP_
