// Copyright 2022 quocdang1998
#ifndef MERLIN_GRID_REGULAR_GRID_HPP_
#define MERLIN_GRID_REGULAR_GRID_HPP_

#include <cstdint>           // std::uint64_t
#include <initializer_list>  // std::initializer_list
#include <string>            // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::Array
#include "merlin/config.hpp"             // __cuhostdev__
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/grid/declaration.hpp"   // merlin::grid::RegularGrid
#include "merlin/vector.hpp"             // merlin::DoubleView, merlin::DoubleVec, merlin::Point

namespace merlin {

/** @brief Multi-dimensional grid of points.*/
class grid::RegularGrid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    RegularGrid(void) = default;
    /** @brief Constructor number of points.
     *  @details Allocate an empty grid of points.
     */
    MERLIN_EXPORTS RegularGrid(std::uint64_t ndim, std::uint64_t num_points = 0);
    /** @brief Constructor from an array of point coordinates.
     *  @param point_coordinates 2D array of shape ``[npoint, ndim]``, in which ``npoint`` is the number of points and
     *  ``ndim`` is the number of dimension of each point.
     */
    MERLIN_EXPORTS RegularGrid(const array::Array & point_coordinates);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    RegularGrid(const grid::RegularGrid & src) = default;
    /** @brief Copy assignment.*/
    grid::RegularGrid & operator=(const grid::RegularGrid & src) = default;
    /** @brief Move constructor.*/
    RegularGrid(grid::RegularGrid && src) = default;
    /** @brief Move assignment.*/
    grid::RegularGrid & operator=(grid::RegularGrid && src) = default;
    /// @}

    /// @name Get members and attributes
    /// @{
    /** @brief Get pointer to grid data.*/
    __cuhostdev__ constexpr DoubleView grid_data(void) const noexcept { return this->grid_data_.get_view(); }
    /** @brief Get dimensions of the grid.*/
    __cuhostdev__ constexpr std::uint64_t ndim(void) const noexcept { return this->ndim_; }
    /** @brief Get total number of points in the grid.*/
    __cuhostdev__ constexpr std::uint64_t size(void) const noexcept { return this->num_points_; }
    /** @brief Get available memory for the number of points in the grid.*/
    __cuhostdev__ constexpr std::uint64_t capacity(void) const noexcept { return this->grid_data_.size(); }
    /// @}

    /// @name Add and remove point in the grid
    /// @{
    /** @brief Add a point to the grid.
     *  @param new_point Vector of coordinates of the point to push.
     */
    MERLIN_EXPORTS void push_back(DoubleView new_point) noexcept;
    /** @brief Remove a point from the grid.
     *  @details Remove the last point from the grid.
     */
    MERLIN_EXPORTS Point pop_back(void) noexcept;
    /// @}

    /// @name Slicing operator
    /// @{
    /** @brief Write coordinate of point to a pre-allocated memory given flatten index.
     *  @param index Flatten index of point in the grid (in C order).
     *  @param point_data Pointer to memory recording point coordinate.
     */
    __cuhostdev__ void get(std::uint64_t index, double * point_data) const noexcept;
    /** @brief Get element at a given flatten index.
     *  @param index Flatten index of point in the grid (in C order).
     */
    Point operator[](std::uint64_t index) const noexcept {
        Point point(this->ndim_);
        this->get(index, point.data());
        return point;
    }
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
        return sizeof(grid::RegularGrid) + this->num_points_ * this->ndim_ * sizeof(double);
    }
    /** @brief Copy the grid from CPU to a pre-allocated memory on GPU.
     *  @details Values of vectors should be copied to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param grid_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors.
     *  @param stream_ptr Pointer to CUDA stream for asynchronous copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(grid::RegularGrid * gpu_ptr, void * grid_data_ptr,
                                      std::uintptr_t stream_ptr = 0) const;
    /** @brief Calculate the minimum number of bytes to allocate in CUDA shared memory to store the grid.*/
    std::uint64_t sharedmem_size(void) const noexcept { return sizeof(grid::RegularGrid); }
#ifdef __NVCC__
    /** @brief Copy grid to a pre-allocated memory region by a GPU block of threads.
     *  @details The copy action is performed by the whole CUDA thread block.
     *  @param dest_ptr Memory region where the grid is copied to.
     *  @param grid_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors, size of
     *  ``DoubleVec[this->ndim()] + double[this->size()]``.
     *  @param thread_idx Flatten ID of the current CUDA thread in the block.
     *  @param block_size Number of threads in the current CUDA block.
     */
    __cudevice__ void * copy_by_block(grid::RegularGrid * dest_ptr, void * grid_data_ptr, std::uint64_t thread_idx,
                                      std::uint64_t block_size) const;
    /** @brief Copy grid to a pre-allocated memory region by a single GPU threads.
     *  @param dest_ptr Memory region where the grid is copied to.
     *  @param grid_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors, size of
     *  ``DoubleVec[this->ndim()] + double[this->size()]``.
     */
    __cudevice__ void * copy_by_thread(grid::RegularGrid * dest_ptr, void * grid_data_ptr) const;
#endif  // __NVCC__
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    MERLIN_EXPORTS ~RegularGrid(void);
    /// @}

  protected:
    /** @brief Grid data.*/
    DoubleVec grid_data_;
    /** @brief Number of dimension.*/
    std::uint64_t ndim_;
    /** @brief Number of points in the grid.*/
    std::uint64_t num_points_;

  private:
    /** @brief Reallocate memory for the grid.*/
    void realloc(std::uint64_t new_npoints);
};

}  // namespace merlin

#endif  // MERLIN_GRID_REGULAR_GRID_HPP_
