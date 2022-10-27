// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_GRID_HPP_
#define MERLIN_INTERPOLANT_GRID_HPP_

#include <cstdint>  // std::uint64_t
#include <initializer_list>  // std::initializer_list

#include "merlin/array/array.hpp"  // merlin::Array
#include "merlin/array/nddata.hpp"  // merlin::NdData, merlin::Iterator
#include "merlin/device/decorator.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/iterator.hpp"  // merlin::Iterator
#include "merlin/vector.hpp"  // merlin::Vector, merlin::intvec, merlin::floatvec

namespace merlin {

/** @brief A base class for all kinds of Grid.*/
class MERLIN_EXPORTS Grid {
  public:
    /** @brief Default constructor.*/
    __cuhostdev__ Grid(void) {}
    /** @brief Destructor.*/
    __cuhostdev__ ~Grid(void) {}

  protected:
    /** @brief Array holding coordinates of points in the Grid.*/
    array::NdData * points_ = NULL;
};

/** @brief A set of multi-dimensional points.*/
class MERLIN_EXPORTS RegularGrid : Grid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    RegularGrid(void) {}
    /** @brief Construct an empty grid from a given number of n-dim points.
     *  @param npoint Number of points in the grid.
     *  @param ndim Number of dimension of points in the grid.
     */
    RegularGrid(std::uint64_t npoint, std::uint64_t ndim);
    /** @brief Construct a grid and copy data from an array.
     *  @param points 2D merlin::Array of points, dimension ``(npoints, ndim)``.
     */
    RegularGrid(const array::Array & points);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    RegularGrid(const RegularGrid & src);
    /** @brief Copy assignment.*/
    RegularGrid & operator=(const RegularGrid & src);
    /** @brief Move constructor.*/
    RegularGrid(RegularGrid && src);
    /** @brief Move assignment.*/
    RegularGrid & operator=(RegularGrid && src);
    /// @}

    /// @name Get members and attributes
    /// @{
    /** @brief Get reference to array of grid points.*/
    array::Array grid_points(void) const {return *(dynamic_cast<array::Array *>(this->points_));}
    /** @brief Number of dimension of each point in the grid.*/
    std::uint64_t ndim(void) const {return this->points_->shape()[1];}
    /** @brief Number of points in the grid.*/
    std::uint64_t size(void) const {return this->npoint_;}
    /** @brief Maximum number of point which the RegularGrid can hold without reallocating memory.*/
    std::uint64_t capacity(void) const {return this->points_->shape()[0];}
    /// @}

    /// @name Iterator
    /// @{
    /** @brief RegularGrid iterator.*/
    using iterator = Iterator;
    /** @brief Begin iterator.*/
    RegularGrid::iterator begin(void);
    /** @brief End iterator.*/
    RegularGrid::iterator end(void);
    /// @}

    /// @name Modify points
    /// @{
    /** @brief Get reference Array to a point.
     *  @param index Index of point to get in the grid.
     */
    array::Array operator[](unsigned int index);
    /** @brief Append a point at the end of the grid.*/
    void push_back(Vector<float> && point);
    /** @brief Remove a point at the end of the grid.*/
    void pop_back(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    ~RegularGrid(void);
    /// @}

  protected:
    /** @brief Number of points in the grid.*/
    std::uint64_t npoint_;
    /** @brief Begin iterator.*/
    intvec begin_;
    /** @brief End iterator.*/
    intvec end_;
};

/** @brief Multi-dimensional Cartesian grid.*/
class MERLIN_EXPORTS CartesianGrid : public Grid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    CartesianGrid(void) {}
    /** @brief Constructor from a list of vector of values.*/
    CartesianGrid(std::initializer_list<floatvec> grid_vectors);
    /// @}

    /// @name Get members and attributes
    /// @{
    /** @brief Get grid vectors.*/
    __cuhostdev__ Vector<floatvec> & grid_vectors(void) {return this->grid_vectors_;}
    /** @brief Full tensor of each point in the CartesianGrid in form of 2D table.*/
    array::Array grid_points(void);
    /** @brief Number of dimension of the CartesianGrid.*/
    __cuhostdev__ std::uint64_t ndim(void) {return this->grid_vectors_.size();}
    /** @brief Number of points in the CartesianGrid.*/
    std::uint64_t size(void);
    /** @brief Shape of the grid.*/
    intvec grid_shape(void);
    /// @}

    /// @name Iterator
    /// @{
    using iterator = Iterator;
    /** @brief Begin iterator.*/
    CartesianGrid::iterator begin(void);
    /** @brief End iterator.*/
    CartesianGrid::iterator end(void);
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
    std::uint64_t malloc_size(void);
    /** @brief Copy the grid from CPU to a pre-allocated memory on GPU.
     *  @details Values of vectors should be copied to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param grid_vector_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors.
     */
    void copy_to_gpu(CartesianGrid * gpu_ptr, void * grid_vector_data_ptr);
    #ifdef __NVCC__
    /** @brief Copy meta-data from GPU global memory to shared memory of a kernel.
     *  @note This operation is single-threaded.
     *  @param share_ptr Dynamically allocated shared pointer on GPU.
     *  @param grid_vector_data_ptr Pointer to a pre-allocated GPU memory storing data of grid vectors.
     */
    __cudevice__ inline void copy_to_shared_mem(CartesianGrid * share_ptr, void * grid_vector_data_ptr);
    #endif  // __NVCC__
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    ~CartesianGrid(void);
    /// @}

  protected:
    /** @brief List of vector of values.*/
    Vector<floatvec> grid_vectors_;
    /** @brief Begin iterator.*/
    intvec begin_;
    /** @brief End iterator.*/
    intvec end_;
};

#ifdef __NVCC__
__cudevice__ inline void CartesianGrid::copy_to_shared_mem(CartesianGrid * share_ptr, void * grid_vector_data_ptr) {
    // shallow copy of grid vector
    bool check_zeroth_thread = (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
    if (check_zeroth_thread) {
        share_ptr->grid_vectors_.data() = reinterpret_cast<floatvec *>(grid_vector_data_ptr);
        share_ptr->grid_vectors_.size() = this->ndim();
    }
    __syncthreads();
    // copy data of each grid vector
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(grid_vector_data_ptr) + this->ndim()*sizeof(floatvec);
    for (int i = 0; i < this->ndim(); i++) {
        this->grid_vectors_[i].copy_to_shared_mem(&(share_ptr->grid_vectors_[i]), reinterpret_cast<float *>(data_ptr));
        data_ptr += this->grid_vectors_[i].size() * sizeof(float);
    }
}
#endif  // __NVCC__

}  // namespace merlin

#endif  // MERLIN_INTERPOLANT_GRID_HPP_
