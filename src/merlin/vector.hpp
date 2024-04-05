// Copyright 2022 quocdang1998
#ifndef MERLIN_VECTOR_HPP_
#define MERLIN_VECTOR_HPP_

#include <cstddef>           // nullptr
#include <cstdint>           // std::uint64_t, std::uintptr_t
#include <initializer_list>  // std::initializer_list
#include <string>            // std::string

#include "merlin/cuda_interface.hpp"  // __cuhost__, __cuhostdev__

namespace merlin {

// Forward declaration
template <typename T>
class Vector;
template <typename T>
__cuhostdev__ bool operator==(const Vector<T> & vec_1, const Vector<T> & vec_2) noexcept;
template <typename T>
__cuhostdev__ bool operator!=(const Vector<T> & vec_1, const Vector<T> & vec_2) noexcept;
template <typename T>
__cuhostdev__ constexpr bool is_same_size(const Vector<T> & vec_1, const Vector<T> & vec_2) noexcept;
template <typename T>
__cuhostdev__ bool is_zeros(const Vector<T> & vec) noexcept;

/** @brief 1D non-resizable dynamic array.
 *  @details Similar to ``std::vector``, but transportable to GPU global memory and shared memory.
 *  @tparam T Numeric type (``float``, ``int``, etc).
 */
template <typename T>
class Vector {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ Vector(void) noexcept {}
    /** @brief Constructor from initializer list.*/
    __cuhostdev__ Vector(std::initializer_list<T> data) noexcept;
    /** @brief Constructor from size and fill-in value.
     *  @param size Number of element.
     *  @param value Value of each element (must be copy constructible or copy assignable)
     */
    __cuhostdev__ explicit Vector(std::uint64_t size, const T & value = T());
    /** @brief Copy constructor from a pointer to first and last element.
     *  @param ptr_first Pointer to the first element.
     *  @param ptr_last Pointer to the last element.
     */
    template <typename Convertable>
    __cuhostdev__ Vector(const Convertable * ptr_first, const Convertable * ptr_last);
    /** @brief Copy constructor from pointer to an array and size.
     *  @tparam Convertable Type convertable to ``T`` (constructor of ``T`` from ``Convertable``, i.e
     *  ``T(Convertable)`` must exists).
     *  @param ptr_src Pointer to the first element of source array.
     *  @param size Size of resulted vector (can be smaller or equals the original array).
     */
    template <typename Convertable>
    __cuhostdev__ Vector(const Convertable * ptr_src, std::uint64_t size);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    __cuhostdev__ Vector(const Vector<T> & src);
    /** @brief Copy assignment.*/
    __cuhostdev__ Vector<T> & operator=(const Vector<T> & src);
    /** @brief Move constructor.*/
    __cuhostdev__ Vector(Vector<T> && src);
    /** @brief Move assignment.*/
    __cuhostdev__ Vector<T> & operator=(Vector<T> && src);
    /// @}

    /// @name Direct assignment
    /// @{
    /** @brief Assign current vector to pre-allocated pointer.
     *  @param ptr_src Pointer to first element.
     *  @param size Size of the new vector.
     */
    __cuhostdev__ void assign(T * ptr_src, std::uint64_t size);
    /** @brief Assign current vector to pre-allocated pointer.
     *  @param ptr_first Pointer to first element.
     *  @param ptr_last Pointer to last element.
     */
    __cuhostdev__ void assign(T * ptr_first, T * ptr_last);
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get reference to pointer of data.*/
    __cuhostdev__ constexpr T *& data(void) { return this->data_; }
    /** @brief Get constant reference to pointer of data.*/
    __cuhostdev__ constexpr const T * data(void) const { return this->data_; }
    /** @brief Get reference to size.*/
    __cuhostdev__ constexpr std::uint64_t & size(void) { return this->size_; }
    /** @brief Get constant reference to size.*/
    __cuhostdev__ constexpr const std::uint64_t & size(void) const { return this->size_; }
    /** @brief Begin iterator.*/
    constexpr T * begin(void) { return this->data_; }
    /** @brief Begin iterator.*/
    constexpr const T * begin(void) const { return this->data_; }
    /** @brief Constant begin iterator.*/
    constexpr const T * cbegin(void) const { return this->data_; }
    /** @brief End iterator.*/
    constexpr T * end(void) { return &this->data_[this->size_]; }
    /** @brief End iterator.*/
    constexpr const T * end(void) const { return &this->data_[this->size_]; }
    /** @brief Constant begin iterator.*/
    constexpr const T * cend(void) const { return &this->data_[this->size_]; }
    /// @}

    /// @name Slicing operator
    /// @{
    /** @brief Get reference to an element.*/
    __cuhostdev__ constexpr T & operator[](std::uint64_t index) { return this->data_[index]; }
    /** @brief Get constant reference to an element.*/
    __cuhostdev__ constexpr const T & operator[](std::uint64_t index) const { return this->data_[index]; }
    /// @}

    /// @name Transfer data from/to GPU
    /// @{
    /** @brief Calculate the minimum amount of memory (in bytes) to allocate for the object and its associated data.*/
    constexpr std::uint64_t cumalloc_size(void) const { return sizeof(Vector<T>) + this->size_ * sizeof(T); }
    /** @brief Copy data from CPU to a pre-allocated memory on GPU.
     *  @param gpu_ptr Pointer to the GPU memory region for the vector.
     *  @param data_ptr Pointer to the GPU memory region for elements of the vector.
     *  @param stream_ptr Pointer to the CUDA stream in case of an asynchronous copy.
     */
    void * copy_to_gpu(Vector<T> * gpu_ptr, void * data_ptr, std::uintptr_t stream_ptr = 0) const;
    /** @brief Copy data from GPU back to CPU.
     *  @param gpu_ptr Pointer to data on GPU.
     *  @param stream_ptr Pointer to the CUDA stream in case of asynchronous copy.
     */
    void * copy_from_gpu(T * gpu_ptr, std::uintptr_t stream_ptr = 0);
    /** @brief Calculate the minimum memory (in bytes) to store the object in CUDA block's shared memory.*/
    constexpr std::uint64_t sharedmem_size(void) const { return this->cumalloc_size(); }
#ifdef __NVCC__
    /** @brief Copy data to a pre-allocated memory region by a block of threads.
     *  @details The copy action is performed parallely by the whole CUDA thread block.
     *  @param dest_ptr Pointer to destination (where the vector is copied to).
     *  @param data_ptr Pointer to memory region for storing elements of the destination vector.
     *  @param thread_idx Flatten ID of the current CUDA thread in the block.
     *  @param block_size Number of threads in the current CUDA block.
     */
    __cudevice__ void * copy_by_block(Vector<T> * dest_ptr, void * data_ptr, std::uint64_t thread_idx,
                                      std::uint64_t block_size) const;
    /** @brief Copy data to a pre-allocated memory region by a single GPU threads.
     *  @details The copy action is perfomed by the current calling CUDA thread.
     *  @param dest_ptr Pointer to destination (where the vector is copied to).
     *  @param data_ptr Pointer to memory region for storing elements of the destination vector.
     */
    __cudevice__ void * copy_by_thread(Vector<T> * dest_ptr, void * data_ptr) const;
#endif  // __NVCC__
    /// @}

    /// @name String representation
    /// @{
    /** @brief String representation.
     *  @param sep Separator between printed elements.
     */
    std::string str(const char * sep = " ") const;
    /// @}

    /// @name Operator
    /// @{
    /** @brief Identical comparison operator.*/
    friend __cuhostdev__ bool operator==<>(const Vector<T> & vec_1, const Vector<T> & vec_2) noexcept;
    /** @brief Different comparison operator.*/
    friend __cuhostdev__ bool operator!=<>(const Vector<T> & vec_1, const Vector<T> & vec_2) noexcept;
    /** @brief Check if 2 vectors have the same size.*/
    friend __cuhostdev__ constexpr bool is_same_size(const Vector<T> & vec_1, const Vector<T> & vec_2) noexcept {
        return vec_1.size_ == vec_2.size_;
    }
    /** @brief Check if all elements of vector are zeros.*/
    friend __cuhostdev__ bool is_zeros<>(const Vector<T> & vec) noexcept;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    __cuhostdev__ ~Vector(void);
    /// @}

  private:
    /** @brief Pointer to data.*/
    T * data_ = nullptr;
    /** @brief Size of data.*/
    std::uint64_t size_ = 0;
    /** @brief Data is aasigned version of a bigger data.*/
    bool assigned_ = false;
};

/** @brief Vector of unsigned integer values.*/
using UIntVec = Vector<std::uint64_t>;

/** @brief Vector of floating-point type.*/
using DoubleVec = Vector<double>;

/** @brief Create a vector from its arguments.*/
template <typename T, typename... Args>
Vector<T> make_vector(std::uint64_t size, Args... args) noexcept;

}  // namespace merlin

#include "merlin/vector.tpp"

#endif  // MERLIN_VECTOR_HPP_
