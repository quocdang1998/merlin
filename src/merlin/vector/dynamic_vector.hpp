// Copyright 2024 quocdang1998
#ifndef MERLIN_VECTOR_DYNAMIC_VECTOR_HPP_
#define MERLIN_VECTOR_DYNAMIC_VECTOR_HPP_

#include <cstdint>           // std::uint64_t
#include <initializer_list>  // std::initializer_list, std::data

#include "merlin/config.hpp"                   // __cuhostdev__, __cudevice__
#include "merlin/vector/declaration.hpp"       // merlin::vector::DynamicVector
#include "merlin/vector/iterator_helpers.hpp"  // merlin::vector::ForwardIterator, merlin::vector::ReverseIterator
#include "merlin/vector/view.hpp"              // merlin::vector::View

namespace merlin {

// Dynamic Vector
// --------------

/** @brief Dynamic array.
 *  @details Similar to ``std::vector``, but transportable to GPU global memory and shared memory.
 */
template <typename T>
class vector::DynamicVector {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ DynamicVector(void) {}
    /** @brief Constructor from initializer list.*/
    __cuhostdev__ DynamicVector(std::initializer_list<T> data);
    /** @brief Constructor from size and fill-in value.
     *  @param size Number of element.
     *  @param value Value of each element (must be copy constructible or copy assignable)
     */
    __cuhostdev__ DynamicVector(std::uint64_t size, const T & value = T());
    /** @brief Constructor from pointer to an iterator and size.
     *  @details Data are copied or assigned from the source location into the vector.
     *  @param data Pointer to the first element of source array.
     *  @param size Size of resulted vector (can be smaller or equals the original array).
     *  @param assign Flag indicating if the vector is assigned to the memory without copying.
     */
    template <typename Pointer>
    requires std::forward_iterator<Pointer> && std::convertible_to<std::iter_reference_t<Pointer>, T &>
    __cuhostdev__ DynamicVector(Pointer data, std::uint64_t size, bool assign = false);
    /** @brief Constructor from a range.
     *  @details Data are copied from the range into the vector.
     *  @param first Pointer to the first element.
     *  @param last Pointer to the last element.
     */
    template <typename Pointer>
    requires std::forward_iterator<Pointer> && std::convertible_to<std::iter_reference_t<Pointer>, const T &>
    __cuhostdev__ DynamicVector(Pointer first, Pointer last);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    __cuhostdev__ DynamicVector(const vector::DynamicVector<T> & src);
    /** @brief Copy assignment.*/
    __cuhostdev__ vector::DynamicVector<T> & operator=(const DynamicVector<T> & src);
    /** @brief Move constructor.*/
    __cuhostdev__ DynamicVector(vector::DynamicVector<T> && src);
    /** @brief Move assignment.*/
    __cuhostdev__ vector::DynamicVector<T> & operator=(vector::DynamicVector<T> && src);
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get reference to pointer of data.*/
    __cuhostdev__ constexpr T * data(void) { return this->data_; }
    /** @brief Get constant reference to pointer of data.*/
    __cuhostdev__ constexpr const T * data(void) const noexcept { return this->data_; }
    /** @brief Get constant reference to size.*/
    __cuhostdev__ constexpr const std::uint64_t & size(void) const noexcept { return this->size_; }
    /// @}

    /// @name Get element
    /// @{
    /** @brief Get reference to an element.*/
    __cuhostdev__ constexpr T & operator[](std::uint64_t i) noexcept { return this->data_[i]; }
    /** @brief Get constant reference to an element.*/
    __cuhostdev__ constexpr const T & operator[](std::uint64_t i) const noexcept { return this->data_[i]; }
    /// @}

    /// @name Assign and disown
    /// @{
    /** @brief Assign the object to a pre-allocated memory.
     *  @details Give up ownership to the current associated memory without deleting it, and assign the pointer to the
     *  pre-allocated memory. Assigned memory will not be freed at destruction.
     *  
     *  Calling this method on an object with ownership to its memory will result in memory leak. This method should
     *  only be used when the memory holding the object is in random state, such as when allocated with ``std::malloc``
     *  or GPU shared memory.
     */
    __cuhostdev__ constexpr void assign(T * data, std::uint64_t size) noexcept {
        this->data_ = data;
        this->size_ = size;
        this->assigned_ = true;
    }
    /** @brief Give up the owner ship of the current pointer.
     *  @details Give up the ownership to the current pointer. After this operation, the pointer associated to the
     *  object is set to ``nullptr``. Size and assignment state are still retrievable, but element accessing methods
     *  will be unusable.
     * 
     *  This function will return the pointer used to be owned by the object.
     */
    __cuhostdev__ constexpr T * disown(void) noexcept {
        T * old_data = this->data_;
        this->data_ = nullptr;
        return old_data;
    }
    /// @}

    /// @name Resize
    /// @{
    /** @brief Resize vector.*/
    __cuhostdev__ void resize(std::uint64_t new_size);
    /// @}

    /// @name Forward iterators
    /// @{
    /** @brief Forward iterator type.*/
    using iterator = vector::ForwardIterator<T>;
    /** @brief Constant forward iterator type.*/
    using const_iterator = vector::ForwardIterator<const T>;
    /** @brief Begin iterator.*/
    __cuhostdev__ constexpr iterator begin(void) { return iterator(this->data_); }
    /** @brief Begin iterator.*/
    __cuhostdev__ constexpr const_iterator begin(void) const { return const_iterator(this->data_); }
    /** @brief Constant begin iterator.*/
    __cuhostdev__ constexpr const_iterator cbegin(void) const { return const_iterator(this->data_); }
    /** @brief End iterator.*/
    __cuhostdev__ constexpr iterator end(void) { return iterator(this->data_ + this->size_); }
    /** @brief End iterator.*/
    __cuhostdev__ constexpr const_iterator end(void) const { return const_iterator(this->data_ + this->size_); }
    /** @brief Constant begin iterator.*/
    __cuhostdev__ constexpr const_iterator cend(void) const { return const_iterator(this->data_ + this->size_); }
    /// @}

    /// @name Reverse iterators
    /// @{
    /** @brief Reverse iterator type.*/
    using reverse_iterator = vector::ReverseIterator<T>;
    /** @brief Constant reverse iterator type.*/
    using const_reverse_iterator = vector::ReverseIterator<const T>;
    /** @brief Reverse begin iterator.*/
    __cuhostdev__ constexpr reverse_iterator rbegin(void) { return reverse_iterator(this->data_ + this->size_); }
    /** @brief Reverse begin iterator.*/
    __cuhostdev__ constexpr const_reverse_iterator rbegin(void) const {
        return const_reverse_iterator(this->data_ + this->size_);
    }
    /** @brief Constant begin iterator.*/
    __cuhostdev__ constexpr const_reverse_iterator crbegin(void) const {
        return const_reverse_iterator(this->data_ + this->size_);
    }
    /** @brief End iterator.*/
    __cuhostdev__ constexpr reverse_iterator rend(void) { return reverse_iterator(this->data_); }
    /** @brief End iterator.*/
    __cuhostdev__ constexpr const_reverse_iterator rend(void) const { return const_reverse_iterator(this->data_); }
    /** @brief Constant begin iterator.*/
    __cuhostdev__ constexpr const_reverse_iterator crend(void) const { return const_reverse_iterator(this->data_); }
    /// @}

    /// @name Get view
    /// @{
    /** @brief Get a view corresponding to the vector.*/
    __cuhostdev__ constexpr vector::View<T> get_view(void) const {
        return vector::View<T>(this->data_, this->size_);
    }
    /// @}

    /// @name Comparison
    /// @{
    friend __cuhostdev__ constexpr bool operator==(const vector::DynamicVector<T> & v1,
                                                   const vector::DynamicVector<T> & v2) {
        return v1.get_view() == v2.get_view();
    }
    friend __cuhostdev__ constexpr bool operator!=(const vector::DynamicVector<T> & v1,
                                                   const vector::DynamicVector<T> & v2) {
        return v1.get_view() != v2.get_view();
    }
    /// @}

    /// @name Transfer data from/to GPU
    /// @{
    /** @brief Calculate the minimum amount of memory (in bytes) to allocate for the object and its associated data.*/
    constexpr std::uint64_t cumalloc_size(void) const {
        return sizeof(vector::DynamicVector<T>) + this->size_ * sizeof(T);
    }
    /** @brief Copy data from CPU to a pre-allocated memory on GPU.
     *  @param gpu_ptr Pointer to the GPU memory region for the vector.
     *  @param data_ptr Pointer to the GPU memory region for elements of the vector.
     *  @param stream_ptr Pointer to the CUDA stream in case of an asynchronous copy.
     */
    void * copy_to_gpu(vector::DynamicVector<T> * gpu_ptr, void * data_ptr, std::uintptr_t stream_ptr = 0) const;
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
    __cudevice__ void * copy_by_block(vector::DynamicVector<T> * dest_ptr, void * data_ptr, std::uint64_t thread_idx,
                                      std::uint64_t block_size) const;
    /** @brief Copy data to a pre-allocated memory region by a single GPU threads.
     *  @details The copy action is perfomed by the current calling CUDA thread.
     *  @param dest_ptr Pointer to destination (where the vector is copied to).
     *  @param data_ptr Pointer to memory region for storing elements of the destination vector.
     */
    __cudevice__ void * copy_by_thread(vector::DynamicVector<T> * dest_ptr, void * data_ptr) const;
#endif  // __NVCC__
    /// @}

    /// @name String representation
    /// @{
    /** @brief String representation.
     *  @param sep Separator between printed elements.
     */
    std::string str(const char * sep = " ") const { return this->get_view().str(sep); }
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    __cuhostdev__ ~DynamicVector(void);
    /// @}

  private:
    /** @brief Pointer to data.*/
    T * data_ = nullptr;
    /** @brief Size of data.*/
    std::uint64_t size_ = 0;
    /** @brief Data is assigned version of a bigger data.*/
    bool assigned_ = false;
};

}  // namespace merlin

#include "merlin/vector/dynamic_vector.tpp"

#endif  // MERLIN_VECTOR_DYNAMIC_VECTOR_HPP_
