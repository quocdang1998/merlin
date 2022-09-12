// Copyright 2022 quocdang1998
#ifndef MERLIN_VECTOR_HPP_
#define MERLIN_VECTOR_HPP_

#include <cstddef>  // NULL
#include <initializer_list>  // std::initializer_list

#include "merlin/decorator.hpp"  // __cuhost__, __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_TEMPLATE_EXPORTS

namespace merlin {

/** @brief 1D un-resizable dynamic array.
 *  @details Similar to ``std::vector``, but transportable to GPU global memory and shared memory.
 *  @tparam T Numeric type (``float``, ``int``, etc)
 */
template <typename T>
class MERLIN_TEMPLATE_EXPORTS Vector {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ Vector(void) {}
    /** @brief Constructor from initializer list.*/
    __cuhostdev__ Vector(std::initializer_list<T> data);
    /** @brief Constructor from size and fill-in value.*/
    __cuhostdev__ Vector(unsigned long int size, T value = 0);
    /** @brief Copy constructor from a pointer to first and last element.
     *  @tparam Convertable Type convertable to ``T`` (constructor of ``T`` from ``Convertable``, i.e ``T(Convertable)``
     *  must exists).
     *  @param ptr_first Pointer to the first element.
     *  @param ptr_last Pointer to the last element.
     */
    template <typename Convertable>
    __cuhostdev__ Vector(const Convertable * ptr_first, const Convertable * ptr_last);
    /** @brief Copy constructor from pointer to an array and size.
     *  @tparam Convertable Type convertable to ``T`` (constructor of ``T`` from ``Convertable``, i.e ``T(Convertable)``
     *  must exists).
     *  @param ptr_src Pointer to the first element of source array.
     *  @param size Size of resulted vector (can be smaller or equals the original array).
     */
    template <typename Convertable>
    __cuhostdev__ Vector(const Convertable * ptr_src, unsigned long int size);
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

    /// @name Get members
    /// @{
    /** @brief Get reference to pointer of data.*/
    __cuhostdev__ T * & data(void) {return this->data_;}
    /** @brief Get constant reference to pointer of data.*/
    __cuhostdev__ const T * data(void) const {return this->data_;}
    /** @brief Get reference to size.*/
    __cuhostdev__ unsigned long int & size(void) {return this->size_;}
    /** @brief Get constant reference to size.*/
    __cuhostdev__ const unsigned long int & size(void) const {return this->size_;}
    /// @}

    /// @name Slicing operator
    /// @{
    /** @brief Get reference to an element.*/
    __cuhostdev__ T & operator[](unsigned long int index) {return this->data_[index];}
    /** @brief Get constant reference to an element.*/
    __cuhostdev__ const T & operator[](unsigned long int index) const {return this->data_[index];}
    /// @}

    /// @name Transfer data from/to GPU
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the object and its data.*/
    __cuhostdev__ unsigned long int malloc_size(void) {return sizeof(Vector<T>) + this->size_*sizeof(unsigned long int);}
    /** @brief Copy data from CPU to a pre-allocated memory on GPU.
     *  @details The object and its data is copied to the global memory of the GPU.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory storing the object.
     *  @param data_ptr Pre-allocated pointer to memory region storing data of the vector.
     */
    void copy_to_gpu(Vector<T> * gpu_ptr, T * data_ptr);
    /** @brief Copy data from GPU to CPU.
     *  @param gpu_ptr Pointer to object on GPU global memory.
     */
    void copy_from_device(Vector<T> * gpu_ptr);
    #ifdef __NVCC__
    /** @brief Copy data from GPU global memory to shared memory of a kernel.
     *  @note This operation is single-threaded.
     *  @param share_ptr Dynamically allocated shared pointer on GPU.
     *  @param data_ptr Pre-allocated pointer to memory region storing data of the vector.
     */
    __cudevice__ void copy_to_shared_mem(Vector<T> * share_ptr, T * data_ptr);
    #endif  // __NVCC__
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    __cuhostdev__ ~Vector(void);
    /// @}

  private:
    /** @brief Pointer to data.*/
    T * data_ = NULL;
    /** @brief Size of data.*/
    unsigned long int size_;
};

/** @brief Vector of unsigned integer values.
 *  @details This class is reserved for array indices, array shape vector and array strides vector.
 */
using intvec = Vector<unsigned long int>;

/** @brief Vector of simple precision values.*/
using floatvec = Vector<float>;

}  // namespace merlin

#include "merlin/vector.tpp"

#endif  // MERLIN_VECTOR_HPP_
