// Copyright 2022 quocdang1998
#ifndef MERLIN_INTVEC_HPP_
#define MERLIN_INTVEC_HPP_

#include <cstddef>  // NULL
#include <initializer_list>  // std::initializer_list

#include "merlin/decorator.hpp"  // __cuhost__, __cuhostdev__

namespace merlin {

/** @brief 1D contiguous dynamic array (similar to ``std::vector``, with support for GPU array).
 *  @tparam T Numeric type (``float``, ``int``, etc)
 */
template <typename T>
class Vector {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ Vector(void) {}
    /** @brief Constructor from initializer list.*/
    Vector(std::initializer_list<T> data);
    /** @brief Constructor from size and fill-in value.*/
    Vector(unsigned long int size, T value = 0);
    /** @brief Copy constructor from a pointer to first and last element.*/
    Vector(T * ptr_first, T * ptr_last);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    Vector(const Vector<T> & src);
    /** @brief Copy assignment.*/
    Vector<T> & operator=(const Vector<T> & src);
    /** @brief Move constructor.*/
    Vector(Vector<T> && src);
    /** @brief Move assignment.*/
    Vector<T> & operator=(Vector<T> && src);
    /// @}

    /// @name Get members (Callable on both CPU and GPU)
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

    /// @name Slicing operator (Callable on both CPU and GPU)
    /// @{
    /** @brief Get reference to an element.*/
    __cuhostdev__ T & operator[](unsigned long int index) {return this->data_[index];}
    /** @brief Get constant reference to an element.*/
    __cuhostdev__ const T & operator[](unsigned long int index) const {return this->data_[index];}
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Minimum number of bytes to allocate in the memory to store the object and its data.
     *  @details <b> Callable on both CPU and GPU. </b>
     */
    __cuhostdev__ unsigned long int malloc_size(void) {return sizeof(Vector<T>) + this->size_*sizeof(unsigned long int);}
    /** @brief Copy data from CPU to a pre-allocated memory on GPU.
     *  @details <b> Callable only on CPU. </b>
     *
     *  The data is copied to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory. Note that this memory reagion must be big enough to store
     *  both the object and the its data.
     */
    void copy_to_device_ptr(Vector<T> * gpu_ptr);
    #ifdef __NVCC__
    /** @brief Copy data to a shared memory inside a kernel.
     *  @details <b> Callable only on GPU. </b>
     *  @param share_ptr Dynamically allocated shared pointer on GPU.
     */
    __cudevice__ void copy_to_shared_mem(Vector<T> * share_ptr);
    #endif  // __NVCC__
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Vector(void);
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

}  // namespace merlin

#include "merlin/vector.tpp"

#endif  // MERLIN_INTVEC_HPP_
