// Copyright 2022 quocdang1998NdData
#ifndef MERLIN_TENSOR_HPP_
#define MERLIN_TENSOR_HPP_

#include <cstdint>  // uintptr_t
#include <initializer_list>  // std::initializer_list

#include "merlin/nddata.hpp"  // merlin::NdData, merlin::Parcel
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

/** @brief Multi-dimensional array on CPU.*/
class Array : public NdData {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor (do nothing).*/
    Array(void) = default;
    /** @brief Construct 1D array holding a float value.
     *  @param value Assigned value.
     */
    Array(float value);
    /** @brief Construct C-contiguous empty array from dimension vector.
     *  @param shape Shape vector.
     */
    Array(std::initializer_list<unsigned long int> shape);
    /** @brief Construct array from pointer, to data and meta-data.
     *  @param data Pointer to data.
     *  @param ndim Number of dimension of tensor.
     *  @param shape Pointer to tensor to size per dimension.
     *  @param strides Pointer to tensor to stride per dimension.
     *  @param copy Copy the original tensor to C-contiguous tensor.
     *  @note The original memory tied to the pointer will not be freed at destruction. But if copy is true, the
        copied tensor is automatically deallocated inside the destructor.
     */
    Array(float * data, unsigned long int ndim,
          const unsigned long int * shape, const unsigned long int * strides, bool copy = false);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Deep copy constructor.*/
    Array(const Array & src);
    /** @brief Deep copy assignment.*/
    Array & operator=(const Array & src);
    /** @brief Move constructor.*/
    Array(Array && src);
    /** @brief Move assignment.*/
    Array & operator=(Array && src);
    /// @}

    /** @brief Iterator of a multi-dimensional array.*/
    class iterator {
      public:
        // Constructor
        // ^^^^^^^^^^^
        /** @brief Constructor from a vector of index and a dimension vector.
         *  @param it Vector of index.
         *  @param shape Reference to l-value vector of dimension.
         */
        iterator(const intvec & it, intvec & shape) : index_(it), shape_(&shape) {}

        // Get members
        // ^^^^^^^^^^^
        /** @brief Get index of current iterator.*/
        intvec & index(void) {return this->index_;}
        /** @brief Get index of constant iterator.*/
        const intvec & index(void) const {return this->index_;}
        /** @brief Get reference to the dimension vector of the iterator.*/
        intvec & shape(void) {return *(this->shape_);}

        // Attributes
        // ^^^^^^^^^^
        [[deprecated("This function should only be used when jumping a step bigger than 1.")]]
        void update(void);

        // Operators
        // ^^^^^^^^^
        /** @brief Pre-increment operator.
         *  @details Increase the index of the last dimension by 1. If the maximum index is reached, set the index to
         *  zero and increment the next dimension.
         *
         *  Example: \f$(0, 0) \rightarrow (0, 1) \rightarrow (0, 2) \rightarrow (1, 0) \rightarrow (1, 1) \rightarrow
         *  (1, 2)\f$.
         */
        iterator& operator++(void);
        /** @brief Post-increment operator.
         *  @details Same role as pre-increment operator.
         */
        iterator operator++(int);
        /** @brief Compare if the first iterator is different than the second one.
         *  @details This operator is used to check if current iterator is the end iterator of Array.
         *  @param left Left iterator.
         *  @param right Right iterator.
         *  @return True if 2 iterators have the same index.
        */
        friend bool operator!= (const Array::iterator& left, const Array::iterator& right);

        private:
        // Members
        // ^^^^^^^
        /** @brief Index vector.*/
        intvec index_;
        /** @brief Pointer to max diemension vector.*/
        intvec * shape_;
    };

    /// @name Atributes
    /// @{
    /** @brief Begin iterator.
     *  @details Vector of index \f$(0, 0, ..., 0)\f$.
     */
    Array::iterator begin(void);
    /** @brief End iterator.
     *  @details Vector of index \f$(d_0, 0, ..., 0)\f$.
     */
    Array::iterator end(void);
    /** @brief Sciling operator.
     *  @details Get an element at a given index.
     *  @param index Vector of indices along each dimension.
     *  @return Reference to the element at the provided index.
     */
    float & operator[] (const intvec & idx);
    /// @}

    /// @name Transfer data
    /// @{
    /** @brief Copy data from GPU array.*/
    void sync_from_gpu(const Parcel & gpu_array, uintptr_t stream = 0);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Array(void);
    /// @}

  protected:
    // Members
    // -------
    /** @brief Decision to delete Array::data_ at destruction or not.*/
    bool force_free;
    /** @brief Index vector of begin element.*/
    intvec begin_;
    /** @brief Index vector of last element.*/
    intvec end_;
};

}  // namespace merlin

#endif  // MERLIN_TENSOR_HPP_
