// Copyright 2022 quocdang1998
#ifndef MERLIN_TENSOR_HPP_
#define MERLIN_TENSOR_HPP_

#include <initializer_list>  // std::initializer_list
#include <vector>  // std::vector

#include "merlin/array.hpp"  // Array

namespace merlin {

/** @brief Multi-dimensional array on CPU.*/
class Tensor : public Array {
  public:
    // Constructors
    // ------------
    /** @brief Default constructor (do nothing).*/
    Tensor(void) = default;
    /** @brief Construct 1D array holding a float value.
     *  @param value Assigned value.
     */
    Tensor(float value);
    /** @brief Construct C-contiguous empty array from dimension vector.
     *  @param shape Shape vector.
     */
    Tensor(std::initializer_list<unsigned int> shape);
    /** @brief Construct array from pointer, to data and meta-data.
     *  @param data Pointer to data.
     *  @param ndim Number of dimension of tensor.
     *  @param shape Pointer to tensor to size per dimension.
     *  @param strides Pointer to tensor to stride per dimension.
     *  @param copy Copy the original tensor to C-contiguous tensor.
     *  @note The original memory tied to the pointer will not be freed at destruction. But if copy is true, the
        copied tensor is automatically deallocated inside the destructor.
     */
    Tensor(float * data, unsigned int ndim, unsigned int * shape, unsigned int * strides, bool copy = false);

    // Copy and move
    // -------------
    /** @brief Deep copy constructor.*/
    Tensor(const Tensor & src);
    /** @brief Deep copy assignment.*/
    Tensor & operator=(const Tensor & src);
    /** @brief Move constructor.*/
    Tensor(Tensor && src);
    /** @brief Move assignment.*/
    Tensor & operator=(Tensor && src);

    // Iterator
    // --------
    class iterator {
      public:
        // Constructor
        // ^^^^^^^^^^^
        /** @brief Constructor from a vector of index and a dimension vector.
         *  @param it Vector of index.
         *  @param shape Reference to l-value vector of dimension.
         */
        iterator(const std::vector<unsigned int> & it, std::vector<unsigned int> & shape) : index_(it),
                                                                                            shape_(&shape) {}

        // Get members
        // ^^^^^^^^^^^
        /** @brief Get index of current iterator.*/
        std::vector<unsigned int> & index(void) {return this->index_;}
        /** @brief Get index of constant iterator.*/
        const std::vector<unsigned int> & index(void) const {return this->index_;}
        /** @brief Get reference to the dimension vector of the iterator.*/
        std::vector<unsigned int> & shape(void) {return *(this->shape_);}

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
         *  @details This operator is used to check if current iterator is the end iterator of Tensor.
         *  @param left Left iterator.
         *  @param right Right iterator.
         *  @return True if 2 iterators have the same index.
        */
        friend bool operator!= (const Tensor::iterator& left, const Tensor::iterator& right);

        private:
        // Members
        // ^^^^^^^
        /** @brief Index vector.*/
        std::vector<unsigned int> index_;
        /** @brief Pointer to max diemension vector.*/
        std::vector<unsigned int> * shape_;
    };

    // Atributes
    // ---------
    /** @brief Begin iterator.
     *  @details Vector of index \f$(0, 0, ..., 0)\f$.
     */
    Tensor::iterator begin(void);
    /** @brief End iterator.
     *  @details Vector of index \f$(d_0, 0, ..., 0)\f$.
     */
    Tensor::iterator end(void);
    /** @brief Sciling operator.
     *  @details Get an element at a given index.
     *  @param index Vector of indices along each dimension.
     *  @return Reference to the element at the provided index.
     */
    float & operator[] (const std::vector<unsigned int> & index);

    // Transfer data
    // -------------
    // Copy data from GPU array
    void sync_from_gpu(const Parcel & gpu_array, uintptr_t stream = 0);

    // Destructor
    // ----------
    /** @brief Destructor.*/
    ~Tensor(void);

  protected:
    // Members
    // -------
    /** @brief Decision to delete Tensor::data_ at destruction or not.*/
    bool force_free;
    /** @brief Index vector of begin element.*/
    std::vector<unsigned int> begin_;
    /** @brief Index vector of last element.*/
    std::vector<unsigned int> end_;
};

}  // namespace merlin

#endif  // MERLIN_TENSOR_HPP_
