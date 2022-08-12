// Copyright 2022 quocdang1998
#ifndef MERLIN_TENSOR_HPP_
#define MERLIN_TENSOR_HPP_

#include <cstdint>
#include <list>
#include <vector>

namespace merlin {

/** @brief Multi-dimensional tensor of simple precision real values.

    This class make the interface between C tensor and Numpy tensor.
*/
class Tensor {
  public:
    /** @brief Multi-dimensional tensor iterator.*/
    class iterator {
      public:
        /** @brief Constructor from a vector of index and a dimension vector.

            @param it Vector of index.
            @param dims Reference to l-value vector of dimension.
        */
        iterator(const std::vector<unsigned int> & it, std::vector<unsigned int> & dims) : index_(it), dims_(&dims) {}

        /** @brief Get index of current iterator.*/
        std::vector<unsigned int> & index(void) {return this->index_;}
        /** @brief Get index of constant iterator.*/
        const std::vector<unsigned int> & index(void) const {return this->index_;}
        /** @brief Get reference to the dimension vector of the iterator.*/
        std::vector<unsigned int> & dims(void) {return *(this->dims_);}

        /** @brief Update the indexes after increasing value of index.

            User can manually add a certain amount to the index vector, making some indexes greater than their
            corresponding dimensions. This function will detect the surplus quantity, and update the indices by
            carrying the surplus to higher stride dimensions.

            Example: Given a dimension vector \f$(5, 2)\f$, the index vector \f$(1, 5)\f$ will be updated to
            \f$(3, 1)\f$.
        */
        void update(void);

        /** @brief Pre-increment operator.

            Increase the index of the last dimension by 1. If the maximum index is reached, set the index to zero and
            increment the next dimension.

            Example: \f$(0, 0) \rightarrow (0, 1) \rightarrow (0, 2) \rightarrow (1, 0) \rightarrow (1, 1) \rightarrow
            (1, 2)\f$.
        */
        iterator& operator++(void);
        /** @brief Post-increment operator.

            Same role as pre-increment operator.
        */
        iterator operator++(int);
        /** @brief Compare if the first iterator is smaller the second one.

            This operator is used to check if current iterator is the end iterator of Tensor.

            @param left Left iterator
            @param right Right iterator
        */
        friend bool operator!= (const Tensor::iterator& left, const Tensor::iterator& right);

      private:
        /** @brief Index vector.*/
        std::vector<unsigned int> index_;
        /** @brief Pointer to max diemension vector.*/
        std::vector<unsigned int> * dims_;
    };


    /** @brief Default constructor.*/
    Tensor(void) = default;
    /** @brief Construct an tensor holding value of one element.

        Construct a simplest tensor of dimension 1.
    */
    Tensor(float value);
    /** @brief Constructor from dims vector.

        Construct an empty contiguous tensor of dimension vector dims.

        @param dims Vector of dimension of the tensor.
    */
    Tensor(const std::vector<unsigned int> & dims);
    /** @brief Create tensor from NumPy tensor.

        @param data Pointer to data.
        @param ndim Number of dimension of tensor.
        @param dims Pointer to tensor to size per dimension.
        @param strides Pointer to tensor to stride per dimension.
        @param copy Copy the original tensor to C-contiguous tensor.
        @note The original memory tied to the pointer will not be freed at destruction. However, if copy is true, the
        copied tensor is freed inside the destructor.
    */
    Tensor(float * data, unsigned int ndim, unsigned int * dims, unsigned int * strides, bool copy = true);
    /** @brief Deep copy constructor.

        @param src Source to copy from.

        @note GPU data is not copied. GPU pointers to data is left untouched in old Tensor object.
    */
    Tensor(const Tensor & src);
    /** @brief Deep copy assignment.

        @param src Source to copy from.

        @note GPU data is not copied. GPU pointers to data is left untouched in old Tensor object.
    */
    Tensor & operator=(const Tensor & src);
    /** @brief Move constructor.

        @param src Source to move from.
    */
    Tensor(Tensor && src);
    /** @brief Move assignment.

        @param src Source to move from.
    */
    Tensor & operator=(Tensor && src);
    /** @brief Destructor.

        @note Destructor returns error when GPU data is not manually freed before its call.
    */
    ~Tensor(void) noexcept(false);


    /** @brief Pointer to (first element in) the data.*/
    float * data(void) {return this->data_;}
    /** @brief Number of dimensions.*/
    unsigned int ndim(void) {return this->ndim_;}
    /** @brief Reference to vector of size on each dimension.*/
    std::vector<unsigned int> & dims(void) {return this->dims_;}
    /** @brief Reference to vector of strides on each dimension.*/
    std::vector<unsigned int> & strides(void) {return this->strides_;}
    /** @brief Indicate if tensor data on CPU RAM is should be freed at destruction.

        If the tensor is copy version, or tensor data is dynamically allocated, delete operator must be called when the
        tensor is destroyed to avoid memory leak.
    */
    bool force_free = false;
    /** @brief Size of the tensor.

        Product of the size of each dimension.
    */
    unsigned int size(void);
    /** @brief Begin iterator.

        Vector of index \f$(0, 0, ..., 0)\f$.
    */
    Tensor::iterator begin(void);
    /** @brief End iterator.

        Vector of index \f$(d_0, 0, ..., 0)\f$.
    */
    Tensor::iterator end(void);
    /** @brief Sciling operator.

        Get an element at a given index.

        @param index Vector of indices along each dimension.
    */
    float & operator[] (const std::vector<unsigned int> & index);


    /** @brief Get GPU data.*/
    std::list<float *> & gpu_data(void) {return this->gpu_data_;}
    /** @brief Copy data to GPU.

        If the pointer to GPU data is not provided, new GPU memory will be allocated, and its pointer is saved to the
        list of pointers Tensor::gpu_data_.

        @code{.cpp}
            merlin::Tensor<double> A(A_data, 2, dims, strides, false);
            A.sync_to_gpu();  // allocate & copy data to gpu (this memory region has index 0)

            // do sth to the tensor

            A.sync_to_gpu(A.gpu_data().back())  // copy data to the same memory region allocated above

            // do sth to tensor on GPU

            A.sync_from_gpu(A.gpu_data().back())  // copy data on GPU back to CPU

            A.free_data_from_gpu(0);
        @endcode

        @param stream Synchronization stream.
        @param gpu_pdata Pointer to an allocated GPU data. If the pointer is NULL, allocate new memory and save the
        pointer to Tensor::gpu_data_.
    */
    void sync_to_gpu(float * gpu_pdata = NULL, uintptr_t stream = 0);
    /** @brief Copy data from GPU.

        @param gpu_pdata Pointer to data in the GPU.
        @param stream Synchronization stream.
    */
    void sync_from_gpu(float * gpu_pdata, uintptr_t stream = 0);
    /** @brief Free data from GPU.

        If index is \f$-1\f$, free all GPU data. Otherwise, free the data corresponding to the index.

        @param index Index of data pointer to be freed in the Tensor::gpu_data_ list.
    */
    void free_data_from_gpu(int index = -1);


  private:
    /** @brief Pointer to data.*/
    float * data_;
    /** @brief Pointer to data in GPU.*/
    std::list<float *> gpu_data_;
    /** @brief Number of dimension.*/
    unsigned int ndim_;
    /** @brief Size of each dimension.*/
    std::vector<unsigned int> dims_;
    /** @brief Strides (address jump) when increasing index of a dimension by 1.*/
    std::vector<unsigned int> strides_;

    /** @brief Begin iterator.*/
    std::vector<unsigned int> begin_;
    /** @brief End iterator.*/
    std::vector<unsigned int>  end_;

    /** @brief Copy from source tensor to a contiguous tensor.

        Copy from a source tensor to a C-contiguous tensor. Here the number of dimension and the sizes on each dimensions
        of the source and destination are the equal.

        @param src Pointer to source data.
        @param src_strides Pointer to strides tensor of the source.
    */
    void contiguous_copy_from_address_(float * src, const unsigned int * src_strides);
};

}  // namespace merlin

#endif  // MERLIN_TENSOR_HPP_
