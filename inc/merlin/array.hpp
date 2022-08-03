// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_HPP_
#define MERLIN_ARRAY_HPP_

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace merlin {

/** @brief Multi-dimensional array of simple precision real values.

    This class make the interface between C array and Numpy array.*/
class Array {
  public:
    /** @brief Multi-dimensional array iterator.*/
    class iterator {
      public:
        /** @brief Constructor from a vector of index and a dimension vector.*/
        iterator(const std::vector<unsigned int> & it,
                 std::vector<unsigned int> & dims) : index_(it), dims_(&dims) {}

        /** @brief Get index of current iterator.*/
        std::vector<unsigned int> & index(void) {return this->index_;}
        /** @brief Get index of constant iterator.*/
        const std::vector<unsigned int> & index(void) const {return this->index_;}
        /** @brief Get reference to the dimension vector of the iterator.*/
        std::vector<unsigned int> & dims(void) {return *(this->dims_);}

        /** @brief Update the indexes after increasing value of index.

            User can manually add a certain amount to the index vector, making
            some indexes greater than their corresponding dimensions. This
            function will detect the surplus quantity, and update the indices
            by carrying the surplus to higher stride dimensions.
            
            Example: Given a dimension vector \f$(5, 2)\f$, the index vector
            \f$(1, 5)\f$ will be updated to \f$(3, 1)\f$*/
        void update(void);

        /** @brief Pre-increment operator.

            Increase the index of the last dimension by 1. If the maximum is
            reached, set the index to zero and increment the next dimension.

            Example: \f$(0, 0) \rightarrow (0, 1) \rightarrow (0, 2)
            \rightarrow (1, 0) \rightarrow (1, 1) \rightarrow (1, 2)\f$.*/
        iterator& operator++(void);
        /** @brief Post-increment operator.
        
            Same role as pre-increment operator.*/
        iterator operator++(int);
        /** @brief Compare if the first iterator is smaller the second one.

            This operator is used to check if current iterator is the end
            iterator of Array.*/
        friend bool operator!= (const Array::iterator& left, const Array::iterator& right);

      private:
        /** @brief Index vector.*/
        std::vector<unsigned int> index_;
        /** @brief Pointer to max diemension vector.*/
        std::vector<unsigned int> * dims_;
    };

    /** @brief Default constructor.*/
    Array(void) = default;
    /** @brief Construct an array holding value of one element.
    
    Construct a simplest array of dimension 1.*/
    Array(float value);
    /** @brief Constructor from dims vector.
    
    Construct an empty contiguous array of dimension vector dims.
    
    @param dims Vector of dimension of the array.*/
    explicit Array(const std::vector<unsigned int> & dims);
    /** @brief Create array from NumPy array.*/
    Array(float * data, unsigned int ndim,
          unsigned int * dims, unsigned int * strides,
          bool copy = true);
    /** @brief Deep copy constructor.
    
    @warning GPU data is not copied automatically. User must call the method sync_to_gpu to clone
    data.*/
    Array(const Array & src);
    /** @brief Deep copy assignment.*/
    Array& operator=(const Array & src);
    /** @brief Move constructor.*/
    Array(Array && src);
    /** @brief Move assignment.*/
    // Array& operator=(Array&& src);
    /** @brief Destructor.*/
    ~Array(void);

    /** @brief Indicate if array data is should be freed at destruction.

    If the array is copy version, or array data is dynamically allocated, delete operator must be
    called when the array is destroyed to avoid memory leak.*/
    bool force_free;
    /** @brief Size of the array.

    Product of the size of each dimension.*/
    unsigned int size(void);
    /** @brief Begin iterator.

    Vector of index \f$(0, 0, ..., 0)\f$.*/
    Array::iterator begin(void);
    /** @brief End iterator.

    Vector of index \f$(d_0, 0, ..., 0)\f$.*/
    Array::iterator end(void);
    /** @brief Sciling operator.

    Get an element at a given index.
    @param index Vector of indices along each dimension.*/
    float & operator[] (const std::vector<unsigned int> & index);

    /** @brief Get GPU data.*/
    float * gpu_data(void) {return this->gpu_data_;}
    #ifdef __NVCC__
    /** @brief Copy data to GPU.
    
        If the synchronization stream is not provided, this function will
        create it own stream to copy and sync the data.

        @code{.cpp}
        merlin::Array<double> A(A_data, 2, dims, strides, false);
        A.sync_to_gpu();
        @endcode

        @param stream Synchronization stream.
    */
    void sync_to_gpu(cudaStream_t stream = NULL);
    /** @brief Copy data from GPU.
    
        If the synchronization stream is not provided, this function will
        create it own stream to copy and sync the data.

        @param stream Synchronization stream.
    */
    void sync_from_gpu(cudaStream_t stream = NULL);
    #endif  // __NVCC__

  private:
    /** @brief Pointer to data.*/
    float * data_;
    /** @brief Pointer to data in GPU.*/
    float * gpu_data_ = NULL;
    /** @brief Number of dimension.*/
    unsigned int ndim_;
    /** @brief Size of each dimension.*/
    std::vector<unsigned int> dims_;
    /** @brief Strides (address jump) when increasing index of a dimension
    by 1.*/
    std::vector<unsigned int> strides_;

    /** @brief Begin iterator.*/
    std::vector<unsigned int> begin_;
    /** @brief End iterator.*/
    std::vector<unsigned int>  end_;

    /** @brief Longest contiguous segment.*/
    unsigned int longest_contiguous_segment_;
    /** @brief Index at which the longest contiguous segment is break.*/
    int break_index_;
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_HPP_
