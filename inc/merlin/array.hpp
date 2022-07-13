// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_HPP_
#define MERLIN_ARRAY_HPP_

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <vector>
#include <stdexcept>

namespace merlin {

/** @brief Multi-dimensional array.

    This class make the interface between C array and Numpy array.*/
template <typename Scalar>
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
        friend bool operator!= (const typename Array<Scalar>::iterator & left,
                                const typename Array<Scalar>::iterator & right) {
            // check if 2 iterators comes from the same array
            if (left.dims_ != right.dims_) {
                throw(std::runtime_error("2 iterators are not comming from the same array."));
            }
    
            // compare index of each iterator
            unsigned int length = left.index().size();
            for (int i = 0; i < length; i++) {
                if (left.index_[i] != right.index_[i]) {
                    return true;
                }
            }
            return false;
        }

      private:
        /** @brief Index vector.*/
        std::vector<unsigned int> index_;
        /** @brief Pointer to max diemension vector.*/
        std::vector<unsigned int> * dims_;
    };

    /** @brief Create array from NumPy array.*/
    Array(Scalar * data, unsigned int ndim,
          unsigned int * dims, unsigned int * strides,
          bool copy = true);
    /** @brief Destructor.*/
    ~Array(void);

    /** @brief Indicate array is copied or assigned to another array.

    If the array is copy version, delete operator is called when the array is
    destroyed to avoid memory leak.*/
    bool is_copy;
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
    Scalar & operator[] (const std::vector<unsigned int> & index);

    #ifdef __NVCC__
    /** @brief Copy data to GPU.*/
    void sync_to_gpu(void);
    /** @brief Copy data from GPU.*/
    void sync_from_gpu(void);
    #endif

  private:
    /** @brief Pointer to data.*/
    Scalar * data_;
    #ifdef __NVCC__
    /** @brief Pointer to data in GPU.*/
    Scalar * gpu_data_;
    #endif
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

// include source of array manipulation
#include "merlin/array/array_src.hpp"

#endif  // MERLIN_ARRAY_HPP_
