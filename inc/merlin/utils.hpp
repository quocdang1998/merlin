// Copyright 2022 quocdang1998
#ifndef MERLIN_UTILS_HPP_
#define MERLIN_UTILS_HPP_

#include <vector>
#include <tuple>

namespace merlin {

/** @brief Calculate stride array from dims vector and size of one element as if the Array is C-contiguous.

    @param dims Dimension vector.
    @param size Size on one element in the array.
*/
std::vector<unsigned int> contiguous_strides(const std::vector<unsigned int> & dims, unsigned int element_size);


/** @brief Calculate the leap.

    Get the number of bytes separating zeroth element and element with a given index.

    @param index Index of the element to get.
    @param strides Strides array.
*/
unsigned int leap(const std::vector<unsigned int> & index, const std::vector<unsigned int> & strides);


/** @brief Calculate the longest contiguous segment and break index of an array.

    Longest contiguous segment is the length (in bytes) of the longest sub-array that is
    C-contiguous in the memory.

    Break index is the index at which the array break.

    For exmample, suppose ``A = [[1.0,2.0,3.0],[4.0,5.0,6.0]]``, then ``A[:,::2]`` will have
    longest contiguous segment of 4 and break index of 0.

    @param dims Dimension vector.
    @param strides Strides vector.
*/
std::tuple<unsigned int, int> lcseg_and_brindex(const std::vector<unsigned int> & dims,
                                                const std::vector<unsigned int> & strides);


/** @brief Convert n-dimensional index to C-contiguous index.

    @param index Multi-dimensional index.
    @param dims Size of each dimension.
*/
unsigned int ndim_to_contiguous_idx(const std::vector<unsigned int> & index, const std::vector<unsigned int> & dims);


/** @brief Convert vector of C-contiguous index to vector of n-dimensional index.

    @param index C-contiguous index.
    @param dims Size of each dimension.
*/
std::vector<std::vector<unsigned int>> contiguous_to_ndim_idx(const std::vector<unsigned int> & index,
                                                              const std::vector<unsigned int> & dims);


std::vector<std::vector<unsigned int>> partition_list(unsigned int length, unsigned int n_segment);

}  // namespace merlin

#endif  // MERLIN_UTILS_HPP_
