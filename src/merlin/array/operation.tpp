// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_OPERATION_TPP_
#define MERLIN_ARRAY_OPERATION_TPP_

#include <algorithm>  // std::copy, std::fill_n, std::min, std::max
#include <cstdlib>    // std::lldiv
#include <sstream>    // std::ostringstream

#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"   // merlin::inner_prod, merlin::prod_elements

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// NdData tools
// ---------------------------------------------------------------------------------------------------------------------

// Copy each segment from source to destination
template <class CopyFunction>
void array::copy(array::NdData * dest, const array::NdData * src, CopyFunction copy) {
    // check if shape vector are the same
    if (src->ndim() != dest->ndim()) {
        FAILURE(std::invalid_argument, "Cannot copy array of different ndim (%u to %u).\n", src->ndim(), dest->ndim());
    }
    std::uint64_t ndim = src->ndim();
    if (!(dest->shape() == src->shape())) {
        FAILURE(std::invalid_argument, "Expected shape of source equals shape of destination.\n");
    }
    // trivial case: size zero
    if (ndim == 0) {
        return;
    }
    // longest contiguous segment and break index
    auto [src_lcs, src_bridx] = array::lcseg_and_brindex(src->shape(), src->strides(), ndim);
    auto [dest_lcs, dest_bridx] = array::lcseg_and_brindex(dest->shape(), dest->strides(), ndim);
    std::uint64_t longest_contiguous_segment = std::min(src_lcs, dest_lcs);
    std::uint64_t break_index = std::max(static_cast<std::int64_t>(src_bridx), static_cast<std::int64_t>(dest_bridx));
    // copy each longest contiguous segment through the copy function
    if (break_index == UINT_MAX) {  // original arrays are perfectly contiguous
        copy(dest->data(), src->data(), longest_contiguous_segment);
        return;
    }
    // otherwise, memcpy each longest_contiguous_segment
    std::uint64_t num_segments = prod_elements(src->shape().data(), ndim);
    for (std::uint64_t i_segment = 0; i_segment < num_segments; i_segment++) {
        // calculate ptr to each segment
        std::uint64_t src_leap = array::get_leap(i_segment, src->shape(), src->strides(), ndim);
        std::uintptr_t src_ptr = reinterpret_cast<std::uintptr_t>(src->data()) + src_leap;
        std::uint64_t dest_leap = array::get_leap(i_segment, dest->shape(), dest->strides(), ndim);
        std::uintptr_t dest_ptr = reinterpret_cast<std::uintptr_t>(dest->data()) + dest_leap;
        // copy the segment
        copy(reinterpret_cast<double *>(dest_ptr), reinterpret_cast<double *>(src_ptr), longest_contiguous_segment);
    }
}
/*
// Fill all array with a given value
template <class CopyFunction>
void array::fill(array::NdData * target, double fill_value, CopyFunction write_engine, std::uint64_t buffer) {
    // trivial case: size zero
    if (target->ndim() == 0) {
        return;
    }
    // allocate a buffer and fill it with data
    auto [longest_contiguous_segment, break_index] = array::lcseg_and_brindex(target->shape(), target->strides());
    std::uint64_t buffer_size = std::min(longest_contiguous_segment, buffer);
    double * buffer_data = reinterpret_cast<double *>(new char[buffer_size]);
    std::fill_n(buffer_data, buffer_size / sizeof(double), fill_value);
    // copy each segment through the copy function
    if (break_index == -1) {  // original arrays are perfectly contiguous
        std::lldiv_t div_result = std::lldiv(longest_contiguous_segment, buffer_size);
        for (std::uint64_t subsegment = 0; subsegment < div_result.quot; subsegment++) {
            write_engine(reinterpret_cast<double *>(reinterpret_cast<char *>(target->data()) + subsegment*buffer_size),
                         buffer_data, buffer_size);
        }
        write_engine(reinterpret_cast<double *>(reinterpret_cast<char *>(target->data()) + div_result.quot*buffer_size),
                     buffer_data, div_result.rem);
    } else {  // memcpy each longest_contiguous_segment
        // initilize index vector
        intvec index(target->ndim(), 0);
        while (true) {
            // calculate ptr to each segment
            std::uint64_t leap = inner_prod(index, target->strides());
            std::uintptr_t des_ptr = reinterpret_cast<std::uintptr_t>(target->data()) + leap;
            // copy the segment
            std::lldiv_t div_result = std::lldiv(longest_contiguous_segment, buffer_size);
            for (std::uint64_t subsegment = 0; subsegment < div_result.quot; subsegment++) {
                write_engine(reinterpret_cast<double *>(des_ptr + subsegment * buffer_size), buffer_data, buffer_size);
            }
            write_engine(reinterpret_cast<double *>(des_ptr + div_result.quot * buffer_size), buffer_data,
                         div_result.rem);
            // increase index at break index by 1 and carry surplus if index value exceeded shape
            index[break_index] += 1;
            std::int64_t update_dim = break_index;
            while ((index[update_dim] == target->shape()[update_dim]) && (update_dim > 0)) {
                index[update_dim] = 0;
                index[--update_dim] += 1;
            }
            // break if all element is looped
            if (index[0] == target->shape()[0]) {
                break;
            }
        }
    }
    // deallocate buffer
    delete[] reinterpret_cast<char *>(buffer_data);
}
*/
// String representation of an array
template <class NdArray>
std::string array::print(const NdArray * target, const std::string & nametype, bool first_call) {
    if (target->data() == nullptr) {
        FAILURE(std::runtime_error, "Cannot access elements of non-initialized array.\n");
    }
    std::ostringstream os;
    // trivial case
    if (target->ndim() == 1) {
        os << "<";
        if (first_call) {
            os << nametype << "(";
        }
        for (std::uint64_t i = 0; i < target->shape()[0]; i++) {
            if (i > 0) {
                os << " ";
            }
            os << target->get({i});
        }
        if (first_call) {
            os << ")";
        }
        os << ">";
        return os.str();
    }
    // recursively call str function of sub_array
    os << "<";
    if (first_call) {
        os << nametype << "(";
    }
    for (std::uint64_t i = 0; i < target->shape()[0]; i++) {
        if (i != 0) {
            os << " ";
        }
        slicevec slice_i(target->ndim());
        slice_i[0] = Slice({i});
        NdArray sliced_array = target->sub_array(slice_i);
        sliced_array.remove_dim(0);
        os << sliced_array.str(false);
    }
    if (first_call) {
        os << ")";
    }
    os << ">";
    return os.str();
}

}  // namespace merlin

#endif  // MERLIN_ARRAY_OPERATION_TPP_
