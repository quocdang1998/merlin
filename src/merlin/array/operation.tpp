// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_OPERATION_TPP_
#define MERLIN_ARRAY_OPERATION_TPP_

#include <algorithm>  // std::copy, std::fill_n, std::min, std::max
#include <sstream>    // std::ostringstream

#include "merlin/logger.hpp"  // merlin::Fatal, merlin::Warning
#include "merlin/utils.hpp"   // merlin::inner_prod, merlin::prod_elements

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Actions on NdData
// ---------------------------------------------------------------------------------------------------------------------

// Copy each segment from source to destination
template <class CopyFunction>
requires array::TransferFunction<CopyFunction>
void array::copy(array::NdData * dest, const array::NdData * src, CopyFunction copy) {
    // check if shape vector are the same
    if (src->ndim() != dest->ndim()) {
        Fatal<std::invalid_argument>("Cannot copy array of different ndim (%u to %u).\n", src->ndim(), dest->ndim());
    }
    std::uint64_t ndim = src->ndim();
    if (!(dest->shape() == src->shape())) {
        Fatal<std::invalid_argument>("Expected shape of source equals shape of destination.\n");
    }
    // trivial case: size zero
    if (ndim == 0) {
        return;
    }
    // longest contiguous segment and break index
    auto [src_lcs, src_bridx] = array::lcseg_and_brindex(src->shape(), src->strides(), ndim);
    auto [dest_lcs, dest_bridx] = array::lcseg_and_brindex(dest->shape(), dest->strides(), ndim);
    std::uint64_t longest_contiguous_segment = std::min(src_lcs, dest_lcs);
    std::int64_t break_index = std::max(src_bridx, dest_bridx);
    // copy each longest contiguous segment through the copy function
    if (break_index == -1) {  // original arrays are perfectly contiguous
        copy(dest->data(), src->data(), longest_contiguous_segment);
        return;
    }
    // otherwise, memcpy each longest_contiguous_segment
    std::uint64_t num_segments = prod_elements(src->shape().data(), ndim);
    for (std::uint64_t i_segment = 0; i_segment < num_segments; i_segment++) {
        // calculate ptr to each segment
        std::uint64_t src_leap = array::get_leap(i_segment, src->shape(), src->strides(), break_index + 1);
        std::uintptr_t src_ptr = reinterpret_cast<std::uintptr_t>(src->data()) + src_leap;
        std::uint64_t dest_leap = array::get_leap(i_segment, dest->shape(), dest->strides(), break_index + 1);
        std::uintptr_t dest_ptr = reinterpret_cast<std::uintptr_t>(dest->data()) + dest_leap;
        // copy the segment
        copy(reinterpret_cast<double *>(dest_ptr), reinterpret_cast<double *>(src_ptr), longest_contiguous_segment);
    }
}

// Fill all array with a given value
template <class CopyFunction, std::uint64_t buffer>
requires array::TransferFunction<CopyFunction>
void array::fill(array::NdData * target, double fill_value, CopyFunction write_engine) {
    // trivial case: size zero
    if (target->ndim() == 0) {
        return;
    }
    // allocate a buffer and fill it with data
    auto [lcseg, break_index] = array::lcseg_and_brindex(target->shape(), target->strides(), target->ndim());
    std::uint64_t buffer_size = std::min(lcseg, buffer * sizeof(double));
    double * buffer_data = reinterpret_cast<double *>(new char[buffer_size]);
    std::fill_n(buffer_data, buffer_size / sizeof(double), fill_value);
    std::uint64_t n_chunk = lcseg / buffer_size, remainder = lcseg % buffer_size;
    // memcpy each longest_contiguous_segment
    std::uint64_t num_segments = (break_index == -1) ? 1 : prod_elements(target->shape().data(), break_index + 1);
    for (std::uint64_t i_segment = 0; i_segment < num_segments; i_segment++) {
        // calculate ptr to each segment
        std::uint64_t leap = array::get_leap(i_segment, target->shape(), target->strides(), break_index + 1);
        std::uintptr_t target_data_ptr = reinterpret_cast<std::uintptr_t>(target->data()) + leap;
        // fill the segment
        for (std::uint64_t i_chunk = 0; i_chunk < n_chunk; i_chunk++) {
            write_engine(reinterpret_cast<double *>(target_data_ptr), buffer_data, buffer_size);
            target_data_ptr += buffer_size;
        }
        if (remainder > 0) {
            write_engine(reinterpret_cast<double *>(target_data_ptr), buffer_data, remainder);
        }
    }
    // deallocate buffer
    delete[] reinterpret_cast<char *>(buffer_data);
}

// Calculate mean and variance of an array
template <class CopyFunction, std::uint64_t buffer>
requires array::TransferFunction<CopyFunction>
std::array<double, 2> array::stat(const array::NdData * target, CopyFunction copy) {
    // trivial case: size zero
    if (target->ndim() == 0) {
        Warning("Zero-elements array do not have mean nor variance, return 0.\n");
        return {0.0, 0.0};
    }
    // allocate a buffer and fill it with data
    auto [lcseg, break_index] = array::lcseg_and_brindex(target->shape(), target->strides(), target->ndim());
    std::uint64_t buffer_size = std::min(lcseg, buffer * sizeof(double));
    double * buffer_data = reinterpret_cast<double *>(new char[buffer_size]);
    std::uint64_t n_chunk = lcseg / buffer_size, remainder = lcseg % buffer_size;
    std::uint64_t buffer_nelems = buffer_size / sizeof(double), remainder_nelems = remainder / sizeof(double);
    double total_mean = 0, total_var = 0, partial_mean, partial_var;
    std::uint64_t total_normal = 0, partial_normal;
    // memcpy each longest_contiguous_segment
    std::uint64_t num_segments = (break_index == -1) ? 1 : prod_elements(target->shape().data(), break_index + 1);
    for (std::uint64_t i_segment = 0; i_segment < num_segments; i_segment++) {
        // calculate ptr to each segment
        std::uint64_t leap = array::get_leap(i_segment, target->shape(), target->strides(), break_index + 1);
        std::uintptr_t target_data_ptr = reinterpret_cast<std::uintptr_t>(target->data()) + leap;
        // fill the segment
        for (std::uint64_t i_chunk = 0; i_chunk < n_chunk; i_chunk++) {
            copy(buffer_data, reinterpret_cast<double *>(target_data_ptr), buffer_size);
            array::calc_mean_variance(buffer_data, buffer_nelems, partial_mean, partial_var, partial_normal);
            array::combine_stas(total_mean, total_var, total_normal, partial_mean, partial_var, partial_normal);
            target_data_ptr += buffer_size;
        }
        // check remainder
        if (remainder > 0) {
            copy(buffer_data, reinterpret_cast<double *>(target_data_ptr), remainder);
            array::calc_mean_variance(buffer_data, remainder_nelems, partial_mean, partial_var, partial_normal);
            array::combine_stas(total_mean, total_var, total_normal, partial_mean, partial_var, partial_normal);
        }
    }
    // deallocate buffer
    delete[] reinterpret_cast<char *>(buffer_data);
    return {total_mean, total_var / total_normal};
}

// String representation of an array
template <class NdArray>
std::string array::print(const NdArray * target, const std::string & nametype, bool first_call) {
    if (target->data() == nullptr) {
        Fatal<std::runtime_error>("Cannot access elements of non-initialized array.\n");
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
        SliceArray slice_i;
        slice_i.fill(Slice());
        slice_i[0] = Slice({i});
        NdArray * p_sliced_array = static_cast<NdArray *>(target->sub_array(slice_i));
        p_sliced_array->remove_dim(0);
        os << p_sliced_array->str(false);
        delete p_sliced_array;
    }
    if (first_call) {
        os << ")";
    }
    os << ">";
    return os.str();
}

}  // namespace merlin

#endif  // MERLIN_ARRAY_OPERATION_TPP_
