// Copyright 2022 quocdang1998
#ifndef MERLIN_PARCEL_HPP_
#define MERLIN_PARCEL_HPP_

#include <cstdint>  // std::uintptr_t
#include <initializer_list>  // std::initializer_list

#include "merlin/nddata.hpp"  // merlin::NdData
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/decorator.hpp"  // __cudevice__, __cuhostdev__
#include "merlin/utils.hpp"

namespace merlin {

/** @brief Multi-dimensional array on GPU.*/
class MERLIN_EXPORTS Parcel : public NdData {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor (do nothing).*/
    Parcel(void);
    /** @brief Construct array from CPU array.*/
    Parcel(const Array & cpu_array, std::uintptr_t stream = 0);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Deep copy constructor.*/
    Parcel(const Parcel & src);
    /** @brief Deep copy assignment.*/
    Parcel & operator=(const Parcel & src);
    /** @brief Move constructor.*/
    Parcel(Parcel && src);
    /** @brief Move assignment.*/
    Parcel & operator=(Parcel && src);
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get reference to ID of device containing data.*/
    int & device_id(void) {return this->device_id_;}
    /** @brief Get constant reference to ID of device containing data of a constant instance.*/
    const int & device_id(void) const {return this->device_id_;}
    /// @}

    /// @name Atributes
    /// @{
    /** @brief Check if current device is the one holding Parcel data.
     *  @return ID of GPU holding value - ID of current GPU.
     */
    int check_device(void) const;
    #ifdef __NVCC__
    /** @brief Get element at a given C-contiguous index.
     *  @param index A C-contiguous index.
     */
    __cudevice__ inline float & operator[](std::uint64_t index);
    /** @brief Get element at a given multi-dimensional index.
     *  @param index A Nd index.
     */
    __cudevice__ inline float & operator[](std::initializer_list<std::uint64_t> index);
    #endif  // __NVCC__
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the object and its data.*/
    std::uint64_t malloc_size(void) {return sizeof(Parcel) + 2*this->ndim_*sizeof(std::uint64_t);}
    /** @brief Copy meta-data (shape and strides) from CPU to a pre-allocated memory on GPU.
     *  @details The meta-data is copied to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param shape_strides_ptr Pointer to a pre-allocated GPU memory of size ``2*ndim``, storing data of shape and stride
     *  vector.
     */
    void copy_to_gpu(Parcel * gpu_ptr, std::uint64_t * shape_strides_ptr);
    #ifdef __NVCC__
    /** @brief Copy meta-data from GPU global memory to shared memory of a kernel.
     *  @note This operation is single-threaded.
     *  @param share_ptr Dynamically allocated shared pointer on GPU.
     *  @param shape_strides_ptr Pointer to a pre-allocated GPU memory of size ``2*ndim``, storing data of shape and stride
     *  vector.
     */
    __cudevice__ inline void copy_to_shared_mem(Parcel * share_ptr, std::uint64_t * shape_strides_ptr);
    #endif  // __NVCC__

    /// @name Utils
    /// @{
    /** @brief Free current data hold by the object.*/
    void free_current_data(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Parcel(void);
    /// @}

  protected:
    // Members
    // -------
    /** @brief Device containing data of Parcel.*/
    int device_id_;
};

#ifdef __NVCC__
// Get element at a given C-contiguous index
__cudevice__ inline float & Parcel::operator[](std::uint64_t index) {
    // calculate index vector
    intvec index_ = contiguous_to_ndim_idx(index, this->shape_);
    // calculate strides
    std::uint64_t strides = inner_prod(index_, this->strides_);
    float * element_ptr = reinterpret_cast<float *>(reinterpret_cast<std::uintptr_t>(this->data_) + strides);
    return *element_ptr;
}

// Get element at a given Nd index
__cudevice__ inline float & Parcel::operator[](std::initializer_list<std::uint64_t> index) {
    // initialize index vector
    intvec index_(index);
    // calculate strides
    std::uint64_t strides = inner_prod(index_, this->strides_);
    float * element_ptr = reinterpret_cast<float *>(reinterpret_cast<std::uintptr_t>(this->data_) + strides);
    return *element_ptr;
}

// Copy to shared memory
__cudevice__ inline void Parcel::copy_to_shared_mem(Parcel * share_ptr, std::uint64_t * shape_strides_ptr) {
    // copy meta data
    share_ptr->data_ = this->data_;
    share_ptr->ndim_ = this->ndim_;
    // copy shape and strides
    this->shape_.copy_to_shared_mem(&(share_ptr->shape_), shape_strides_ptr);
    this->strides_.copy_to_shared_mem(&(share_ptr->strides_), shape_strides_ptr+this->ndim_);
}
#endif  // __NVCC__

}  // namespace merlin

#endif  // MERLIN_PARCEL_HPP_
