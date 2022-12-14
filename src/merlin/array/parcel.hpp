// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_PARCEL_HPP_
#define MERLIN_ARRAY_PARCEL_HPP_

#include <cstdint>  // std::uint64_t, std::uintptr_t
#include <mutex>  // std::mutex

#include "merlin/array/nddata.hpp"  // merlin::array::Array, merlin::array::NdData
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/cuda_decorator.hpp"  // __cudevice__, __cuhostdev__
#include "merlin/cuda/gpu_query.hpp"  // merlin::cuda::Device
#include "merlin/cuda/stream.hpp"  // merlin::cuda::Stream
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

/** @brief Multi-dimensional array on GPU.*/
class MERLIN_EXPORTS array::Parcel : public array::NdData {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor (do nothing).*/
    __cuhostdev__ Parcel(void) {}
    /** @brief Construct array from CPU array.*/
    Parcel(const array::Array & cpu_array, const cuda::Stream & stream = cuda::Stream());
    /** @brief Constructor from a slice.
     *  @param whole merlin::array::NdData of the original array.
     *  @param slices List of merlin::array::Slice on each dimension.
     */
    Parcel(const array::Parcel & whole, const Vector<array::Slice> & slices);
    /** @brief Construct a contiguous array from shape on GPU.*/
    Parcel(const intvec & shape);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Deep copy constructor.*/
    Parcel(const array::Parcel & src);
    /** @brief Deep copy assignment.*/
    array::Parcel & operator=(const array::Parcel & src);
    /** @brief Move constructor.*/
    Parcel(array::Parcel && src);
    /** @brief Move assignment.*/
    array::Parcel & operator=(array::Parcel && src);
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get reference to ID of device containing data.*/
    cuda::Device & device(void) {return this->device_;}
    /** @brief Get constant reference to ID of device containing data of a constant instance.*/
    const cuda::Device & device(void) const {return this->device_;}
    /// @}

    /// @name Atributes
    /// @{
    #ifdef __NVCC__
    /** @brief Get element at a given C-contiguous index.
     *  @param index A C-contiguous index.
     */
    __cudevice__ float & operator[](std::uint64_t index);
    /** @brief Get element at a given multi-dimensional index.
     *  @param index A Nd index.
     */
    __cudevice__ float & operator[](std::initializer_list<std::uint64_t> index);
    #endif  // __NVCC__
    /// @}

    /// @name Get and set element
    /// @{
    /** @brief Get value of element at a n-dim index.*/
    float get(const intvec & index) const;
    /** @brief Get value of element at a C-contiguous index.*/
    float get(std::uint64_t index) const;
    /** @brief Set value of element at a n-dim index.*/
    void set(const intvec index, float value);
    /** @brief Set value of element at a C-contiguous index.*/
    void set(std::uint64_t index, float value);
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the object and its data.*/
    std::uint64_t malloc_size(void) {return sizeof(Parcel) + 2*this->ndim_*sizeof(std::uint64_t);}
    /** @brief Copy meta-data (shape and strides) from CPU to a pre-allocated memory on GPU.
     *  @details The meta-data should be to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param shape_strides_ptr Pointer to a pre-allocated GPU memory of size ``2*ndim``, storing data of shape and
     *  stride vector.
     */
    void copy_to_gpu(array::Parcel * gpu_ptr, void * shape_strides_ptr);
    #ifdef __NVCC__
    /** @brief Copy meta-data from GPU global memory to shared memory of a kernel.
     *  @note This operation is single-threaded.
     *  @param share_ptr Dynamically allocated shared pointer on GPU.
     *  @param shape_strides_ptr Pointer to a pre-allocated GPU memory of size ``2*ndim``, storing data of shape and
     *  stride vector.
     */
    __cudevice__ void copy_to_shared_mem(array::Parcel * share_ptr, void * shape_strides_ptr);
    #endif  // __NVCC__

    /// @name Utils
    /// @{
    /** @brief Free current data hold by the object.*/
    __cuhostdev__ void free_current_data(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Parcel(void);
    /// @}

  protected:
    /** @brief Decision to delete Array::data_ at destruction or not.*/
    bool force_free = true;
    /** @brief Device containing data of Parcel.*/
    cuda::Device device_;
    /** @brief Mutex lock at destruction time.*/
    static std::mutex mutex_;
};

}  // namespace merlin::array

#endif  // MERLIN_ARRAY_PARCEL_HPP_
