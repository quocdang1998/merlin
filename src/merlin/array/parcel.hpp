// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_PARCEL_HPP_
#define MERLIN_ARRAY_PARCEL_HPP_

#include <cstdint>  // std::uint64_t, std::uintptr_t
#include <mutex>  // std::mutex

#include "merlin/array/nddata.hpp"  // merlin::array::Array, merlin::array::NdData
#include "merlin/cuda_decorator.hpp"  // __cudevice__, __cuhostdev__
#include "merlin/cuda/device.hpp"  // merlin::cuda::Device
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
    /** @brief Construct a contiguous array from shape on GPU.*/
    Parcel(const intvec & shape);
    /** @brief Constructor from a slice.
     *  @param whole merlin::array::NdData of the original array.
     *  @param slices List of merlin::array::Slice on each dimension.
     */
    __cuhostdev__ Parcel(const array::Parcel & whole, const Vector<array::Slice> & slices);
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
    /** @brief Get constant reference to ID of device containing data of a constant instance.*/
    __cuhostdev__ constexpr const cuda::Device & device(void) const noexcept {return this->device_;}
    /// @}

    /// @name Atributes
    /// @{
    #ifdef __NVCC__
    /** @brief Get element at a given C-contiguous index.
     *  @param index A C-contiguous index.
     */
    __cudevice__ double & operator[](std::uint64_t index);
    #endif  // __NVCC__
    /// @}

    /// @name Get and set element
    /// @{
    /** @brief Get value of element at a n-dim index.*/
    double get(const intvec & index) const;
    /** @brief Get value of element at a C-contiguous index.*/
    double get(std::uint64_t index) const;
    /** @brief Set value of element at a n-dim index.*/
    void set(const intvec index, double value);
    /** @brief Set value of element at a C-contiguous index.*/
    void set(std::uint64_t index, double value);
    /// @}

    /// @name Transfer data to GPU
    /// @{
    /** @brief Transfer data to GPU from CPU array.
     *  @note Stream synchronization is not included in this function.
     */
    void transfer_data_to_gpu(const array::Array & cpu_array, const cuda::Stream & stream = cuda::Stream());
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the object and its data.*/
    std::uint64_t malloc_size(void) const {return sizeof(array::Parcel) + 2*this->ndim_*sizeof(std::uint64_t);}
    /** @brief Copy meta-data (shape and strides) from CPU to a pre-allocated memory on GPU.
     *  @details The meta-data should be to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param shape_strides_ptr Pointer to a pre-allocated GPU memory of size ``2*ndim``, storing data of shape and
     *  stride vector.
     */
    void * copy_to_gpu(array::Parcel * gpu_ptr, void * shape_strides_ptr) const;
    #ifdef __NVCC__
    /** @brief Copy meta-data from GPU global memory to shared memory of a kernel.
     *  @note This operation is single-threaded.
     *  @param share_ptr Dynamically allocated shared pointer on GPU.
     *  @param shape_strides_ptr Pointer to a pre-allocated GPU memory of size ``2*ndim``, storing data of shape and
     *  stride vector.
     */
    __cudevice__ void * copy_to_shared_mem(array::Parcel * share_ptr, void * shape_strides_ptr);
    #endif  // __NVCC__

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Parcel(void);
    /// @}

  protected:
    /** @brief Device containing data of Parcel.*/
    cuda::Device device_;
    /** @brief Mutex lock at destruction time.*/
    static std::mutex & mutex_;

  private:
    /** @brief Free current data hold by the object.*/
    void free_current_data(void);
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_PARCEL_HPP_
