// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_PARCEL_HPP_
#define MERLIN_ARRAY_PARCEL_HPP_

#include <cstdint>  // std::uint64_t, std::uintptr_t

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Parcel
#include "merlin/array/nddata.hpp"       // merlin::array::NdData
#include "merlin/cuda/device.hpp"        // merlin::cuda::Device
#include "merlin/cuda/stream.hpp"        // merlin::cuda::Stream
#include "merlin/cuda_interface.hpp"     // __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"            // MERLIN_EXPORTS

namespace merlin {

/** @brief Multi-dimensional array on GPU.*/
class array::Parcel : public array::NdData {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor (do nothing).*/
    Parcel(void) = default;
    /** @brief Construct a contiguous array from shape on GPU.*/
    MERLIN_EXPORTS Parcel(const Index & shape, const cuda::Stream & stream = cuda::Stream());
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Deep copy constructor.*/
    MERLIN_EXPORTS Parcel(const array::Parcel & src);
    /** @brief Deep copy assignment.*/
    MERLIN_EXPORTS array::Parcel & operator=(const array::Parcel & src);
    /** @brief Move constructor.*/
    MERLIN_EXPORTS Parcel(array::Parcel && src);
    /** @brief Move assignment.*/
    MERLIN_EXPORTS array::Parcel & operator=(array::Parcel && src);
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get constant reference to ID of device containing data of a constant instance.*/
    __cuhostdev__ constexpr const cuda::Device & device(void) const noexcept { return this->device_; }
    /// @}

    /// @name Atributes
    /// @{
#ifdef __NVCC__
    /** @brief Get reference to element at a given ndim index.*/
    __cudevice__ double & operator[](const Index & index);
    /** @brief Get reference to element at a given C-contiguous index.*/
    __cudevice__ double & operator[](std::uint64_t index);
    /** @brief Get constant reference to element at a given ndim index.*/
    __cudevice__ const double & operator[](const Index & index) const;
    /** @brief Get const reference to element at a given C-contiguous index.*/
    __cudevice__ const double & operator[](std::uint64_t index) const;
#endif  // __NVCC__
    /// @}

    /// @name Get and set element
    /// @{
    /** @brief Get value of element at a n-dim index.
     *  @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS double get(const Index & index) const;
    /** @brief Get value of element at a C-contiguous index.
     *  @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS double get(std::uint64_t index) const;
    /** @brief Set value of element at a n-dim index.
     *  @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS void set(const Index & index, double value);
    /** @brief Set value of element at a C-contiguous index.
     *  @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS void set(std::uint64_t index, double value);
    /// @}

    /// @name Operations
    /// @{
    /** @brief Set value of all elements.*/
    MERLIN_EXPORTS void fill(double value);
    /** @brief Create a sub-array.*/
    array::NdData * sub_array(const slicevec & slices) const {
        array::Parcel * p_result = new array::Parcel();
        this->create_sub_array(*p_result, slices);
        p_result->device_ = this->device_;
        return p_result;
    }
    /// @}

    /// @name Transfer data to GPU
    /// @{
    /** @brief Transfer data to GPU from CPU array.
     *  @note Stream synchronization is not included in this function.
     */
    MERLIN_EXPORTS void transfer_data_to_gpu(const array::Array & cpu_array,
                                             const cuda::Stream & stream = cuda::Stream());
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the object and its data.*/
    std::uint64_t cumalloc_size(void) const noexcept { return sizeof(array::Parcel); }
    /** @brief Copy meta-data (shape and strides) from CPU to a pre-allocated memory on GPU.
     *  @details The meta-data should be to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param data_ptr Pointer to a pre-allocated GPU memory storing other data.
     *  @param stream_ptr Pointer to CUDA stream for asynchronous copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(array::Parcel * gpu_ptr, void * data_ptr, std::uintptr_t stream_ptr = 0) const;
    /** @brief Calculate the minimum number of bytes to allocate in CUDA shared memory to store the array.*/
    std::uint64_t sharedmem_size(void) const noexcept { return this->cumalloc_size(); }
#ifdef __NVCC__
    /** @brief Copy metadata to a pre-allocated memory region by a GPU block of threads.
     *  @details The copy action is performed by the whole CUDA thread block.
     *  @param dest_ptr Pre-allocated memory region storing the new object on GPU.
     *  @param data_ptr Pointer to a pre-allocated GPU memory storing other data.
     *  @param thread_idx Flatten ID of the current CUDA thread in the block.
     *  @param block_size Number of threads in the current CUDA block.
     */
    __cudevice__ void * copy_by_block(array::Parcel * dest_ptr, void * data_ptr, std::uint64_t thread_idx,
                                      std::uint64_t block_size) const;
    /** @brief Copy metadata to a pre-allocated memory region by a single GPU threads.
     *  @param dest_ptr Memory region where the new object resides.
     *  @param @param data_ptr Pointer to a pre-allocated GPU memory storing other data.
     */
    __cudevice__ void * copy_by_thread(array::Parcel * dest_ptr, void * data_ptr) const;
#endif  // __NVCC__
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(bool first_call = true) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Free current data hold by the object.
     *  @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS void free_current_data(const cuda::Stream & stream = cuda::Stream());
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Parcel(void);
    /// @}

  protected:
    /** @brief Device containing data of Parcel.*/
    cuda::Device device_;
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_PARCEL_HPP_
