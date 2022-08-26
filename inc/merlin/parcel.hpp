// Copyright 2022 quocdang1998
#ifndef MERLIN_PARCEL_HPP_
#define MERLIN_PARCEL_HPP_

#include <cstdint>  // uintptr_t

#include "merlin/nddata.hpp"  // merlin::NdData
#include "merlin/decorator.hpp"  // __cudevice__, __cuhostdev__

namespace merlin {

/** @brief Multi-dimensional array on GPU.*/
class Parcel : public NdData {
  public:
    // Constructors
    // ------------
    /** @brief Default constructor (do nothing).*/
    Parcel(void);
    /** @brief Construct array from CPU array.*/
    Parcel(const Array & cpu_array, uintptr_t stream = 0);

    // Copy and Move
    // -------------
    /** @brief Deep copy constructor.*/
    Parcel(const Parcel & src);
    /** @brief Deep copy assignment.*/
    Parcel & operator=(const Parcel & src);
    /** @brief Move constructor.*/
    Parcel(Parcel && src);
    /** @brief Move assignment.*/
    Parcel & operator=(Parcel && src);

    // Get members
    // -----------
    /** @brief Get reference to ID of device containing data.*/
    int & device_id(void) {return this->device_id_;}
    /** @brief Get constant reference to ID of device containing data of a constant instance.*/
    const int & device_id(void) const {return this->device_id_;}

    // Attributes
    // ----------
    #ifdef __NVCC__
    /** @brief Check if current device is the one holding Parcel data.
     *  @return ID of GPU holding value - ID of current GPU.
     */
    __cuhostdev__ int check_device(void) const;
    /** @brief Get element at a given contiguous index.*/
    __cudevice__ float & operator[](unsigned int index) {return this->data_[index];}
    #endif  // __NVCC__

    // GPU related features
    // --------------------
    unsigned long int malloc_size(void) {return sizeof(Parcel) + 2*this->ndim_*sizeof(unsigned long int);}
    void copy_to_device_ptr(Parcel * gpu_ptr);
    __cudevice__ inline void copy_to_shared_ptr(Parcel * share_ptr);


    // Utils
    // -----
    /** @brief Free current data hold by the object.*/
    void free_current_data(void);

    // Destructor
    // ----------
    /** @brief Destructor.*/
    ~Parcel(void);

  protected:
    // Members
    // -------
    /** @brief Device containing data of Parcel.*/
    int device_id_;
};

#ifdef __NVCC__
__cudevice__ inline void Parcel::copy_to_shared_ptr(Parcel * share_ptr) {
    // copy meta data
    share_ptr->data_ = this->data_;
    share_ptr->ndim_ = this->ndim_;
    // assign shape and strides pointer to data
    share_ptr->shape_.data() = (unsigned long int *) &share_ptr[1];
    share_ptr->shape_.size() = this->ndim_;
    share_ptr->strides_.data() = share_ptr->shape_.data() + this->ndim_;
    share_ptr->strides_.size() = this->ndim_;
    // copy shape and strides
    bool check_zeroth_thread = (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)
                            && (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
    if (check_zeroth_thread) {
        for (int i = 0; i < this->ndim_; i++) {
            share_ptr->shape_[i] = this->shape_[i];
            share_ptr->strides_[i] = this->strides_[i];
        }
    }
    __syncthreads();
}
#endif

}  // namespace merlin

#endif  // MERLIN_PARCEL_HPP_
