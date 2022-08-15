// Copyright 2022 quocdang1998
#ifndef MERLIN_PARCEL_HPP_
#define MERLIN_PARCEL_HPP_

#include <cstdint>  // uintptr_t

#include "merlin/array.hpp"  // Array

namespace merlin {

/** @brief Multi-dimensional array on GPU.*/
class Parcel : public Array {
  public:
    // Constructors
    // ------------
    /** @brief Default constructor (do nothing).*/
    Parcel(void);
    /** @brief Construct array from CPU array.*/
    Parcel(const Tensor & cpu_array, uintptr_t stream = 0);

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
    /** @brief Get ID of device containing data.*/
    int & device_id(void) {return this->device_id_;}
    /** @brief Get ID of device containing data of a constant instance.*/
    const int & device_id(void) const {return this->device_id_;}

    // Attributes
    // ----------
    #ifdef __NVCC__
    /** @brief Check if current device is the one holding Parcel data.
     *  @return ID of GPU holding value - ID of current GPU.
     */
    __host__ __device__ int check_device(void) const;
    /** @brief Get element at a given contiguous index.*/
    __device__ float & operator[](unsigned int index);
    #endif  // __NVCC__

    // Utils
    // -----
    /** @brief Free current data hold by the object.*/
    void free_current_data(void);
    /** @brief Update the shape vector and strides vector on GPU memory.*/
    void copy_metadata(void);

    // Destructor
    // ----------
    /** @brief Destructor.*/
    ~Parcel(void);

  protected:
    // Members
    // -------
    /** @brief Device containing data of Parcel.*/
    int device_id_;
    /** @brief Shape vector on device memory.*/
    unsigned int * dshape_ = NULL;
    /** @brief Stride vector on device memory.*/
    unsigned int * dstrides_ = NULL;
};

}  // namespace merlin

#endif  // MERLIN_PARCEL_HPP_
