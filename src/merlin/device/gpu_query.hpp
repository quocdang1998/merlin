// Copyright 2022 quocdang1998
#ifndef MERLIN_DEVICE_GPU_QUERY_HPP_
#define MERLIN_DEVICE_GPU_QUERY_HPP_

#include <cstdint>  // std::uint64_t
#include <map>  // std::map

#include "merlin/device/decorator.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::device {

/** @brief Get ID of current active device.*/
__cuhostdev__ int get_current_gpu(void);

/** @brief Class representing CPU device.*/
class MERLIN_EXPORTS Device {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor from GPU ID.*/
    __cuhostdev__ Device(int id = -1);
    /// @}

    /// @name Copy and Move
    /// @details Move constructor and Move assignment are deleted.
    /// @{
    /** @brief Copy constructor.*/
    __cuhostdev__ Device(const Device & src) {this->id_ = src.id_;}
    /** @brief Copy assignment.*/
    __cuhostdev__ Device & operator=(const Device & src) {
        this->id_ = src.id_;
        return *this;
    }
    /// @}

    /// @name GPU query
    /// @{
    /** @brief Get total number of CUDA capable GPU.*/
    __cuhostdev__ static int get_num_gpu(void);
    /** @brief Print GPU specifications.*/
    void print_specification(void);
    /** @brief Test functionality of GPU.
     *  @details This function tests if the installed CUDA is compatible with the GPU driver by perform an addition of two
     *  integers on GPU.
     */
    bool test_gpu(void);
    /// @}

    /// @name GPU action
    /// @{
    /** @brief Reset GPU.
     *  @details Destroy all allocations and reset all state on the current device in the current process.
     */
    static void reset_all(void);
    /// @}

  private:
    /** @brief ID of device.*/
    int id_;
};

/** @brief Print GPU specifications.
 *  @details Print GPU specifications (number of threads, total global memory, max shared memory) and API limitation (max
 *  thread per block, max block per grid).
 *  @param device ID of device (an integer ranging from 0 to the number of device). A value of `-1` will print details of
 *  all GPU.
 */
MERLIN_EXPORTS void print_all_gpu_specification(void);

/** @brief Perform an addition of two integers on GPU.
 *  @details This function tests if the installed CUDA is compatible with the GPU.
 *  @return ``true`` if all tests on all GPU pass.
 */
MERLIN_EXPORTS bool test_all_gpu(void);

/** @brief Map from GPU ID to is details.*/
MERLIN_EXPORTS extern std::map<int, Device> gpu_map;

}  // namespace merlin::device

#endif  // MERLIN_DEVICE_GPU_QUERY_HPP_
