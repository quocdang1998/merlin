// Copyright 2022 quocdang1998
#ifndef MERLIN_ENV_HPP_
#define MERLIN_ENV_HPP_

#include <atomic>   // std::atomic_uint, std::atomic_uint64_t
#include <cstdint>  // std::uintptr_t
#include <map>      // std::map
#include <mutex>    // std::mutex
#include <random>   // std::mt19937_64
#include <utility>  // std::pair
#include <vector>   // std::vector

#include "merlin/exports.hpp"  // MERLINSHARED_EXPORTS

namespace merlin {

/** @brief Execution environment of merlin.*/
class MERLINSHARED_EXPORTS Environment {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Environment(void);
    /// @}

    /// @name Shared variables
    /// @{
    /** @brief Check if the environment is initialized or not.*/
    static bool is_initialized;
    /** @brief Number of Environment instances created.*/
    static std::atomic_uint num_instances;
    /** @brief Mutex for locking threads.*/
    static std::mutex mutex;
    /** @brief Random generator.*/
    static std::mt19937_64 random_generator;
    /// @}

    /** @brief Default CUDA kernel block size.
     *  @details Should be multiple of 32.
     */
    static std::uint64_t default_block_size;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Environment(void);
    /// @}
};

/** @brief Default environment.*/
MERLINSHARED_EXPORTS extern Environment default_environment;

/** @brief Initialize the default CUDA context.*/
void initialize_cuda_context(void);

/** @brief Alarm for CUDA error.
 *  @details Double-check if CUDA operations have an unthrown error.
 */
void alarm_cuda_error(void);

}  // namespace merlin

#endif  // MERLIN_ENV_HPP_
