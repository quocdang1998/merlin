// Copyright 2022 quocdang1998
#ifndef MERLIN_ENV_HPP_
#define MERLIN_ENV_HPP_

#include <cstdint>  // std::uintptr_t
#include <map>      // std::map
#include <mutex>    // std::mutex
#include <random>   // std::mt19937_64

#include "merlin/exports.hpp"  // MERLINENV_EXPORTS

namespace merlin {

/** @brief Execution environment of merlin.*/
class Environment {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    MERLINENV_EXPORTS Environment(void);
    /// @}

    /// @name Shared variables
    /// @{
    /** @brief Mutex for locking threads.*/
    MERLINENV_EXPORTS static std::mutex mutex;
    /** @brief Random generator.*/
    MERLINENV_EXPORTS static std::mt19937_64 random_generator;
    /// @}

    /// @name CUDA environment
    /// @{
    /** @brief Check if CUDA environment is initialized or not.*/
    MERLINENV_EXPORTS static bool is_cuda_initialized;
    /** @brief Initialize CUDA context.*/
    MERLINENV_EXPORTS static void init_cuda(void);
    /** @brief Primary context of each GPU.*/
    MERLINENV_EXPORTS static std::map<int, std::uintptr_t> primary_ctx;
    /// @}
};

/** @brief Default environment.*/
MERLINENV_EXPORTS extern Environment default_env;

/** @brief Throw an error if CUDA environment has not been initialized.*/
MERLINENV_EXPORTS void check_cuda_env(void);

}  // namespace merlin

#endif  // MERLIN_ENV_HPP_
