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
    Environment(void) = default;
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
    /** @brief Primary context of each GPU.*/
    MERLINENV_EXPORTS static std::map<int, std::uintptr_t> primary_ctx;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_ENV_HPP_
