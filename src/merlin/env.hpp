// Copyright 2022 quocdang1998
#ifndef MERLIN_ENV_HPP_
#define MERLIN_ENV_HPP_

#include <atomic>  // std::atomic_uint64_t
#include <cstdint>  // std::uintptr_t
#include <map>  // std::map
#include <mutex>  // std::mutex

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

    /// @name Environment class-bounded properties
    /// @{
    /** @brief Check if the environment is initialized or not.*/
    static bool is_initialized;
    /** @brief Mutex for locking threads.*/
    static std::mutex mutex;
    /// @}

    /// @name Array allocation limit
    /// @{
    /** @brief Memory limit of a process for allocating ``merlin::array::Array``.
     *  @details Default value: 20GB.
     */
    static std::uint64_t cpu_mem_limit;
    /// @}

    /// @name CPU parallelism
    /// @{
    /** @brief Minimum size over which the loop is parallelized.
     *  @details Default value: 96 (LCM of 24, 32 and 48).
     */
    static std::uint64_t parallel_chunk;
    /// @}

    /// @name CUDA related settings
    /// @{
    /** @brief ID of default GPU.*/
    static int default_gpu;
    /** @brief Attributes of the context.*/
    struct MERLINSHARED_EXPORTS ContextAttribute {
        /// @name Constructor
        /// @{
        /** @brief Default constructor.*/
        ContextAttribute(void) = default;
        /** @brief Member constructor.*/
        ContextAttribute(std::uint64_t ref_count, int gpu_id);
        /// @}

        /// @name Copy and move
        /// @{
        /** @brief Copy constructor.*/
        ContextAttribute(const Environment::ContextAttribute & src);
        /** @brief Copy assignment.*/
        Environment::ContextAttribute & operator=(const Environment::ContextAttribute & src);
        /// @}

        /// @name Members
        /// @{
        /** @brief Reference count of the current context.
         *  @details Number of instances representing the same context.
         */
        std::atomic_uint64_t reference_count;
        /** @brief GPU of the current context.
         *  @details GPU device binded to the context.
         */
        int gpu;
        /// @}

        /// @name Destructor
        /// @{
        /** @brief Destructor.*/
        ~ContextAttribute(void) = default;
        /// @}
    };
    /** @brief Map from context pointers to their attributes.*/
    static std::map<std::uintptr_t, Environment::ContextAttribute> attribute;
    /** @brief CUDA primary contexts.*/
    static std::map<int, std::uintptr_t> primary_contexts;
    /** @brief Default CUDA kernel block size.
     *  @details Should be multiple of 32.
     */
    static std::uint64_t default_block_size;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Environment(void) = default;
    /// @}
};

/** @brief Default environment.*/
MERLINSHARED_EXPORTS extern Environment default_environment;

/** @brief Initialize CUDA primary contexts.*/
void initialize_cuda_context(void);

}  // namespace merlin

#endif  // MERLIN_ENV_HPP_
