// Copyright 2023 quocdang1998
#ifndef MERLIN_THREAD_DIVIDER_HPP_
#define MERLIN_THREAD_DIVIDER_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/cuda_interface.hpp"  // __cuhostdev__

namespace merlin {

/** @brief Dividing threads into groups for parallelizing tasks.*/
struct ThreadDivider {
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ ThreadDivider(void) {}
    /** @brief Constructor from number of tasks, index of the current thread and the total number of threads.*/
    __cuhostdev__ ThreadDivider(const std::uint64_t & num_task, const std::uint64_t & thread_idx,
                                const std::uint64_t & num_threads) {
        if (num_task >= num_threads) {
            // when number of task is bigger than number of thread
            this->num_groups = num_threads;
            this->numthreads_pertask = 1;
            this->group_idx = thread_idx;
            this->thread_idx_in_group = 0;
        } else {
            // when number of thread is bigger than number of task
            this->num_groups = num_task;
            std::uint64_t extra_thread = num_threads % num_task;
            std::uint64_t equi_partition = num_threads / num_task;
            bool is_inextra = (thread_idx < extra_thread * (equi_partition + 1));
            this->numthreads_pertask = (is_inextra) ? (equi_partition + 1) : equi_partition;
            std::uint64_t effective_thread_idx = (is_inextra) ? thread_idx : (thread_idx - extra_thread);
            this->group_idx = effective_thread_idx / this->numthreads_pertask;
            this->thread_idx_in_group = effective_thread_idx % this->numthreads_pertask;
        }
    }
    /// @}

    /// @name Attributes
    /// @{
    /** @brief Number of threads per task.*/
    std::uint64_t numthreads_pertask;
    /** @brief Number of thread groups.*/
    std::uint64_t num_groups;
    /** @brief Index of the group containing the current thread.*/
    std::uint64_t group_idx;
    /** @brief Index of the thread in the current group.*/
    std::uint64_t thread_idx_in_group;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_THREAD_DIVIDER_HPP_
