// Copyright 2023 quocdang1998
#ifndef MERLIN_SYNCHRONIZER_HPP_
#define MERLIN_SYNCHRONIZER_HPP_

#include <future>   // std::shared_future
#include <utility>  // std::in_place_type, std::forward, std::move
#include <variant>  // std::variant

#include "merlin/cuda/stream.hpp"  // merlin::cuda::Stream

namespace merlin {

/** @brief Type of processor.*/
enum class ProcessorType : unsigned int {
    /** @brief CPU.*/
    Cpu = 0,
    /** @brief GPU.*/
    Gpu = 1
};

/** @brief Synchronizer of CPU or GPU asynchronous tasks.*/
struct Synchronizer {
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Synchronizer(void) = default;
    /** @brief Constructor from CPU synchronizer.*/
    Synchronizer(std::shared_future<void> && cpu_sync) :
    synchronizer(std::in_place_type<std::shared_future<void>>, std::forward<std::shared_future<void>>(cpu_sync)) {}
    /** @brief Constructor from GPU synchronizer.*/
    Synchronizer(cuda::Stream && gpu_sync) :
    synchronizer(std::in_place_type<cuda::Stream>, std::forward<cuda::Stream>(gpu_sync)) {}
    /// @}

    /// Copy and move
    /// @{
    /** @brief Copy constructor (deleted).*/
    Synchronizer(const Synchronizer & src) = delete;
    /** @brief Copy assignment (deleted).*/
    Synchronizer & operator=(const Synchronizer & src) = delete;
    /** @brief Move constructor.*/
    Synchronizer(Synchronizer && src) = default;
    /** @brief Move assignment.*/
    Synchronizer & operator=(Synchronizer && src) = default;
    /// @}

    /// @name Action
    /// @{
    void synchronize(void) {
        switch (this->synchronizer.index()) {
            case 0 : {
                std::shared_future<void> & cpu_sync = std::get<std::shared_future<void>>(this->synchronizer);
                if (cpu_sync.valid()) {
                    cpu_sync.get();
                }
                break;
            }
            case 1 : {
                std::get<cuda::Stream>(this->synchronizer).synchronize();
                break;
            }
        }
    }
    /// @}

    /// @name Attributes
    /// @{
    /** @brief Synchronizer.*/
    std::variant<std::shared_future<void>, cuda::Stream> synchronizer;
    /// @}

    /// @name Destructor
    /// @{
    ~Synchronizer(void) = default;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_SYNCHRONIZER_HPP_
