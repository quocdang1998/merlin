// Copyright 2023 quocdang1998
#ifndef MERLIN_SYNCHRONIZER_HPP_
#define MERLIN_SYNCHRONIZER_HPP_

#include <future>   // std::future
#include <utility>  // std::in_place_type, std::forward
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
    Synchronizer(std::future<void> * cpu_sync) :
        proc_type(ProcessorType::Cpu),
        synchronizer(std::in_place_type<std::future<void> *>, cpu_sync) {}
    /** @brief Constructor from GPU synchronizer.*/
    Synchronizer(cuda::Stream && gpu_sync) :
        proc_type(ProcessorType::Gpu),
        synchronizer(std::in_place_type<cuda::Stream>, std::forward<cuda::Stream>(gpu_sync)) {}
    /// @}

    /// @name Action
    /// @{
    void synchronize(void) {
        switch (this->proc_type) {
            case ProcessorType::Cpu : {
                std::get<std::future<void> *>(this->synchronizer)->get();
                break;
            }
            case ProcessorType::Gpu : {
                std::get<cuda::Stream>(this->synchronizer).synchronize();
                break;
            }
        }
    }
    /// @}

    /// @name Attributes
    /// @{
    /** @brief Processor type on which the synchronizer acts.*/
    ProcessorType proc_type;
    /** @brief Synchronizer.*/
    std::variant<std::future<void> *, cuda::Stream> synchronizer;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_SYNCHRONIZER_HPP_
