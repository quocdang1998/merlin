// Copyright 2023 quocdang1998
#ifndef MERLIN_SYNCHRONIZER_HPP_
#define MERLIN_SYNCHRONIZER_HPP_

#include <future>   // std::future
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

/** @brief %Synchronizer of CPU or GPU asynchronous tasks.*/
struct Synchronizer {
    /// @name Constructor
    /// @{
    /** @brief Constructor from processor type.*/
    Synchronizer(ProcessorType proc_type = ProcessorType::Cpu) {
        if (proc_type == ProcessorType::Cpu) {
            this->core = std::variant<std::future<void>, cuda::Stream>(std::in_place_type<std::future<void>>,
                                                                       std::future<void>());
        } else {
            this->core = std::variant<std::future<void>, cuda::Stream>(std::in_place_type<cuda::Stream>,
                                                                       cuda::Stream(cuda::StreamSetting::NonBlocking));
        }
    }
    /** @brief Constructor from CPU synchronizer.*/
    Synchronizer(std::future<void> && cpu_sync) :
    core(std::in_place_type<std::future<void>>, std::forward<std::future<void>>(cpu_sync)) {}
    /** @brief Constructor from GPU synchronizer.*/
    Synchronizer(cuda::Stream && gpu_sync) :
    core(std::in_place_type<cuda::Stream>, std::forward<cuda::Stream>(gpu_sync)) {}
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
    /** @brief Halt the current thread until the asynchronous action registered on the synchronizer finished.*/
    void synchronize(void) {
        switch (this->core.index()) {
            case 0 : {
                std::future<void> & cpu_sync = std::get<std::future<void>>(this->core);
                if (cpu_sync.valid()) {
                    cpu_sync.get();
                }
                break;
            }
            case 1 : {
                std::get<cuda::Stream>(this->core).synchronize();
                break;
            }
        }
    }
    /// @}

    /// @name Attributes
    /// @{
    /** @brief %Synchronizer core.*/
    std::variant<std::future<void>, cuda::Stream> core;
    /// @}

    /// @name Destructor
    /// @{
    ~Synchronizer(void) = default;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_SYNCHRONIZER_HPP_
