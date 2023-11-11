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

/** @brief Synchronizer of CPU or GPU asynchronous tasks.*/
struct Synchronizer {
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Synchronizer(void) = default;
    /** @brief Constructor from CPU synchronizer.*/
    Synchronizer(std::future<void> * cpu_sync) :
    proc_type(ProcessorType::Cpu), synchronizer(std::in_place_type<std::future<void> *>, cpu_sync) {}
    /** @brief Constructor from GPU synchronizer.*/
    Synchronizer(cuda::Stream && gpu_sync) :
    proc_type(ProcessorType::Gpu),
    synchronizer(std::in_place_type<cuda::Stream>, std::forward<cuda::Stream>(gpu_sync)) {}
    /// @}

    /// Copy and move
    /// @{
    /** @brief Copy constructor (deleted).*/
    Synchronizer(const Synchronizer & src) = delete;
    Synchronizer & operator=(const Synchronizer & src) = delete;
    Synchronizer(Synchronizer && src) :
    proc_type(src.proc_type),
    synchronizer(std::forward<std::variant<std::future<void> *, cuda::Stream>>(src.synchronizer)) {
        if (src.proc_type == ProcessorType::Cpu) {
            std::get<std::future<void> *>(src.synchronizer) = nullptr;
        }
    }
    Synchronizer & operator=(Synchronizer && src) {
        this->proc_type = src.proc_type;
        this->synchronizer = std::move(src.synchronizer);
        if (src.proc_type == ProcessorType::Cpu) {
            std::get<std::future<void> *>(src.synchronizer) = nullptr;
        }
        return *this;
    }
    /// @}

    /// @name Action
    /// @{
    void synchronize(void) {
        switch (this->proc_type) {
            case ProcessorType::Cpu : {
                std::future<void> * synch = std::get<std::future<void> *>(this->synchronizer);
                if (synch != nullptr) {
                    if (synch->valid()) {
                        synch->get();
                    }
                }
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

    /// @name Destructor
    /// @{
    ~Synchronizer(void) {
        if (this->proc_type == ProcessorType::Cpu) {
            std::future<void> * cpu_synchronizer = std::get<std::future<void> *>(this->synchronizer);
            if (cpu_synchronizer != nullptr) {
                delete cpu_synchronizer;
            }
        }
    }
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_SYNCHRONIZER_HPP_
