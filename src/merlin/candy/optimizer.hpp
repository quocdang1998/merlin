// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTIMIZER_HPP_
#define MERLIN_CANDY_OPTIMIZER_HPP_

#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/cuda_decorator.hpp"  // __cuhostdev__
#include "merlin/vector.hpp"  // merlin::Vector
#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin {

/** @brief Base class for optimizer of model.*/
class candy::Optimizer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Optimizer(void) = default;
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    Optimizer(const candy::Optimizer & src) = default;
    /** @brief Copy assignment.*/
    candy::Optimizer & operator=(const candy::Optimizer & src) = default;
    /** @brief Move constructor.*/
    Optimizer(candy::Optimizer && src) = default;
    /** @brief Move assignment.*/
    candy::Optimizer & operator=(candy::Optimizer && src) = default;
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model by gradient.
     *  @param model Candecomp model to be trained.
     *  @param gradient Value of the gradient of the parameter.
     *  @param i_param Contiguous index of the parameter.
     *  @param param_dim Dimension of the current parameter.
     *  @param param_index Index of parameter in the model.
     *  @param param_rank Rank of the parameter in the model.
     */
    virtual void update_cpu(candy::Model & model, double gradient, std::uint64_t i_param, std::uint64_t param_dim,
                            std::uint64_t param_index, std::uint64_t param_rank) {}
    #ifdef __NVCC__
    /** @brief Update model by gradient value of current thread.*/
    __cudevice__ virtual void update_gpu(candy::Model * p_model, double gradient, std::uint64_t i_param,
                                         std::uint64_t param_dim, std::uint64_t param_index,
                                         std::uint64_t param_rank) {}
    #endif  // __NVCC__
    /// @}

    /// @name GPU related features
    /// @{
    #ifdef __NVCC__
    /** @brief Copy data to shared memory.*/
    __cudevice__ virtual void * copy_to_shared_mem(candy::Optimizer * share_ptr, void * data_ptr) const {
        return data_ptr;
    }
    #endif  // __NVCC__
    /// @}

    /// @name Destructor
    /// @{
    virtual __cuhostdev__ ~Optimizer(void);
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTIMIZER_HPP_
