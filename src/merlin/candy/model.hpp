// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_MODEL_HPP_
#define MERLIN_CANDY_MODEL_HPP_

#include <array>  // std::array
#include <cstdint>  // std::uint64_t
#include <string>  // std::string

#include "merlin/cuda_decorator.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin {

/** @brief Canonical decomposition model.*/
class candy::Model {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Model(void) = default;
    /** @brief Constructor from shape and rank.*/
    MERLIN_EXPORTS Model(const intvec & shape, std::uint64_t rank);
    /** @brief Constructor from model values.*/
    MERLIN_EXPORTS Model(const Vector<Vector<double>> & parameter, std::uint64_t rank);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    MERLIN_EXPORTS Model(const candy::Model & src) = default;
    /** @brief Copy assignment.*/
    MERLIN_EXPORTS candy::Model & operator=(const candy::Model & src) = default;
    /** @brief Move constructor.*/
    MERLIN_EXPORTS Model(candy::Model && src) = default;
    /** @brief Move assignment.*/
    MERLIN_EXPORTS candy::Model & operator=(candy::Model && src) = default;
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get constant reference to parameters.*/
    __cuhostdev__ constexpr const Vector<Vector<double>> & parameters(void) const noexcept {return this->parameters_;}
    /** @brief Number of dimension.*/
    __cuhostdev__ constexpr std::uint64_t ndim(void) const noexcept {return this->parameters_.size();}
    /** @brief Get shape.*/
    __cuhostdev__ intvec get_model_shape(std::uint64_t * data_ptr = nullptr) const noexcept;
    /** @brief Get rank.*/
    __cuhostdev__ constexpr std::uint64_t rank(void) const noexcept {return this->rank_;}
    /** @brief Size (number of elements).*/
    __cuhostdev__ std::uint64_t size(void) const noexcept;
    /// @}

    /// @name Get and set parameters
    /// @{
    /** @brief Get value of element at a given index.*/
    __cuhostdev__ constexpr double get(std::uint64_t i_dim, std::uint64_t index,
                                               std::uint64_t rank) const noexcept {
        return this->parameters_[i_dim][index*this->rank_ + rank];
    }
    /** @brief Set value of element at a given index.*/
    __cuhostdev__ constexpr void set(std::uint64_t i_dim, std::uint64_t index, std::uint64_t rank,
                                     double && value) noexcept {
        this->parameters_[i_dim][index*this->rank_ + rank] = value;
    }
    /** @brief Get dimension and index from contiguous index.*/
    __cuhostdev__ std::array<std::uint64_t, 2> convert_contiguous(std::uint64_t index) const noexcept;
    /** @brief Get reference to element from flattened index.*/
    __cuhostdev__ const double & get(std::uint64_t index) const noexcept;
    /** @brief Set value of element from flattened index.*/
    __cuhostdev__ void set(std::uint64_t index, double && value) noexcept;
    /// @}

    /// @name Evaluation of the model
    /// @{
    /** @brief Evaluate result of the model at a given index.*/
    __cuhostdev__ double eval(const intvec & index) const noexcept;
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the model and its data.*/
    MERLIN_EXPORTS std::uint64_t malloc_size(void) const;
    /** @brief Copy the model from CPU to a pre-allocated memory on GPU.
     *  @details Values of vectors should be copied to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param parameters_data_ptr Pointer to a pre-allocated GPU memory storing data of parameters.
     *  @param stream_ptr Pointer to CUDA stream for asynchronious copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(candy::Model * gpu_ptr, void * parameters_data_ptr,
                                      std::uintptr_t stream_ptr = 0) const;
    /** @brief Calculate the minimum number of bytes to allocate in CUDA shared memory to store the model.*/
    std::uint64_t shared_mem_size(void) const {
        return sizeof(candy::Model) + this->ndim() * sizeof(Vector<double>);
    }
    #ifdef __NVCC__
    /** @brief Copy meta-data from GPU global memory to shared memory of a kernel.
     *  @note This operation is single-threaded.
     *  @param share_ptr Dynamically allocated shared pointer on GPU.
     *  @param parameters_data_ptr Pointer to a pre-allocated GPU memory storing data of parameters.
     */
    __cudevice__ void * copy_to_shared_mem(candy::Model * share_ptr, void * parameters_data_ptr) const;
    /** @brief Copy meta-data from GPU global memory to shared memory of a kernel.
     *  @param share_ptr Dynamically allocated shared pointer on GPU.
     *  @param parameters_data_ptr Pointer to a pre-allocated GPU memory storing data of parameters.
     */
    __cudevice__ void * copy_to_shared_mem_single(candy::Model * share_ptr,
                                                  void * parameters_data_ptr) const;
    #endif  // __NVCC__
    /** @brief Copy data from GPU to CPU.*/
    MERLIN_EXPORTS void * copy_from_gpu(void * data_ptr, std::uintptr_t stream_ptr = 0);
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Model(void);
    /// @}

  protected:
    /** @brief Pointer to values of parameters.*/
    Vector<Vector<double>> parameters_;
    /** @brief Rank of the decomposition.*/
    std::uint64_t rank_ = 0;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_MODEL_HPP_
