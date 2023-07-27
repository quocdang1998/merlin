// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_MODEL_HPP_
#define MERLIN_CANDY_MODEL_HPP_

#include <array>  // std::array
#include <cstdint>  // std::uint64_t
#include <string>  // std::string

#include "merlin/array/nddata.hpp"  // merlin::array::Array
#include "merlin/array/slice.hpp"  // merlin::slicevec
#include "merlin/cuda_decorator.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/vector.hpp"  // merlin::Vector, merlin::floatvec

namespace merlin {

namespace candy {

/** @brief Random distribution for model initialization.
 *  @details Initial values of a canonical decomposition model must follow 2 rules: all entries must be non-zero, and
 *  values of the same rank and belonging to the same dimension cannot having the same value. Thus, a random generator
 *  is required to initialize model, based on mean and variance of the train data set.
 */
enum class RandomInitializer {
    /** @brief Uniform distribution between @f$ [-1, 1] \setminus \{0\} @f$.*/
    DefaultDistribution = 0x00,
    /** @brief Uniform distribution between @f$ [-m, m] \setminus \{0\} @f$, @f$ m @f$ is max value in train data.*/
    UniformDistribution = 0x01,
    /** @brief Normal distribution @f$ \mathcal{N}(\mu, \sigma) \setminus \{0\} @f$. The mean value @f$ \mu =
     *  \frac{1}{r} \sqrt[n]{M} @f$, in which @f$ r @f$ is the rank, @f$ n @f$ is the number of dimension, and
     *  @f$ M @f$ is the mean value, while @f$ \sigma @f$ is the standard deviation on the dimension to which the
     *  parameter belongs.
     */
    NormalDistribution = 0x02,
};

}  // namespace candy

/** @brief Canonical decomposition model.*/
class candy::Model {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Model(void) = default;
    /** @brief Constructor from train data shape and rank.*/
    MERLIN_EXPORTS Model(const intvec & shape, std::uint64_t rank);
    /** @brief Constructor from model values.*/
    MERLIN_EXPORTS Model(const Vector<floatvec> & parameter, std::uint64_t rank);
    /** @brief Slicing constructor.
     *  @details Partite the full model into multiple smaller potions.
     */
    MERLIN_EXPORTS Model(Model & full_model, const slicevec & slices);
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
        /** @brief Get reference to parameters.*/
    __cuhostdev__ constexpr Vector<floatvec> & parameters(void) noexcept {return this->parameters_;}
    /** @brief Get constant reference to parameters.*/
    __cuhostdev__ constexpr const Vector<floatvec> & parameters(void) const noexcept {return this->parameters_;}
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
    __cuhostdev__ constexpr double get(std::uint64_t i_dim, std::uint64_t index, std::uint64_t rank) const noexcept {
        return this->parameters_[i_dim][index*this->rank_ + rank];
    }
    /** @brief Set value of element at a given index.*/
    __cuhostdev__ constexpr void set(std::uint64_t i_dim, std::uint64_t index, std::uint64_t rank,
                                     double value) noexcept {
        this->parameters_[i_dim][index*this->rank_ + rank] = value;
    }
    /** @brief Get dimension and index from contiguous index.*/
    __cuhostdev__ std::array<std::uint64_t, 2> convert_contiguous(std::uint64_t index) const noexcept;
    /** @brief Get reference to element from flattened index.*/
    __cuhostdev__ const double & get(std::uint64_t index) const noexcept;
    /** @brief Set value of element from flattened index.*/
    __cuhostdev__ void set(std::uint64_t index, double value) noexcept;
    /// @}

    /// @name Evaluation of the model
    /// @{
    /** @brief Evaluate result of the model at a given index.*/
    __cuhostdev__ double eval(const intvec & index) const noexcept;
    /// @}

    /// @name Initialization model
    /// @{
    /** @brief Initialize values of model based on train data.
     *  @param train_data Data to train the model.
     *  @param random_distribution Random distribution from which values are sampled.
     *  @param n_thread Number of parallel threads for calculation the mean and variance.
     */
    MERLIN_EXPORTS void initialize(const array::Array & train_data,
                                   candy::RandomInitializer random_distribution, std::uint64_t n_thread = 1);
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the model and its data.*/
    MERLIN_EXPORTS std::uint64_t cumalloc_size(void) const;
    /** @brief Copy the model from CPU to a pre-allocated memory on GPU.
     *  @details Values of vectors should be copied to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param parameters_data_ptr Pointer to a pre-allocated GPU memory storing data of parameters.
     *  @param stream_ptr Pointer to CUDA stream for asynchronious copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(candy::Model * gpu_ptr, void * parameters_data_ptr,
                                      std::uintptr_t stream_ptr = 0) const;
    /** @brief Calculate the minimum number of bytes to allocate in CUDA shared memory to store the model.*/
    std::uint64_t sharedmem_size(void) const {return this->cumalloc_size();}
    #ifdef __NVCC__
    /** @brief Copy model to pre-allocated memory region by current CUDA block of threads.
     *  @details The copy action is performed by the whole CUDA thread block.
     *  @param dest_ptr Memory region where the model is copied to.
     *  @param parameters_data_ptr Pointer to a pre-allocated GPU memory storing data of parameters, size of
     *  ``floatvec[this->ndim()] + double[this->size()]``.
     *  @param thread_idx Flatten ID of the current CUDA thread in the block.
     *  @param block_size Number of threads in the current CUDA block.
     */
    __cudevice__ void * copy_by_block(candy::Model * dest_ptr, void * parameters_data_ptr, std::uint64_t thread_idx,
                                      std::uint64_t block_size) const;
    /** @brief Copy model to a pre-allocated memory region by a single GPU threads.
     *  @param dest_ptr Memory region where the model is copied to.
     *  @param parameters_data_ptr Pointer to a pre-allocated GPU memory storing data of parameters, size of
     *  ``floatvec[this->ndim()] + double[this->size()]``.
     */
    __cudevice__ void * copy_by_thread(candy::Model * dest_ptr, void * parameters_data_ptr) const;
    #endif  // __NVCC__
    /** @brief Copy data from GPU to CPU.*/
    MERLIN_EXPORTS void * copy_from_gpu(candy::Model * gpu_ptr, std::uintptr_t stream_ptr = 0);
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
    Vector<floatvec> parameters_;
    /** @brief Rank of the decomposition.*/
    std::uint64_t rank_ = 0;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_MODEL_HPP_
