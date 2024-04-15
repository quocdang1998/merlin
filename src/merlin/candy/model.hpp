// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_MODEL_HPP_
#define MERLIN_CANDY_MODEL_HPP_

#include <array>    // std::array
#include <cstdint>  // std::uint64_t
#include <string>   // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::Array
#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/config.hpp"             // __cudevice__, __cuhostdev__, merlin::DPtrArray, merlin::Index
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/vector.hpp"             // merlin::DoubleVec, merlin::Vector

namespace merlin {

/** @brief Canonical decomposition model.*/
class candy::Model {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Model(void) = default;
    /** @brief Constructor from train data shape and rank.
     *  @param shape Shape of the desired reconstructed data.
     *  @param rank Rank of the model.
     */
    MERLIN_EXPORTS Model(const Index & shape, std::uint64_t rank);
    /** @brief Constructor from model values.
     *  @param param_vectors Vector of flatten parameters.
     *  @param rank Rank of the model.
     */
    MERLIN_EXPORTS Model(const Vector<DoubleVec> & param_vectors, std::uint64_t rank);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    MERLIN_EXPORTS Model(const candy::Model & src);
    /** @brief Copy assignment.*/
    MERLIN_EXPORTS candy::Model & operator=(const candy::Model & src);
    /** @brief Move constructor.*/
    Model(candy::Model && src) = default;
    /** @brief Move assignment.*/
    candy::Model & operator=(candy::Model && src) = default;
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get reference to parameters.*/
    __cuhostdev__ constexpr DPtrArray & param_vectors(void) noexcept { return this->param_vectors_; }
    /** @brief Get constant reference to parameters.*/
    __cuhostdev__ constexpr const DPtrArray & param_vectors(void) const noexcept { return this->param_vectors_; }
    /** @brief Get number of dimension.*/
    __cuhostdev__ constexpr std::uint64_t ndim(void) const noexcept { return this->ndim_; }
    /** @brief Get rank by shape.*/
    __cuhostdev__ constexpr const Index & rshape(void) const noexcept { return this->rshape_; }
    /** @brief Get rank.*/
    __cuhostdev__ constexpr std::uint64_t rank(void) const noexcept { return this->rank_; }
    /** @brief Get number of parameters.*/
    __cuhostdev__ std::uint64_t num_params(void) const noexcept { return this->parameters_.size(); }
    /// @}

    /// @name Get and set parameters
    /// @{
    /** @brief Get reference to an element at a given dimension, index and rank.*/
    __cuhostdev__ constexpr double & get(std::uint64_t i_dim, std::uint64_t index, std::uint64_t rank) noexcept {
        return this->param_vectors_[i_dim][index * this->rank_ + rank];
    }
    /** @brief Get constant reference to an element at a given dimension, index and rank.*/
    __cuhostdev__ constexpr const double & get(std::uint64_t i_dim, std::uint64_t index,
                                               std::uint64_t rank) const noexcept {
        return this->param_vectors_[i_dim][index * this->rank_ + rank];
    }
    /** @brief Get reference to element from flattened index.*/
    __cuhostdev__ double & operator[](std::uint64_t index) noexcept { return this->parameters_[index]; }
    /** @brief Get constant reference to element from flattened index.*/
    __cuhostdev__ const double & operator[](std::uint64_t index) const noexcept { return this->parameters_[index]; }
    /// @}

    /// @name Initialization
    /// @{
    /** @brief Initialize values of model based on train data.
     *  @details Initialize values randomly with normal distribution @f$ \mathcal{N}(\mu, \sigma) \setminus
     *  \{\mathbb{R}^-\} @f$. The mean value @f$ \mu = \frac{1}{r} \sqrt[n]{M} @f$, in which @f$ r @f$ is the rank,
     *  @f$ n @f$ is the number of dimension, and @f$ M @f$ is the mean value of the hyper-slice at index corresponding
     *  to parameter. The standard deviation @f$ \sigma @f$ the standard deviation of the hyper-slice corrected in
     *  corresponding with the mean.
     *  @param train_data Data to train the model.
     */
    MERLIN_EXPORTS void initialize(const array::Array & train_data);
    /** @brief Initialize values of model based on rank-1 model.
     *  @details Initialized values randomly of a model based on another trained rank-1 model.
     *  @param rank_1_model Rank-1 model, must have the same shape as the current model.
     *  @param rtol Relative tolerance of the randomized values.
     */
    MERLIN_EXPORTS void initialize(const candy::Model & rank_1_model, double rtol = 0.01);
    /// @}

    /// @name Evaluation of the model
    /// @{
    /** @brief Evaluate result of the model at a given ndim index in the resulted array.*/
    __cuhostdev__ double eval(const Index & index) const noexcept;
    /// @}

    /// @name Check negative value
    /// @{
    /** @brief Check if these is a negative parameter in the model.
     *  @return Return ``false`` if these is a negative parameter in the model.
     */
    MERLIN_EXPORTS bool check_negative(void) const noexcept;
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the model and its data.*/
    std::uint64_t cumalloc_size(void) const noexcept {
        return sizeof(candy::Model) + this->num_params() * sizeof(double);
    }
    /** @brief Copy the model from CPU to a pre-allocated memory on GPU.
     *  @details Values of vectors should be copied to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param parameters_data_ptr Pointer to a pre-allocated GPU memory storing data of parameters.
     *  @param stream_ptr Pointer to CUDA stream for asynchronious copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(candy::Model * gpu_ptr, void * parameters_data_ptr,
                                      std::uintptr_t stream_ptr = 0) const;
    /** @brief Calculate the minimum number of bytes to allocate in CUDA shared memory to store the model.*/
    std::uint64_t sharedmem_size(void) const noexcept { return this->cumalloc_size(); }
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
    /** @brief Copy data from GPU back to CPU.*/
    MERLIN_EXPORTS void * copy_from_gpu(double * data_from_gpu, std::uintptr_t stream_ptr = 0) noexcept;
    /// @}

    /// @name Serialization
    /// @{
    /** @brief Write model into a file.
     *  @param fname Name of the output file.
     *  @param lock Lock the file when writing to prevent data race. The lock action may cause a delay.
     */
    MERLIN_EXPORTS void save(const std::string & fname, bool lock = false) const;
    /** @brief Read model from a file.
     *  @param fname Name of the input file.
     *  @param lock Lock the file when reading to prevent data race. The lock action may cause a delay.
     */
    MERLIN_EXPORTS void load(const std::string & fname, bool lock = false);

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
    /** @brief Vector of contiguous parameters per dimension and rank.*/
    DoubleVec parameters_;
    /** @brief Number of parameters per dimension.
     *  @details Equals to rank times shape.
     */
    Index rshape_;
    /** @brief Number of dimensions.*/
    std::uint64_t ndim_ = 0;
    /** @brief Rank of the decomposition.*/
    std::uint64_t rank_ = 0;

  private:
    /** @brief Pointer to the first parameters per dimension.*/
    DPtrArray param_vectors_;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_MODEL_HPP_
