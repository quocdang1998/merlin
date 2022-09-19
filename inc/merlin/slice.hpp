// Copyright 2022 quocdang1998
// Todo: implement strides < 0 and slice < 0
#ifndef MERLIN_SLICE_HPP_
#define MERLIN_SLICE_HPP_

#include <cmath>  // std::abs
#include <cstdint>  // std::int64_t, std::uint64_t

#include "merlin/decorator.hpp"  // __cuhost__, __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

/** @brief Slice of an Array.*/
class Slice {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.
     *  @details Construct a full slice (start at 0, end at last element, step 1).
     */
    Slice(void) = default;
    /** @brief Member constructor.
     *  @details Construct Slice object from values of its members.
     *  @param start Start position (must be positive).
     *  @param stop Stop position (count from the last element, modulo if range exceeded).
     *  @param step Step (must be positive).
     */
    __cuhostdev__ Slice(std::uint64_t start, std::uint64_t stop,
                        std::uint64_t step) : start_(start), stop_(stop), step_(step) {
        if (step <= 0) {
            #ifdef __CUDA_ARCH__
            CUDAOUT("Step must be strictly positive.\n");
            #else
            FAILURE(std::invalid_argument, "Step must be strictly positive.\n");
            #endif
        }
    }
    /** @brief Constructor from an initializer list.
     *  @param list Initializer list of length 3.
     */
    __cuhostdev__ inline Slice(std::initializer_list<std::uint64_t> list);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    __cuhostdev__ Slice(const Slice & src) : start_(src.start_), stop_(src.stop_), step_(src.step_) {}
    /** @brief Copy assignment.*/
    __cuhostdev__ Slice & operator=(const Slice & src) {
        this->start_ = src.start_;
        this->stop_ = src.stop_;
        this->step_ = src.step_;
        return *this;
    }
    /** @brief Move constructor.*/
    __cuhostdev__ Slice(Slice && src) : start_(src.start_), stop_(src.stop_), step_(src.step_) {}
    /** @brief Move assignment.*/
    __cuhostdev__ Slice & operator=(Slice && src) {
        this->start_ = src.start_;
        this->stop_ = src.stop_;
        this->step_ = src.step_;
        return *this;
    }
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get reference to start value.*/
    __cuhostdev__ std::uint64_t & start(void) {return this->start_;}
    /** @brief Get constant reference to start value.*/
    __cuhostdev__ const std::uint64_t & start(void) const {return this->start_;}
    /** @brief Get reference to stop value.*/
    __cuhostdev__ std::uint64_t & stop(void) {return this->stop_;}
    /** @brief Get constant reference to stop value.*/
    __cuhostdev__ const std::uint64_t & stop(void) const {return this->stop_;}
    /** @brief Get reference to step value.*/
    __cuhostdev__ std::uint64_t & step(void) {return this->step_;}
    /** @brief Get constant reference to step value.*/
    __cuhostdev__ const std::uint64_t & step(void) const {return this->step_;}
    /// @}

    /// @name Convert to range
    /// @{
    /** @brief Get indices corresponding to element represented by the slice.
     *  @param length Length of the array.
     */
    __cuhostdev__ intvec range(void);
    /// @}

  protected:
    /** @brief Start index.*/
    std::uint64_t start_ = 0;
    /** @brief Stop index, count from last element.
     *  @details Positive means count from zeroth element, negative means count from last element.
     */
    std::uint64_t stop_ = 0;
    /** @brief Step
     * @details Positive means stepping to the right, Negative means stepping to the left.
     */
    std::uint64_t step_ = 1;
};

// Constructor from an initializer list
__cuhostdev__ inline Slice::Slice(std::initializer_list<std::uint64_t> list) {
    const std::uint64_t * list_data = list.begin();
    switch (list.size()) {
    case 1:  // 1 element = get 1 element
        this->start_ = list_data[0];
        this->stop_ = list_data[0] + 1;
        break;
    case 2:  // 2 element = {start, stop}
        this->start_ = list_data[0];
        this->stop_ = list_data[1];
        break;
    case 3:
        this->start_ = list_data[0];
        this->stop_ = list_data[1];
        this->step_ = list_data[2];
        break;
    default:
        #ifdef __CUDA_ARCH__
        CUDAOUT("Expected intializer list with size at most 3, got %I64u.\n", list.size());
        #else
        FAILURE(std::invalid_argument, "Expected intializer list with size at most 3, got %d.\n", list.size());
        #endif
        break;
    }
}

// Conver Slice to vector of corresponding indices
__cuhostdev__ inline intvec Slice::range(void) {
    std::uint64_t range_length = (this->start_-this->stop_) / this->step_;
    intvec range(range_length, INT64_MAX);
    for (unsigned int i = 0; i < range_length; i += 1) {
        range[i] = this->start_ + i*this->step_;
    }
    return range;
}

}  // namespace merlin

#endif  // MERLIN_SLICE_HPP_
