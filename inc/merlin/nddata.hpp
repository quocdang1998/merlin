// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_HPP_
#define MERLIN_ARRAY_HPP_

#include <cstddef>  // NULL
#include <initializer_list>  // std::initializer_list
#include <tuple>  // std::tie

#include "merlin/decorator.hpp"  // __cuhost__, __cuhostdev__
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

/** @brief Abstract class of N-dim array.*/
class NdData {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor (do nothing).*/
    NdData(void) = default;
    /** @brief Constructor from data pointer and meta-data.
     *  @details This constructor is used to construct explicitly an NdData in C++ interface.
     *  @param data Pointer to data.
     *  @param ndim Number of dimension.
     *  @param shape Shape vector.
     *  @param strides Strides vector.
     */
    NdData(float * data, unsigned long int ndim, std::initializer_list<unsigned long int> shape,
           std::initializer_list<unsigned long int> strides);
    /** @brief Constructor from data pointer and meta-data pointers.
     *  @details This constructor is designed for initializing object from Numpy np.array.
     *  @param data Pointer to data.
     *  @param ndim Number of dimension.
     *  @param shape Pointer to shape vector.
     *  @param strides Pointer to strides vector.
     */
    NdData(float * data, unsigned long int ndim, unsigned long int * shape, unsigned long int * strides);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Shallow copy constructor.*/
    NdData(const NdData & source) = default;
    /** @brief Shallow copy assignment.*/
    NdData & operator= (const NdData & source) = default;
    /** @brief Move constructor.*/
    NdData(NdData && source) = default;
    /** @brief Move assignment.*/
    NdData & operator= (NdData && source) = default;
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get pointer to data.*/
    __cuhostdev__ float * data(void) const {return this->data_;}
    /** @brief Get number of dimension.*/
    __cuhostdev__ unsigned long int ndim(void) const {return this->ndim_;}
    /** @brief Get reference to shape vector.*/
    __cuhostdev__ intvec & shape(void) {return this->shape_;}
    /** @brief Get constant reference to shape vector.*/
    __cuhostdev__ const intvec & shape(void) const {return this->shape_;}
    /** @brief Get reference to stride vector.*/
    __cuhostdev__ intvec & strides(void) {return this->strides_;}
    /** @brief Get constant reference to stride vector.*/
    __cuhostdev__ const intvec & strides(void) const {return this->strides_;}
    /// @}

    /// @name Atributes
    /// @{
    /** @brief Number of element.*/
    __cuhostdev__ unsigned long int size(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    virtual ~NdData(void) = default;
    /// @}


  protected:
    /** @brief Pointer to data.*/
    float * data_ = NULL;
    /** @brief Number of dimension.*/
    unsigned long int ndim_;
    /** @brief Shape vector.
     *  @details Size of each dimension.
     */
    intvec shape_;
    /** @brief Stride vector.
     *  @details Number of incresing bytes in memory when an intvec of a dimension jumps by 1.
     */
    intvec strides_;
};

// Forward declaration of subclasses
class Array;  // CPU Array, defined in tensor.hpp
class Parcel;  // GPU Array, defined in parcel.hpp


#ifdef FUTURE
/** @brief Slice of an Array.*/
class Slice {
  public:
    // Constructors
    // ------------
    /** @brief Default constructor.
     *  @details Construct a full slice (start at 0, end at last element, step 1).
     */
    Slice(void) = default;
    /** @brief Member constructor.
     *  @details Construct Slice obejct from value of its members.
     *  @param start Start position (must be positive).
     *  @param stop Stop position (count from the last element, modulo if range exceeded).
     *  @param step Step (positive means step to right, negative means step to the left).
     */
    Slice(unsigned int start, int stop, int step);
    /** @brief Constructor from an initializer list.
     *  @param list Initializer list of length 3.
     */
    Slice(std::initializer_list<int> list);

    // Get members
    // -----------
    /** @brief Get reference to start value.*/
    unsigned int & start(void) {return this->start_;}
    /** @brief Get constant reference to start value.*/
    const unsigned int & start(void) const {return this->start_;}
    /** @brief Get reference to stop value.*/
    int & stop(void) {return this->stop_;}
    /** @brief Get constant reference to stop value.*/
    const int & stop(void) const {return this->stop_;}
    /** @brief Get reference to step value.*/
    int & step(void) {return this->step_;}
    /** @brief Get constant reference to step value.*/
    const int & step(void) const {return this->step_;}

    // Convert to range
    // ----------------
    /** @brief Get indices corresponding to element represented by the slice.
     *  @param length Length of the array.
     */
    std::vector<unsigned int> range(unsigned int length);

  protected:
    // Members
    // -------
    /** @brief Start index.*/
    unsigned int start_ = 0;
    /** @brief Stop index, count from last element.
     *  @details Positive means count from zeroth element, negative means count from last element.
     */
    int stop_ = 0;
    /** @brief Step
     * @details Positive means stepping to the right, Negative means stepping to the left.
     */
    int step_ = 1;
};
#endif


}  // namespace merlin

#endif  // MERLIN_ARRAY_HPP_
