// Copyright 2022 quocdang1998
#ifndef MERLIN_NDDATA_HPP_
#define MERLIN_NDDATA_HPP_

#include <cstddef>  // NULL
#include <tuple>  // std::tie

#include "merlin/decorator.hpp"  // __cuhost__, __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

/** @brief Abstract class of N-dim array.*/
class MERLIN_EXPORTS NdData {
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
    NdData(float * data, unsigned long int ndim, const intvec & shape, const intvec & strides);
    /** @brief Constructor from data pointer and meta-data pointers.
     *  @details This constructor is designed for initializing object from Numpy np.array.
     *  @param data Pointer to data.
     *  @param ndim Number of dimension.
     *  @param shape Pointer to shape vector.
     *  @param strides Pointer to strides vector.
     */
    NdData(float * data, unsigned long int ndim, const unsigned long int * shape, const unsigned long int * strides);
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

/** @brief Iterator of multi-dimensional array.*/
class MERLIN_EXPORTS Iterator {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Iterator(void) = default;
    /** @brief Constructor from multi-dimensional index and container.*/
    Iterator(const intvec & index, NdData & container);
    /** @brief Constructor from C-contiguous index.*/
    Iterator(unsigned long int index, NdData & container);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    Iterator(const Iterator & src) = default;
    /** @brief Copy assignment.*/
    Iterator & operator=(const Iterator & src) = default;
    /** @brief Move constructor.*/
    Iterator(Iterator && src) = default;
    /** @brief Move assignment.*/
    Iterator & operator=(Iterator && src) = default;
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get multi-dimensional index of an iterator.*/
    intvec & index(void) {return this->index_;}
    /** @brief Get constant multi-dimensional index of an iterator.*/
    const intvec & index(void) const {return this->index_;}
    /// @}

    /// @name Operators
    /// @{
    /** @brief Dereference operator of an iterator.*/
    float & operator*(void) {return *(this->item_ptr_);}
    /** @brief Comparison operator.*/
    MERLIN_EXPORTS friend bool operator!=(const Iterator & left, const Iterator & right) {
        return left.item_ptr_ != right.item_ptr_;
    }
    /** @brief Pre-increment operator.*/
    Iterator & operator++(void);
    /** @brief Post-increment operator.*/
    Iterator operator++(int) {return ++(*this);}
    /** @brief Update index vector to be consistent with the shape.*/
    void MERLIN_DEPRECATED update(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Iterator(void) = default;
    /// @}

  protected:
    /** @brief Pointer to item.*/
    float * item_ptr_ = NULL;
    /** @brief Index vector.*/
    intvec index_;
    /** @brief Pointer to NdData object possessing the item.*/
    NdData * container_ = NULL;
};

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
     *  @param step Step (positive means step to right, negative means step to the left).
     */
    Slice(unsigned long int start, long int stop, long int step) : start_(start), stop_(stop), step_(step) {
        if (step == 0) {
            FAILURE(std::invalid_argument, "Step must not be 0.\n");
        }
    }
    /** @brief Constructor from an initializer list.
     *  @param list Initializer list of length 3.
     */
    Slice(std::initializer_list<int> list);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    Slice(const Slice & src) = default;
    /** @brief Copy assignment.*/
    Slice & operator=(const Slice & src) = default;
    /** @brief Move constructor.*/
    Slice(Slice && src) = default;
    /** @brief Move assignment.*/
    Slice & operator=(Slice && src)= default;
    /// @}

    // Get members
    // -----------
    /** @brief Get reference to start value.*/
    unsigned long int & start(void) {return this->start_;}
    /** @brief Get constant reference to start value.*/
    const unsigned long int & start(void) const {return this->start_;}
    /** @brief Get reference to stop value.*/
    long int & stop(void) {return this->stop_;}
    /** @brief Get constant reference to stop value.*/
    const long int & stop(void) const {return this->stop_;}
    /** @brief Get reference to step value.*/
    long int & step(void) {return this->step_;}
    /** @brief Get constant reference to step value.*/
    const long int & step(void) const {return this->step_;}

    // Convert to range
    // ----------------
    /** @brief Get indices corresponding to element represented by the slice.
     *  @param length Length of the array.
     */
    intvec range(unsigned long int length);

  protected:
    // Members
    // -------
    /** @brief Start index.*/
    unsigned long int start_ = 0;
    /** @brief Stop index, count from last element.
     *  @details Positive means count from zeroth element, negative means count from last element.
     */
    long int stop_ = 0;
    /** @brief Step
     * @details Positive means stepping to the right, Negative means stepping to the left.
     */
    long int step_ = 1;
};

}  // namespace merlin

#endif  // MERLIN_NDDATA_HPP_
