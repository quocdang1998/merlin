// Copyright 2022 quocdang1998
/** @file array.hpp
    @brief Provide definition of a basic array.
*/
#ifndef MERLIN_ARRAY_HPP_
#define MERLIN_ARRAY_HPP_

#include <cstdint>  // uintptr_t
#include <initializer_list>  // std::initializer_list
#include <tuple>  // std::tie
#include <vector>  // std::vector

namespace merlin {

/** @brief Basic array object.*/
class Array {
  public:
    // Constructor
    // -----------
    /** @brief Default constructor (do nothing).*/
    Array(void) = default;
    /** @brief Constructor from data pointer and meta-data.
     *  @details This constructor is used to construct explicitly an Array in C++ interface.
     *  @param data Pointer to data.
     *  @param ndim Number of dimension.
     *  @param shape Shape vector.
     *  @param strides Strides vector.
     */
    Array(float * data, unsigned int ndim, std::initializer_list<unsigned int> shape,
          std::initializer_list<unsigned int> strides);
    /** @brief Constructor from data pointer and meta-data pointers.
     *  @details This constructor is designed for initializing object from Numpy np.array.
     *  @param data Pointer to data.
     *  @param ndim Number of dimension.
     *  @param shape Pointer to shape vector.
     *  @param strides Pointer to strides vector.
     */
    Array(float * data, unsigned int ndim, unsigned int * shape, unsigned int * strides);

    // Copy and move
    // -------------
    /** @brief Shallow copy constructor.*/
    Array(const Array & source) = default;
    /** @brief Shallow copy assignment.*/
    Array & operator= (const Array & source) = default;
    /** @brief Move constructor.*/
    Array(Array && source) = default;
    /** @brief Move assignment.*/
    Array & operator= (Array && source) = default;

    // Get members
    // -----------
    /** @brief Get element pointer to data.*/
    float * data(void) const {return this->data_;}
    /** @brief Get number of dimension.*/
    unsigned int ndim(void) const {return this->ndim_;}
    /** @brief Get reference to shape vector.*/
    std::vector<unsigned int> & shape(void) {return this->shape_;}
    /** @brief Get constant reference to shape vector.*/
    const std::vector<unsigned int> & shape(void) const {return this->shape_;}
    /** @brief Get reference to shape vector.*/
    std::vector<unsigned int> & strides(void) {return this->strides_;}
    /** @brief Get constant reference to shape vector.*/
    const std::vector<unsigned int> & strides(void) const {return this->strides_;}

    // Atributes
    // ---------
    /** @brief Number of element.*/
    unsigned int size(void);

    // Destructor
    // ----------
    /** @brief Default destructor.*/
    virtual ~Array(void) = default;


  protected:
    // Members
    // -------
    /** @brief Pointer to data.*/
    float * data_ = NULL;
    /** @brief Number of dimension.*/
    unsigned int ndim_;
    /** @brief Shape vector.
     *  @details Size of each dimension.
     */
    std::vector<unsigned int> shape_;
    /** @brief Stride vector.\
     *  @details Number of incresing bytes in memory when an index of a dimension jumps by 1.
     */
    std::vector<unsigned int> strides_;
};

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

// Forward declaration of subclasses
class Tensor;  // CPU Array, defined in tensor.hpp
class Parcel;  // GPU Array, defined in parcel.hpp


}  // namespace merlin

#endif  // MERLIN_ARRAY_HPP_
