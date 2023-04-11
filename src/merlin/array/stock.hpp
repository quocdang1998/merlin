// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_STOCK_HPP_
#define MERLIN_ARRAY_STOCK_HPP_

#include <cstdint>  // std::uint64_t, std::uintptr_t
#include <cstdio>   // std::FILE
#include <string>   // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::Array
#include "merlin/array/nddata.hpp"       // merlin::array::NdData
#include "merlin/array/slice.hpp"        // merlin::array::Slice
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/filelock.hpp"           // merlin::FileLock
#include "merlin/vector.hpp"             // merlin::Vector

namespace merlin {

/** @brief Multi-dimensional array exported to a file.
 *  @details The read/write is garanteed to the threadsafe by the member merlin::array::Stock::thread_safe_. When it is
 *  set to ``true``, multiple threads/processes can read file at the same time, but only one can write the file. If a
 *  thread/process opening the file for reading (respectively writing), it will have to wait until all currently writing
 *  (respectively reading) threads/processes finished their jobs. By switching off this option, user is responsible for
 *  assuring that concurrent I/O operations do not crash one another.
 */
class array::Stock : public array::NdData {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    Stock(void) = default;
    /** @brief Open an empty file for storing data.
     *  @details This file may not exist. If it not, it will create an empty file. If the file exists, it will resize
     *  the file to fit the data to record.
     *  @param filename Name of input file (or output file).
     *  @param shape Shape of the array to be written to file.
     *  @param offset Starting position from the beginning of file.
     *  @param thread_safe Threasafe policy.
     */
    MERLIN_EXPORTS Stock(const std::string & filename, const intvec & shape, std::uint64_t offset = 0,
                         bool thread_safe = true);
    /** @brief Open an already existing file for reading and storing data.
     *  @param filename Name of input file (or output file).
     *  @param offset Starting position from the beginning of file.
     *  @param thread_safe Threasafe policy.
     */
    MERLIN_EXPORTS Stock(const std::string & filename, std::uint64_t offset = 0, bool thread_safe = true);
    /** @brief Constructor from a slice.
     *  @param whole merlin::array::Stock of the original array.
     *  @param slices List of merlin::array::Slice on each dimension.
     */
    MERLIN_EXPORTS Stock(const array::Stock & whole, const Vector<array::Slice> & slices);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor (deleted).*/
    Stock(const array::Stock & src) = delete;
    /** @brief Copy assignment (deleted).*/
    array::Stock & operator=(const array::Stock & src) = delete;
    /** @brief Move constructor.*/
    Stock(array::Stock && src) = default;
    /** @brief Move assignment.*/
    array::Stock & operator=(array::Stock && src) = default;
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get filename.*/
    constexpr const std::string & filename(void) const noexcept { return this->filename_; }
    /** @brief Get pointer to file.*/
    std::FILE * get_file_ptr(void) const noexcept { return this->file_ptr_; }
    /** @brief Get filelock.*/
    FileLock & get_file_lock(void) const noexcept { return this->flock_; }
    /** @brief Check if the policy is thread safe.*/
    constexpr bool is_thread_safe(void) const noexcept { return this->thread_safe_; }
    /// @}

    /// @name Get and set element
    /// @{
    /** @brief Get value of element at a n-dim index.*/
    MERLIN_EXPORTS double get(const intvec & index) const;
    /** @brief Get value of element at a C-contiguous index.*/
    MERLIN_EXPORTS double get(std::uint64_t index) const;
    /** @brief Set value of element at a n-dim index.*/
    MERLIN_EXPORTS void set(const intvec index, double value);
    /** @brief Set value of element at a C-contiguous index.*/
    MERLIN_EXPORTS void set(std::uint64_t index, double value);
    /// @}

    /// @name Operations
    /// @{
    /** @brief Reshape the dataset.
     *  @param new_shape New shape.
     */
    MERLIN_EXPORTS void reshape(const intvec & new_shape);
    /** @brief Collapse dimensions with size 1.
     *  @param i_dim Index of dimension to collapse.
     */
    MERLIN_EXPORTS void remove_dim(std::uint64_t i_dim = 0);
    /** @brief Set value of all elements.*/
    MERLIN_EXPORTS void fill(double value);
    /// @}

    /// @name Write to file
    /// @{
    /** @brief Write data from a merlin::array::Array to a file.*/
    MERLIN_EXPORTS void record_data_to_file(const Array & src);
    /// @}

    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Stock(void);

  protected:
    /** @brief Pointer to file stream.*/
    mutable std::FILE * file_ptr_;
    /** @brief Lock file.*/
    mutable FileLock flock_;
    /** @brief Thread safe read/write.*/
    bool thread_safe_ = true;
    /** @brief Start writing/reading position wrt. the beginning of file.*/
    std::uint64_t offset_;

    /** @brief Filename.*/
    std::string filename_;

  private:
    /** @brief Read metadata from file.
     *  @return Porsion of the cursor at the end of written position.
     */
    std::uint64_t read_metadata(void);
    /** @brief Write metadata to file.
     *  @return Porsion of the cursor at the end of written position.
     */
    std::uint64_t write_metadata(void);
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_STOCK_HPP_
