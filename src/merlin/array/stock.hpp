// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_STOCK_HPP_
#define MERLIN_ARRAY_STOCK_HPP_

#include <cstdint>  // std::uint64_t
#include <cstdio>   // std::FILE
#include <string>   // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Stock
#include "merlin/array/nddata.hpp"       // merlin::array::NdData
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/io/file_lock.hpp"       // merlin::io::FileLock

namespace merlin {

/** @brief Multi-dimensional array exported to a file.
 *  @details The read/write is guaranteed to the threadsafe by the member merlin::array::Stock::thread_safe_. When it is
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
     *  @param thread_safe Threadsafe policy.
     */
    MERLIN_EXPORTS Stock(const std::string & filename, const Index & shape, std::uint64_t offset = 0,
                         bool thread_safe = true);
    /** @brief Open an already existing file for reading and storing data.
     *  @param filename Name of input file (or output file).
     *  @param offset Starting position from the beginning of file.
     *  @param thread_safe Threadsafe policy.
     */
    MERLIN_EXPORTS Stock(const std::string & filename, std::uint64_t offset = 0, bool thread_safe = true);
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
    io::FileLock & get_file_lock(void) const noexcept { return this->flock_; }
    /** @brief Check if the policy is thread safe.*/
    constexpr bool is_thread_safe(void) const noexcept { return this->thread_safe_; }
    /** @brief Check if the file is created with the same endianess as the machine.*/
    constexpr bool is_same_endianess(void) const noexcept { return this->same_endianess_; }
    /// @}

    /// @name Get and set element
    /// @{
    /** @brief Get value of element at a n-dim index.*/
    MERLIN_EXPORTS double get(const Index & index) const;
    /** @brief Get value of element at a C-contiguous index.*/
    MERLIN_EXPORTS double get(std::uint64_t index) const;
    /** @brief Set value of element at a n-dim index.*/
    MERLIN_EXPORTS void set(const Index & index, double value);
    /** @brief Set value of element at a C-contiguous index.*/
    MERLIN_EXPORTS void set(std::uint64_t index, double value);
    /// @}

    /// @name Operations
    /// @{
    /** @brief Set value of all elements.*/
    MERLIN_EXPORTS void fill(double value);
    /** @brief Calculate mean and variance of all non-zero and finite elements.*/
    MERLIN_EXPORTS std::array<double, 2> get_mean_variance(void) const;
    /** @brief Create a polymorphic sub-array.*/
    array::NdData * get_p_sub_array(const SliceArray & slices) const {
        array::Stock * p_result = new array::Stock();
        this->create_sub_array(*p_result, slices);
        p_result->file_ptr_ = this->file_ptr_;
        p_result->flock_ = this->flock_;
        p_result->offset_ = this->offset_;
        p_result->thread_safe_ = this->thread_safe_;
        p_result->filename_ = this->filename_;
        return p_result;
    }
    /** @brief Create sub-array with the same type.*/
    array::Stock get_sub_array(const SliceArray & slices) const {
        array::Stock sub_array;
        this->create_sub_array(sub_array, slices);
        sub_array.file_ptr_ = this->file_ptr_;
        sub_array.flock_ = this->flock_;
        sub_array.offset_ = this->offset_;
        sub_array.thread_safe_ = this->thread_safe_;
        sub_array.filename_ = this->filename_;
        return sub_array;
    }
    /// @}

    /// @name Write to file
    /// @{
    /** @brief Write data from a merlin::array::Array to a file.*/
    MERLIN_EXPORTS void record_data_to_file(const Array & src);
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(bool first_call = true) const;
    /// @}

    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Stock(void);

  protected:
    /** @brief Filename.*/
    std::string filename_;
    /** @brief Thread safe read/write.*/
    bool thread_safe_ = true;
    /** @brief Start writing/reading position wrt. the beginning of file.*/
    std::uint64_t offset_;
    /** @brief Flag indicating if the stock file is created with the same endianess as the machine.*/
    bool same_endianess_ = true;

    /** @brief Pointer to file stream.*/
    mutable std::FILE * file_ptr_;
    /** @brief Lock file.*/
    mutable io::FileLock flock_;

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
