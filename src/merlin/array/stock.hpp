// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_STOCK_HPP_
#define MERLIN_ARRAY_STOCK_HPP_

#include <cstdint>  // std::uint64_t, std::uintptr_t
#include <cstdio>  // std::FILE
#include <string>  // std::string

#include "merlin/array/nddata.hpp"  // merlin::array::NdData
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/filelock.hpp"  // merlin::FileLock

namespace merlin::array {

/** @brief Multi-dimensional array exported to a file.*/
class MERLIN_EXPORTS Stock : public NdData {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    Stock(void) = default;
    /** @brief Constructor from a filename.
     *  @param filename Name of input file (or output file).
     *  @param mode Open mode:
     *    - r: Read only.
     *    - w: Write only (crash old file if exist).
     *    - a: Thread safe read and write (multiple thread can read simultaneously, but only one can write at a time).
     *  @param offset Starting position from the beginning of file.
     *  @param threadsafe When it is set to ``true``, multiple threads/processes can read file at the same time, but
     *  only one can write the file. If a thread/process opening the file for reading (respectively writing), it will
     *  have to wait until all currently writing (respectively reading) threads/processes finished their jobs.
     */
    Stock(const std::string & filename, char mode = 'a', std::uint64_t offset = 0, bool thread_safe = true);
    Stock(const std::string & filename, const intvec & shape, bool thread_safe = true);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor (deleted).*/
    Stock(const Stock & src) = delete;
    /** @brief Copy assignment (deleted).*/
    Stock & operator=(const Stock & src) = delete;
    /** @brief Move constructor.*/
    Stock(Stock && src) = default;
    /** @brief Move assignment.*/
    Stock & operator=(Stock && src) = default;
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get filename.*/
    std::string filename(void) const {return this->filename_;}
    /** @brief Get pointer to file.*/
    std::FILE * file_ptr(void) const {return this->file_ptr_;}
    /// @}

    /// @name Get and set element
    /// @{
    /** @brief Get value of element at a n-dim index.*/
    float get(const intvec & index) const;
    /** @brief Get value of element at a C-contiguous index.*/
    float get(std::uint64_t index) const;
    /** @brief Set value of element at a n-dim index.*/
    void set(const intvec index, float value);
    /** @brief Set value of element at a C-contiguous index.*/
    void set(std::uint64_t index, float value);
    /// @}

    /// @name Read from file
    /// @{
    /** @brief Read metadata from file.*/
    void read_metadata(void);
    /** @brief Copy data from file to a merlin::Array.*/
    void copy_to_array(Array & arr);
    /** @brief Convert to an merlin::Array.*/
    Array to_array(void);
    /// @}

    /// @name Write to file
    /// @{
    /** @brief Get metadata from a merlin::Array.*/
    void get_metadata(Array & src);
    /** @brief Write metadata to file.*/
    void write_metadata(void);
    /** @brief Write data from a merlin::Array to a file.*/
    void write_data_to_file(Array & src);
    /// @}

    /** @brief Destructor.*/
    ~Stock(void);

  protected:
    /** @brief Pointer to file stream.*/
    std::FILE * file_ptr_;
    /** @brief Lock file.*/
    mutable FileLock flock_;
    /** @brief Filename.*/
    std::string filename_;
    /** @brief Open mode.*/
    char mode_;
    /** @brief Thread safe read/write.*/
    bool thread_safe_ = true;
    /** @brief Start writing/reading position wrt. the beginning of file.*/
    std::uint64_t offset_;
};

}  // namespace merlin::array

#endif  // MERLIN_ARRAY_STOCK_HPP_
