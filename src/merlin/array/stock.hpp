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
     *    - w: Write only.
     *    - a: Read and write (equivalent to ``r+``).
     *    - p: Parallel write (multiple processes can write at the same time).
     *    - s: Shared (multiple processes can read and write at the same time).
     *  @param offset Starting position from the beginning of file.
     *  @note In mode ``p`` and mode ``s``, user is responsible to prevent data race (each process reads/writes a
     *  different subset of the data file).
     */
    Stock(const std::string & filename, char mode = 'a', std::uint64_t offset = 0);
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
    std::FILE * file_ptr_;
    FileLock flock_;
    std::string filename_;
    char mode_;
    std::uint64_t offset_;
};

}  // namespace merlin::array

#endif  // MERLIN_ARRAY_STOCK_HPP_
