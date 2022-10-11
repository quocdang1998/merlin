// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_STOCK_HPP_
#define MERLIN_ARRAY_STOCK_HPP_

#include <cstdint>  // std::uint64_t
#include <fstream>  // std::fstream
#include <mutex>  // std::mutex
#include <string>  // std::string

#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/array/nddata.hpp"  // merlin::array::NdData

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
     *    - a: Read and write.
     */
    Stock(const std::string & filename, char mode = 'a');
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    Stock(const Stock & src) = delete;
    /** @brief Copy assignment.*/
    Stock & operator=(const Stock & src) = delete;
    /** @brief Move constructor.*/
    Stock(Stock && src) = default;
    /** @brief Move assignment.*/
    Stock & operator=(Stock && src) = default;
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get reference to the file stream.*/
    std::fstream & file_stream(void) const {return this->file_stream_;}
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
    std::string filename_;
    char mode_;
    std::uint64_t stream_pos_;
    std::ios_base::fmtflags format_flag_;

  private:
    mutable std::fstream file_stream_;
    void get_fstream_metadata(void);
    void reopen_fstream(void);
    static std::mutex mutex_;
    static std::ios_base::openmode char_to_openmode(char mode);
};

}  // namespace merlin::array

#endif  // MERLIN_ARRAY_STOCK_HPP_
