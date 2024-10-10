// Copyright 2024 quocdang1998
#ifndef MERLIN_IO_FILE_POINTER_HPP_
#define MERLIN_IO_FILE_POINTER_HPP_

#include <cstdio>   // std::FILE
#include <memory>   // std::unique_ptr

#include "merlin/exports.hpp"         // MERLIN_EXPORTS
#include "merlin/io/declaration.hpp"  // merlin::io::FileDeleter

namespace merlin {

// File deleter
// ------------

/** @brief Wrapper for deleter.*/
struct io::FileDeleter {
    MERLIN_EXPORTS void operator()(std::FILE * file) const;
};

// File pointer
// ------------

namespace io {

/** @brief File pointer.*/
class FilePointer : public std::unique_ptr<std::FILE, io::FileDeleter> {
  public:
    /** @brief Default constructor.*/
    FilePointer(void) : std::unique_ptr<std::FILE, io::FileDeleter>(nullptr) {}
    /** @brief Constructor from file pointer.*/
    FilePointer(std::FILE * fp) : std::unique_ptr<std::FILE, io::FileDeleter>(fp) {}
    /** @brief Move cursor to a given position.*/
    MERLIN_EXPORTS void seek(std::uint64_t cursor);
};

/** @brief Open or create a file if it does not exist, and return a file in append mode.
 *  @param fname Name of the file.
 *  @param exist_ok Flag indicating whether to throw an error if the file already exist.
 */
MERLIN_EXPORTS FilePointer open_file(const char * fname, bool exist_ok = true);

}  // namespace io

}  // namespace merlin

#endif  // MERLIN_IO_FILE_POINTER_HPP_
