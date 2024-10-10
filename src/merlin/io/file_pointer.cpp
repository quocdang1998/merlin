// Copyright 2024 quocdang1998
#include "merlin/io/file_pointer.hpp"

#include <cstddef>     // nullptr
#include <filesystem>  // std::filesystem::filesystem_error
#include <ios>         // std::ios_base::failure

#include "merlin/logger.hpp"  // Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// File deleter
// ---------------------------------------------------------------------------------------------------------------------

// Wrapper for deleter
void io::FileDeleter::operator()(std::FILE * file) const {
    if (file != nullptr) {
        if (std::fclose(file) != 0) {
            Fatal<std::ios_base::failure>("Failed to close file.\n");
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// File pointer
// ---------------------------------------------------------------------------------------------------------------------

// Move cursor to a given position
void io::FilePointer::seek(std::uint64_t cursor) {
    if (std::fseek(this->get(), cursor, SEEK_SET)) {
        Fatal<std::filesystem::filesystem_error>("Seek error.\n");
    }
}

// Create a file if it does not exist, and return a file in append mode
io::FilePointer io::open_file(const char * fname, bool exist_ok) {
    if (std::FILE * read_file = std::fopen(fname, "rb")) {
        std::fclose(read_file);
        if (!exist_ok) {
            Fatal<std::invalid_argument>("File \"%s\" already exist.\n");
        }
    } else {
        std::FILE * create_file = std::fopen(fname, "wb");
        if (create_file == nullptr) {
            Fatal<std::invalid_argument>("Failed to create file \"%s\".\n");
        }
        std::fclose(create_file);
    }
    return io::FilePointer(std::fopen(fname, "rb+"));
}

}  // namespace merlin
