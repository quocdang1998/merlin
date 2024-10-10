// Copyright 2024 quocdang1998
#include "merlin/io/io_engine.hpp"

#include <filesystem>  // std::filesystem::filesystem_error

#include "merlin/logger.hpp"  // merlin::Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Get the number of element
std::size_t io::get_count(std::uint64_t size, std::size_t element_size) {
    if (size % element_size != 0) {
        Fatal<std::invalid_argument>("Size must be a multiple of element size.\n");
    }
    return size / element_size;
}

// ---------------------------------------------------------------------------------------------------------------------
// Read engine
// ---------------------------------------------------------------------------------------------------------------------

// Read data from file at the current cursor
void io::read_data(std::FILE * file, void * dest, std::size_t count, std::size_t element_size) {
    std::size_t success_read = std::fread(dest, element_size, count, file);
    if (success_read < count) {
        Fatal<std::filesystem::filesystem_error>("Error occurred when reading the file.\n");
    }
}

// Read data from file at a specified location
void io::read_data(std::FILE * file, void * dest, const void * src, std::size_t count, std::size_t element_size) {
    if (std::fseek(file, reinterpret_cast<std::uintptr_t>(src), SEEK_SET)) {
        Fatal<std::filesystem::filesystem_error>("Seek cursor error.\n");
    }
    io::read_data(file, dest, count, element_size);
}

// ---------------------------------------------------------------------------------------------------------------------
// Write engine
// ---------------------------------------------------------------------------------------------------------------------

// Write data to file at the current cursor
void io::write_data(std::FILE * file, const void * src, std::size_t count, std::size_t element_size) {
    std::size_t success_write = std::fwrite(src, element_size, count, file);
    if (success_write < count) {
        Fatal<std::filesystem::filesystem_error>("Error occurred when writing the file.\n");
    }
}

// Write data to file at a specified location
void io::write_data(std::FILE * file, void * dest, const void * src, std::size_t count, std::size_t element_size) {
    if (std::fseek(file, reinterpret_cast<std::uintptr_t>(dest), SEEK_SET)) {
        Fatal<std::filesystem::filesystem_error>("Seek cursor error.\n");
    }
    io::write_data(file, src, count, element_size);
}

}  // namespace merlin
