// Copyright 2024 quocdang1998
#ifndef MERLIN_IO_DECLARATION_HPP_
#define MERLIN_IO_DECLARATION_HPP_

#include <type_traits>  // std::is_arithmetic_v

namespace merlin::io {

class FileLock;

struct FileDeleter;
class FilePointer;

template <typename T>
requires std::is_arithmetic_v<T>
class IoBuffer;
template <typename T>
requires std::is_arithmetic_v<T>
class ReadEngine;
template <typename T>
requires std::is_arithmetic_v<T>
class WriteEngine;

}  // namespace merlin::io

#endif  // MERLIN_IO_DECLARATION_HPP_
