// Copyright 2022 quocdang1998
#include "merlin/filelock.hpp"

#include <ios>      // std::ios_base::failure
#include <sstream>  // std::ostringstream

#include "merlin/logger.hpp"    // FAILURE
#include "merlin/platform.hpp"  // __MERLIN_LINUX__, __MERLIN_WINDOWS__

#if defined(__MERLIN_WINDOWS__)
    #include <cstring>    // std::memset
    #include <io.h>       // _get_osfhandle
    #include <windows.h>  // LockFileEx, UnlockFileEx, GetLastError
#elif defined(__MERLIN_LINUX__)
    #include <fcntl.h>  // flock
#endif

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// FileLock (Windows)
// ---------------------------------------------------------------------------------------------------------------------

#if defined(__MERLIN_WINDOWS__)

// Constructor from file pointer
FileLock::FileLock(std::FILE * file_ptr) { this->file_descriptor = ::_fileno(file_ptr); }

// Exclusively lock file handle
void FileLock::lock(void) {
    const unsigned long int len = ULONG_MAX;
    static OVERLAPPED ovrlap;
    intptr_t handle = ::_get_osfhandle(this->file_descriptor);
    if (handle == -1) {
        FAILURE(std::ios_base::failure, "Invalid file handle returned.\n");
    }
    bool succeed = ::LockFileEx(reinterpret_cast<void *>(handle), LOCKFILE_EXCLUSIVE_LOCK, 0, len, len, &ovrlap);
    if (!succeed) {
        std::string err_message = throw_windows_last_error(::GetLastError());
        FAILURE(std::ios_base::failure, "Exclusive lock file failed with message \"%s\".\n", err_message.c_str());
    }
}

// Attemp to exclusively lock file
bool FileLock::try_lock(void) {
    const unsigned long int len = ULONG_MAX;
    static OVERLAPPED ovrlap;
    std::memset(&ovrlap, 0, sizeof(OVERLAPPED));
    intptr_t handle = ::_get_osfhandle(this->file_descriptor);
    if (handle == -1) {
        FAILURE(std::ios_base::failure, "Invalid file handle returned.\n");
    }
    bool succeed = ::LockFileEx(reinterpret_cast<void *>(handle), LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                                0, len, len, &ovrlap);
    if (!succeed) {
        unsigned long int err_ = ::GetLastError();
        if (err_ == ERROR_LOCK_VIOLATION) {
            return false;
        }
        std::string err_message = throw_windows_last_error(err_);
        FAILURE(std::ios_base::failure, "Try exclusive lock file failed with message \"%s\".\n", err_message.c_str());
    }
    return true;
}

// Exclusively unlock file handle
void FileLock::unlock(void) {
    const unsigned long int len = ULONG_MAX;
    OVERLAPPED ovrlap;
    std::memset(&ovrlap, 0, sizeof(OVERLAPPED));
    intptr_t handle = ::_get_osfhandle(this->file_descriptor);
    if (handle == -1) {
        FAILURE(std::ios_base::failure, "Invalid file handle returned.\n");
    }
    bool succeed = ::UnlockFileEx(reinterpret_cast<void *>(handle), 0, len, len, &ovrlap);
    if (!succeed) {
        std::string err_message = throw_windows_last_error(::GetLastError());
        FAILURE(std::ios_base::failure, "Unlock exclusively file failed with message \"%s\".\n", err_message.c_str());
    }
}

// Sharable lock file handle
void FileLock::lock_shared(void) {
    const unsigned long int len = ULONG_MAX;
    OVERLAPPED ovrlap;
    std::memset(&ovrlap, 0, sizeof(OVERLAPPED));
    intptr_t handle = ::_get_osfhandle(this->file_descriptor);
    if (handle == -1) {
        FAILURE(std::ios_base::failure, "Invalid file handle returned.\n");
    }
    bool succeed = ::LockFileEx(reinterpret_cast<void *>(handle), 0, 0, len, len, &ovrlap);
    if (!succeed) {
        std::string err_message = throw_windows_last_error(::GetLastError());
        FAILURE(std::ios_base::failure, "Shared lock file failed with message \"%s\".\n", err_message.c_str());
    }
}

// Attemp to sharably lock file
bool FileLock::try_lock_shared(void) {
    const unsigned long int len = ULONG_MAX;
    static OVERLAPPED ovrlap;
    std::memset(&ovrlap, 0, sizeof(OVERLAPPED));
    intptr_t handle = ::_get_osfhandle(this->file_descriptor);
    if (handle == -1) {
        FAILURE(std::ios_base::failure, "Invalid file handle returned.\n");
    }
    bool succeed = ::LockFileEx(reinterpret_cast<void *>(handle), LOCKFILE_FAIL_IMMEDIATELY, 0, len, len, &ovrlap);
    if (!succeed) {
        unsigned long int err_ = ::GetLastError();
        if (err_ == ERROR_LOCK_VIOLATION) {
            return false;
        }
        std::string err_message = throw_windows_last_error(err_);
        FAILURE(std::ios_base::failure, "Try shared lock file failed with message \"%s\".\n", err_message.c_str());
    }
    return true;
}

#endif  // __MERLIN_WINDOWS__

// ---------------------------------------------------------------------------------------------------------------------
// FileLock (Linux)
// ---------------------------------------------------------------------------------------------------------------------

#if defined(__MERLIN_LINUX__)

// Constructor from file pointer
FileLock::FileLock(std::FILE * file_ptr) { this->file_descriptor = ::fileno(file_ptr); }

// Exclusively lock file handle
void FileLock::lock(void) {
    ::flock lock_;
    lock_.l_type = F_WRLCK;
    lock_.l_whence = SEEK_SET;
    lock_.l_start = 0;
    lock_.l_len = 0;
    int err_ = ::fcntl(this->file_descriptor, F_SETLKW, &lock_);
    if (err_ == -1) {
        std::string err_message = throw_linux_last_error();
        FAILURE(std::ios_base::failure, "Exclusive lock file failed with message \"%s\".\n", err_message.c_str());
    }
}

// Attemp to exclusively lock file
bool FileLock::try_lock(void) {
    ::flock lock_;
    lock_.l_type = F_WRLCK;
    lock_.l_whence = SEEK_SET;
    lock_.l_start = 0;
    lock_.l_len = 0;
    int err_ = ::fcntl(this->file_descriptor, F_SETLK, &lock_);
    if (err_ == -1) {
        if (errno == EAGAIN || errno == EACCES) {
            return false;
        }
        std::string err_message = throw_linux_last_error();
        FAILURE(std::ios_base::failure, "Try exclusive lock file failed with message \"%s\".\n", err_message.c_str());
    }
    return true;
}

// Exclusively unlock file handle
void FileLock::unlock(void) {
    ::flock lock_;
    lock_.l_type = F_UNLCK;
    lock_.l_whence = SEEK_SET;
    lock_.l_start = 0;
    lock_.l_len = 0;
    int err_ = ::fcntl(this->file_descriptor, F_SETLK, &lock_);
    if (err_ == -1) {
        std::string err_message = throw_linux_last_error();
        FAILURE(std::ios_base::failure, "Unlock exclusively file failed with message \"%s\".\n", err_message.c_str());
    }
}

// Sharable lock file handle
void FileLock::lock_shared(void) {
    ::flock lock_;
    lock_.l_type = F_RDLCK;
    lock_.l_whence = SEEK_SET;
    lock_.l_start = 0;
    lock_.l_len = 0;
    int err_ = ::fcntl(this->file_descriptor, F_SETLKW, &lock_);
    if (err_ == -1) {
        std::string err_message = throw_linux_last_error();
        FAILURE(std::ios_base::failure, "Exclusive lock file failed with message \"%s\".\n", err_message.c_str());
    }
}

// Attemp to sharably lock file
bool FileLock::try_lock_shared(void) {
    ::flock lock_;
    lock_.l_type = F_RDLCK;
    lock_.l_whence = SEEK_SET;
    lock_.l_start = 0;
    lock_.l_len = 0;
    int err_ = ::fcntl(this->file_descriptor, F_SETLK, &lock_);
    if (err_ == -1) {
        if (errno == EAGAIN || errno == EACCES) {
            return false;
        }
        std::string err_message = throw_linux_last_error();
        FAILURE(std::ios_base::failure, "Try exclusive lock file failed with message \"%s\".\n", err_message.c_str());
    }
    return true;
}

#endif  // __MERLIN_LINUX__

// String representation
std::string FileLock::str() const {
    std::ostringstream os;
    os << "<Filelock for file descriptor: " << this->file_descriptor << ">";
    return os.str();
}

}  // namespace merlin
