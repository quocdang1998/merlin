// Copyright 2022 quocdang1998
#ifndef MERLIN_FILELOCK_HPP_
#define MERLIN_FILELOCK_HPP_

#include <cstdint>  // std::uintptr_t
#include <cstdio>  // std::FILE
#include <string>  // std::string

#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin {

/** @brief Mutex for file.
 *  @details Force the current thread/process to wait if the file is locked by another process. The execution is
 *  continued once the lock on the file is released. It works similar to a ``std::shared_mutex`` using directly the file as
 *  the mutex for interprocess communication.
 *
 *  Two types of locks are provided:
 *  - **shared** : multiple threads and processes can access the file at a time, but exclusive lock must wait until all
 *  shared locks are released. **Use this type of lock for reading file**.
 *  - **exclusive** : only one thread of a process can access the file at a time, tasks with share locks must wait
 *  until the exclusive lock is released. **Use this type of lock for writing file**.
 *  @note Locking mecanism depends on operating system:
 *   - On **Windows**, the lock is binded to read/write permission. Shared lock only blocks write permission, but
 *  exclusive lock blocks both. If another file stream ``std::FILE`` binded to the locked file is created by the same
 *  thread (or another thread of the same process, or another process) ``fread`` and ``fwrite`` will successfully
 *  return, but **file content cannot be read/write**.
 *   - On **Linux**, the lock is not binded to read/write permission. A thread/process **without locking can read/write
 *  the file normally** even if the file is locked by another.
 */
class FileLock {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    FileLock(void) = default;
    /** @brief Constructor from C file stream pointer.*/
    MERLIN_EXPORTS FileLock(std::FILE * file_ptr);
    /// @}

    /// @name Exclusive lock
    /// @{
    /** @brief Exclusively lock file.*/
    MERLIN_EXPORTS void lock(void);
    /** @brief Attemp to exclusively lock file.
     * @return ``True`` if lock succeeds and file is locked. ``False`` otherwise.
     */
    MERLIN_EXPORTS bool try_lock(void);
    /// @}

    /// @name Share lock
    /// @{
    /** @brief Sharable lock file.*/
    MERLIN_EXPORTS void lock_shared(void);
    /** @brief Attemp to sharable lock file.
     * @return ``True`` if lock succeeds and file is locked. ``False`` otherwise.
     */
    MERLIN_EXPORTS bool try_lock_shared(void);
    /// @}

    /// @name Unlock
    /// @{
    /** @brief Exclusively unlock file.*/
    MERLIN_EXPORTS void unlock(void);
    /// @}

    /// @name String representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str() const;

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~FileLock(void) = default;
    /// @}

  private:
    /** @brief POSIX file descriptor.*/
    int file_descriptor = 0;
};

}  // namespace merlin

#endif  // MERLIN_FILELOCK_HPP_
