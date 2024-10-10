// Copyright 2022 quocdang1998
#ifndef MERLIN_IO_FILE_LOCK_HPP_
#define MERLIN_IO_FILE_LOCK_HPP_

#include <cstdint>  // std::uintptr_t
#include <cstdio>   // std::FILE
#include <string>   // std::string

#include "merlin/exports.hpp"         // MERLIN_EXPORTS
#include "merlin/io/declaration.hpp"  // merlin::io::FileLock

namespace merlin {

/** @brief Interprocess file lock.
 *  @details Pause the current thread/process if the target file is currently used by another process. The execution is
 *  continued when the lock is released. It works similar to a ``std::shared_mutex``, but utilizes the file as the
 *  indicator in interprocess communication.
 *
 *  Two types of locks are provided:
 *  - **shared** : multiple threads/processes with shared lock can access the file at the same time, but those with
 *    exclusive lock must wait until all shared locks are released. **This type of lock is used for processes reading
 *    the target file**.
 *  - **exclusive** : only one thread/process can access the file at a time, and all other tasks must wait until the
 *    lock is released. **Use this type of lock for processes writing the target file**.
 *  @note Locking mechanism depends on operating system:
 *  - On **Windows**, the lock is bound to read/write permission. Shared lock only blocks write permission, but
 *    exclusive lock blocks both. If another file stream ``std::FILE`` associated to the locked file is created,
 *    read/write operations when the lock is acquired will successfully return, but the operation will have no effect.
 *  - On **Linux**, the lock is not bound to read/write permission. Any other thread/process can read/write the file
 *    normally even when the file is locked by another process.
 */
class io::FileLock {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    FileLock(void) = default;
    /** @brief Constructor from C file stream pointer.
     *  @details Create a file lock and associate it with the file stream pointer provided.
     *  @param file_ptr File stream pointer to associate the file lock with.
     */
    MERLIN_EXPORTS FileLock(std::FILE * file_ptr);
    /// @}

    /// @name Exclusive lock
    /// @{
    /** @brief Exclusively lock file.
     *  @details Lock the file so that only the current process can access the file. This type of lock should be used
     *  to avoid race condition of writing processes.
     *  @warning Double lock a file may lead to an infinite loop.
     */
    MERLIN_EXPORTS void lock(void);
    /** @brief Attempt to exclusively lock file.
     *  @details Lock the file with exclusive lock and fail immediately if the file is currently locked by another
     *  process.
     *  @return ``True`` if lock succeeds and file is locked. ``False`` otherwise.
     */
    MERLIN_EXPORTS bool try_lock(void);
    /// @}

    /// @name Shared lock
    /// @{
    /** @brief Shared lock file.
     *  @details Only processes with shared lock can access the file. Exclusive locks are paused until all shared locks
     *  are released. This type of lock should be used to avoid race condition of reading processes.
     */
    MERLIN_EXPORTS void lock_shared(void);
    /** @brief Attempt to shared lock file.
     *  @details Lock the file with shared lock and fail immediately if the file is currently locked by another
     *  process.
     *  @return ``True`` if lock succeeds and file is locked. ``False`` otherwise.
     */
    MERLIN_EXPORTS bool try_lock_shared(void);
    /// @}

    /// @name Unlock
    /// @{
    /** @brief Unlock the file.
     *  @details Unlock exclusive lock imposed on the file by the current process.
     */
    MERLIN_EXPORTS void unlock(void);
    /** @brief Unlock the file.
     *  @details Unlock shared lock imposed on the file by the current process.
     */
    void unlock_shared(void) { this->unlock(); }
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str() const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~FileLock(void) = default;
    /// @}

  private:
    /** @brief POSIX file descriptor.
     *  @details Under POSIX norm, each file stream invoked by the OS is associated to an integer called the file
     *  descriptor. The descriptor is used to indicate the file to be locked to the OS.
     */
    int file_descriptor = 0;
};

}  // namespace merlin

#endif  // MERLIN_IO_FILE_LOCK_HPP_
