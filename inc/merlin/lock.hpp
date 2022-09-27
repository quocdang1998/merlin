#ifndef MERLIN_LOCK_HPP_
#define MERLIN_LOCK_HPP_

namespace merlin {

/** @brief Create a critical section allowing only one thread block executes at a time.*/
class KernelLock {
  public:
    KernelLock(void) {
        cudaMalloc(&this->state_, sizeof(int));
        int temp = 0;
        cudaMemcpy(this->state_, &temp, sizeof(int), cudaMemcpyHostToDevice);
        this->reference_count_ = 1;
    }

    KernelLock(const KernelLock & src) : state_(src.state_) {
        this->reference_count_ = src.reference_count_ + 1;
    }

    KernelLock & operator=(const KernelLock & src) {
        this->state_ = src.state_;
        this->reference_count_ = src.reference_count_ + 1;
        return *this;
    }

    KernelLock(KernelLock && src) = delete;
    KernelLock & operator=(KernelLock && src) = delete;

    __device__ void lock(void) {
        if (threadIdx.x == 0) {
            while (atomicCAS(this->state_, 0, 1) != 0) {}
        }
    }
    __device__ void unlock(void) {
        if (threadIdx.x == 0) {
            atomicExch(this->state_, 0);
        }
    }

    ~KernelLock(void) {
        if (this->reference_count_ == 1) {
            cudaFree(this->state_);
        }
    }

  private:
    int * state_ = NULL;
    int reference_count_ = 0;
};

}  // namespace merlin

#endif  // MERLIN_LOCK_HPP_
