// Copyright 2022 quocdang1998
#include "merlin/cuda/device.hpp"

#include <cstdio>   // std::printf
#include <map>      // std::map
#include <sstream>  // std::ostringstream

#include "cuda.h"  // ::CUcontext, ::cuCtxGetCurrent, ::cuCtxPopCurrent, ::cuCtxPushCurrent, ::cuDeviceGetName

#include "merlin/env.hpp"     // merlin::Environment
#include "merlin/logger.hpp"  // DEBUGLG, WARNING, FAILURE, cuda_runtime_error

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Convert GPU major.minor version to number of CUDA core
// Adapted from function _ConvertSMVer2Cores
// For more info, see https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
static int convert_SM_version_to_core(int major, int minor) {
    static std::map<int, int> num_gpu_arch_cores_per_SM = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60,  64},
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},
        {0x72,  64},
        {0x75,  64},
        {0x80,  64},
        {0x86, 128},
        {0x87, 128},
        {0x89, 128},
        {0x90, 128}
    };
    int SM = (major << 4) + minor;
    if (num_gpu_arch_cores_per_SM.find(SM) == num_gpu_arch_cores_per_SM.end()) {
        FAILURE(cuda_runtime_error, "Cannot detect SM number in the map \"num_gpu_arch_cores_per_SM\".\n");
    }
    return num_gpu_arch_cores_per_SM[SM];
}

// Get current CUDA context
static inline std::uintptr_t get_current_context(void) {
    // check for current context as regular context
    ::CUcontext current_ctx;
    ::cudaError_t err_ = static_cast<::cudaError_t>(::cuCtxGetCurrent(&current_ctx));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get current context failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    // a dummy context initialized (return nullptr)
    return reinterpret_cast<std::uintptr_t>(current_ctx);
}

// ---------------------------------------------------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------------------------------------------------

// Print limit of device
void cuda::Device::print_specification(void) const {
    if (this->id_ == -1) {
        WARNING("Device initialized without a valid id (id = %d).\n", this->id_);
    }
    ::cudaDeviceProp prop;
    ::cudaGetDeviceProperties(&prop, this->id_);
    // Device name
    std::printf("    Name : %s.\n", prop.name);
    // Architechture
    std::printf("    Architechture : %d.%d.\n", prop.major, prop.minor);
    // Max multi-processor
    std::printf("    Number of multiprocessors on the device: %d.\n", prop.multiProcessorCount);
    // Number of CUDA core
    int core_per_multiprocessor = convert_SM_version_to_core(prop.major, prop.minor);
    std::printf("    Number of CUDA core per multiprocessor: %d.\n", core_per_multiprocessor);
    std::printf("    Total number of CUDA core: %d.\n", core_per_multiprocessor * prop.multiProcessorCount);
    // Max thread per multi-processor
    std::printf("    Maximum resident threads per multiprocessor: %d.\n", prop.maxThreadsPerMultiProcessor);
    // Warp size
    std::printf("    Thread-warp size: %d.\n", prop.warpSize);

    // Max threads per block
    std::printf("    Maximum number of threads per block: %d.\n", prop.maxThreadsPerBlock);
    // Max blockDim
    std::printf("    Maximum (x,y,z)-dimension of a block: (%d, %d, %d).\n", prop.maxThreadsDim[0],
                prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    // Max gridDim
    std::printf("    Maximum (x,y,z)-dimension of a grid: (%d, %d, %d).\n", prop.maxGridSize[0], prop.maxGridSize[1],
                prop.maxGridSize[2]);

    // Total global memory
    std::printf("    Total amount of global memory: %f GB.\n", static_cast<float>(prop.totalGlobalMem) / 1073741824.0f);
    // Max shared memory per block
    std::printf("    Maximum amount of shared memory available to a thread block: %zu bytes.\n",
                prop.sharedMemPerBlock);
    // Max register memory per block
    std::printf("    Maximum number of 32-bit register available to a thread block: %d bytes.\n", prop.regsPerBlock);
    // Managed memory support
    std::printf("    Support for managed memory: %s.\n", ((prop.managedMemory == 1) ? "true" : "false"));
    // Max constant memory
    std::printf("    Memory available on device for __constant__ variables in a CUDA C kernel: %zu bytes.\n",
                prop.totalConstMem);
}

// Test functionality of a GPU
bool cuda::Device::test_gpu(void) const {
    // initialize
    int cpu_int[3] = {2, 4, 0};
    int * gpu_int;
    ::cudaError_t err_;
    int reference = cpu_int[0] + cpu_int[1];
    // set device (also change the current context)
    err_ = ::cudaSetDevice(this->id_);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "cudaSetDevice for id = %d failed with message \"%s\".\n", this->id_,
                ::cudaGetErrorName(err_));
    }
    // malloc
    err_ = ::cudaMalloc(&gpu_int, 3 * sizeof(int));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "cudaMalloc failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    // copy to gpu
    err_ = ::cudaMemcpy(gpu_int, cpu_int, 3 * sizeof(int), cudaMemcpyHostToDevice);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "cudaMemcpyHostToDevice failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    // launch kernel
    cuda::add_integers_on_gpu(gpu_int, gpu_int + 1, gpu_int + 2);
    err_ = ::cudaGetLastError();
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Launch kernel failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    // copy to cpu
    err_ = ::cudaMemcpy(cpu_int, gpu_int, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "cudaMemcpyDeviceToHost failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    // check result
    if (cpu_int[2] != reference) {
        WARNING("Expected result of adding %d and %d on GPU ID %d is %d, got %d.\n", cpu_int[0], cpu_int[1], this->id_,
                reference, cpu_int[2]);
        return false;
    }
    return true;
}

// Set as current GPU
void cuda::Device::set_as_current(void) const {
    // set GPU to current context
    ::cudaError_t err_ = ::cudaSetDevice(this->id_);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "cudaSetDevice failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
}

// Push the primary context associated to the GPU to the context stack
std::uintptr_t cuda::Device::push_context(void) const {
    DEBUGLG("Environment::mutex is locked.\n");
    Environment::mutex.lock();
    std::uintptr_t current_context = get_current_context();
    ::cudaError_t err_;
    if (current_context != 0) {
        err_ = static_cast<::cudaError_t>(::cuCtxPushCurrent(reinterpret_cast<::CUcontext>(current_context)));
        if (err_ != 0) {
            FAILURE(cuda_runtime_error, "cuCtxPushCurrent failed with message \"%s\".\n", ::cudaGetErrorName(err_));
        }
    }
    err_ = ::cudaSetDevice(this->id_);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "cudaSetDevice failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    return current_context;
}

// Pop the current context out of the context stack
void cuda::Device::pop_context(std::uintptr_t previous_context) {
    if (previous_context != 0) {
        ::CUcontext ctx;
        ::cudaError_t err_ = static_cast<::cudaError_t>(::cuCtxPopCurrent(&ctx));
        if (err_ != 0) {
            FAILURE(cuda_runtime_error, "cuCtxPopCurrent failed with message \"%s\".\n", ::cudaGetErrorName(err_));
        }
        if (previous_context != reinterpret_cast<std::uintptr_t>(ctx)) {
            FAILURE(std::invalid_argument, "Wrong context provided.\n");
        }
    }
    Environment::mutex.unlock();
    DEBUGLG("Environment::mutex is unlocked.\n");
}

// Get and set GPU limit
std::uint64_t cuda::Device::limit(cuda::DeviceLimit limit, std::uint64_t size) {
    std::uint64_t result;
    if (size == UINT64_MAX) {
        size_t limit_value;
        ::cudaDeviceGetLimit(&limit_value, static_cast<::cudaLimit>(limit));
        result = static_cast<std::uint64_t>(limit_value);
    } else {
        size_t limit_value = static_cast<size_t>(size);
        ::cudaError_t err_ = ::cudaDeviceSetLimit(static_cast<::cudaLimit>(limit), limit_value);
        if (err_ != 0) {
            FAILURE(cuda_runtime_error, "cudaDeviceSetLimit failed with message \"%s\".\n", ::cudaGetErrorName(err_));
        }
        result = size;
    }
    return result;
}

// Reset all GPU
void cuda::Device::reset_all(void) { ::cudaDeviceReset(); }

// Synchronize the current GPU
void cuda::Device::synchronize(void) {
    ::cudaError_t err_ = ::cudaDeviceSynchronize();
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "cudaDeviceSynchronize failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
}

// String representation
std::string cuda::Device::str(void) const {
    char name[256];
    ::cuDeviceGetName(name, sizeof(name), this->id_);
    std::ostringstream os;
    os << "<GPU " << name << ", ID " << this->id_ << ">";
    return os.str();
}

// Print limit of all GPU
void cuda::print_gpus_spec(void) {
    int tot_device = cuda::Device::get_num_gpu();
    for (int i = 0; i < tot_device; i++) {
        std::printf("GPU Id: %d.\n", i);
        cuda::Device gpu(i);
        gpu.print_specification();
    }
}

// Test functionality of all GPU
bool cuda::test_all_gpu(void) {
    int tot_device = cuda::Device::get_num_gpu();
    bool result = true;
    for (int i = 0; i < tot_device; i++) {
        std::printf("Checking device: %d...", i);
        cuda::Device gpu(i);
        result = result && gpu.test_gpu();
        if (!result) {
            WARNING("\rCheck on device %d has failed.\n", i);
        } else {
            std::printf("\r");
        }
    }
    return result;
}

}  // namespace merlin
