// Copyright 2022 quocdang1998
#include "merlin/device/gpu_query.hpp"

#include <cstdio>  // std::printf
#include <map>  // std::map

#include "merlin/logger.hpp"  // FAILURE, cuda_runtime_error

namespace merlin {

// Get total number of GPU
int get_device_count(void) {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

// Convert GPU major.minor version to number of CUDA core
// Adapted from function _ConvertSMVer2Cores, see https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
static int convert_SM_version_to_core(int major, int minor) {
    std::map<int, int> num_gpu_arch_cores_per_SM = {
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
        {0x87, 128}
    };
    int SM = (major << 4) + minor;
    if (num_gpu_arch_cores_per_SM.find(SM) == num_gpu_arch_cores_per_SM.end()) {
        FAILURE(cuda_runtime_error, "Cannot detect SM number in the map \"num_gpu_arch_cores_per_SM\".\n");
    }
    return num_gpu_arch_cores_per_SM[SM];
}

// Get limit of a given device
static void print_limit_of_one_device(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // Device name
    std::printf("    Name : %s.\n", prop.name);
    // Max multi-processor
    std::printf("    Number of multiprocessors on the device: %d.\n", prop.multiProcessorCount);
    // Number of CUDA core
    int core_per_multiprocessor = convert_SM_version_to_core(prop.major, prop.minor);
    std::printf("    Number of CUDA core per multiprocessor: %d.\n", core_per_multiprocessor);
    std::printf("    Total number of CUDA core: %d.\n", core_per_multiprocessor*prop.multiProcessorCount);
    // Max thread per multi-processor
    std::printf("    Maximum resident threads per multiprocessor: %d.\n", prop.maxThreadsPerMultiProcessor);

    // Max threads per block
    std::printf("    Maximum number of threads per block: %d.\n", prop.maxThreadsPerBlock);
    // Max blockDim
    std::printf("    Maximum (x,y,z)-dimension of a block: (%d, %d, %d).\n",
                prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    // Max gridDim
    std::printf("    Maximum (x,y,z)-dimension of a grid: (%d, %d, %d).\n",
                prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    // Total global memory
    std::printf("    Total amount of global memory: %f GB.\n", static_cast<float>(prop.totalGlobalMem)/1073741824.0f);
    // Max shared memory per block
    std::printf("    Maximum amount of shared memory available to a thread block: %zu bytes.\n", prop.sharedMemPerBlock);
    // Max constant memory
    std::printf("    Memory available on device for __constant__ variables in a CUDA C kernel: %zu bytes.\n",
                prop.totalConstMem);
}

// Print limit of a device
void print_device_limit(int device) {
    int tot_device = get_device_count();
    if (device != -1) {
        // Check valid argument
        if ((device < 0) || (device >= tot_device)) {
            FAILURE(std::invalid_argument, "Invalid number of device (expected value >= 0 and < %d, got %d.\n",
                    tot_device, device);
        }
        // Print only one device
        std::printf("Chosen device: %d.\n", device);
        print_limit_of_one_device(device);
        return;
    }
    // if no device is selected, query for all devices
    for (int i = 0; i < tot_device; i++) {
        std::printf("GPU Id: %d.\n", i);
        print_limit_of_one_device(i);
    }
}

// Add 2 integer on GPU
__global__ static void add_2_int_on_gpu(int * p_a, int * p_b, int * p_result) {
    *p_result = *p_a + *p_b;
}

// Test functionality of one GPU
static bool test_gpu_on_a_device(int device) {
    // initialize
    int cpu_int[3] = {2, 4, 0};
    int * gpu_int;
    cudaError_t err_;
    int reference = cpu_int[0] + cpu_int[1];
    // set device
    err_ = cudaSetDevice(device);
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "cudaSetDevice for id = %d failed with message \"%s\".\n",
                device, cudaGetErrorName(err_));
    }
    // malloc
    err_ = cudaMalloc(&gpu_int, 3*sizeof(int));
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "cudaMalloc failed with message \"%s\".\n", cudaGetErrorName(err_));
    }
    // copy to gpu
    err_ = cudaMemcpy(gpu_int, cpu_int, 3*sizeof(int), cudaMemcpyHostToDevice);
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "cudaMemcpyHostToDevice failed with message \"%s\".\n", cudaGetErrorName(err_));
    }
    // launch kernel
    add_2_int_on_gpu<<<1, 1>>>(gpu_int, gpu_int+1, gpu_int+2);
    cudaDeviceSynchronize();
    err_ = cudaGetLastError();
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "Launch kernel failed with message \"%s\".\n", cudaGetErrorName(err_));
    }
    // copy to cpu
    err_ = cudaMemcpy(cpu_int, gpu_int, 3*sizeof(int), cudaMemcpyDeviceToHost);
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "cudaMemcpyDeviceToHost failed with message \"%s\".\n", cudaGetErrorName(err_));
    }
    // check result
    if (cpu_int[2] != reference) {
        WARNING("Expected result of adding %d and %d on GPU ID %d is %d, got %d.\n",
                cpu_int[0], cpu_int[1], device, reference, cpu_int[2]);
        return false;
    }
    return true;
}

// Test functionality of GPU
bool test_gpu(int device) {
    int tot_device = get_device_count();
    bool result = true;
    if (device != -1) {
        // Check valid argument
        if ((device < 0) || (device >= tot_device)) {
            FAILURE(std::invalid_argument, "Invalid number of device (expected value >= 0 and < %d, got %d.\n",
                    tot_device, device);
        }
        // Test for only one device
        std::printf("Checking device: %d...", device);
        result = result && test_gpu_on_a_device(device);
        if (!result) {
            WARNING("\rCheck on device %d has failed.\n", device);
        } else {
            std::printf("\r");
        }
        return result;
    }
    // if no device is selected, test for all devices
    for (int i = 0; i < tot_device; i++) {
        std::printf("Checking device: %d...", i);
        result = result && test_gpu_on_a_device(i);
        if (!result) {
            WARNING("\rCheck on device %d has failed.\n", i);
        } else {
            std::printf("\r");
        }
    }
    return result;
}

}  // namespace merlin
