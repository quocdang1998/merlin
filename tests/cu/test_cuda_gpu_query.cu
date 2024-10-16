#include <cinttypes>
#include <functional>
#include <memory>
#include <vector>

#include "merlin/cuda/device.hpp"
#include "merlin/cuda/enum_wrapper.hpp"
#include "merlin/cuda/event.hpp"
#include "merlin/cuda/graph.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/cuda/event.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"

using namespace merlin;

__global__ void foo(void) {
    std::printf("Static CUDA graph kernel execution.\n");
}

void str_callback(const std::string & s, std::vector<int> && vec) {
    Message m("{}:", s);
    for (int & v : vec) {
        m << v << " ";
    }
    m << "\n";
}

__global__ void print_array(double * array) {
    CudaOut("Print array element (dynamic graph): %f.\n", array[flatten_kernel_index()]);
}


void graph_callback(std::vector<int> && vec) {
    Message m("Graph callback function for dynamic graph:");
    for (int & v : vec) {
        m << v << " ";
    }
    m << "\n";
}

int main(void) {
    // set GPU
    cuda::Device gpu(0);
    gpu.set_as_current();

    // print GPU specifications
    Message("Test 1 : GPU specification\n");
    cuda::print_gpus_spec();

    // get GPU limits
    Message m2("Test 2 : GPU limits\n");
    std::uint64_t stack_size = cuda::Device::limit(cuda::DeviceLimit::StackSize);
    m2 << "Stack size: " << stack_size << ".\n";
    cuda::test_all_gpu();

    // static graph with callback
    Message m3("Test 3 : Static CUDA Graph\n");
    cuda::Stream s(cuda::StreamSetting::NonBlocking);
    m3 << "Default stream: " << s.str() << ".\n";
    s.check_cuda_context();
    cuda::begin_capture_stream(s);
    ::foo<<<2, 2, 0, reinterpret_cast<::cudaStream_t>(s.get_stream_ptr())>>>();
    cuda::Graph static_graph = cuda::end_capture_stream(s);
    static_graph.execute(s);
    std::string data = "static callback message!";
    // s.add_callback(str_callback, data, 4);
    s.add_callback(str_callback, std::cref(data), std::vector<int>({1, 2, 5, 4}));
    s.add_callback(str_callback, std::cref(data), std::vector<int>({6, 8, 7, 3}));
    s.add_callback([&data](void) { Message("Lambda capturing by l-value called \"{}\"\n", data); });
    std::unique_ptr<std::string> p_data = std::make_unique<std::string>(data);
    s.add_callback([p_data = std::move(p_data)](void) {
        Message("Lambda capturing by r-value called \"{}\"\n", *p_data);
    });
    s.synchronize();

    // dynamic graph with callback
    Message("Test 4 : Dynamic CUDA Graph\n");
    cuda::Event ev(cuda::EventCategory::BlockingSync | cuda::EventCategory::DisableTiming);
    cuda::Graph dynamic_graph(0);
    auto [mem_alloc_node, data_ptr_void] = dynamic_graph.add_malloc_node(4 * sizeof(double), cuda::begin_graphnode);
    double * data_ptr = reinterpret_cast<double *>(data_ptr_void);
    double data_cpu[4] = {5.8, 4.3, 2.5, 1.6};
    cuda::GraphNode memcpy_node = dynamic_graph.add_memcpy_node(data_ptr, data_cpu, 2 * sizeof(double),
                                                                cuda::MemcpyKind::HostToDevice, {mem_alloc_node});
    cuda::GraphNode kernel_node = dynamic_graph.add_kernel_node(&print_array, 1, 2, 0, {memcpy_node}, data_ptr);
    cuda::GraphNode memcpy_node2 = dynamic_graph.add_memcpy_node(data_ptr + 2, data_cpu + 2, 2 * sizeof(double),
                                                                 cuda::MemcpyKind::HostToDevice, {mem_alloc_node});
    cuda::GraphNode kernel_node2 = dynamic_graph.add_kernel_node(&print_array, 1, 2, 0, {memcpy_node2}, data_ptr + 2);
    cuda::GraphNode event_record_node = dynamic_graph.add_event_record_node(ev, {kernel_node});
    cuda::GraphNode event_wait_node = dynamic_graph.add_event_wait_node(ev, {kernel_node2});
    cuda::GraphNode memfree_node = dynamic_graph.add_memfree_node(data_ptr, {kernel_node, kernel_node2});
    cuda::GraphNode graph_callback_node = dynamic_graph.add_host_node(graph_callback, {memfree_node},
                                                                      std::vector<int>({0, 9, 1, 7}));
    dynamic_graph.export_to_dot("output.dot");
    dynamic_graph.execute(s);
    s.synchronize();
}

