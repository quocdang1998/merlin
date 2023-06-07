#include "merlin/cuda/device.hpp"
#include "merlin/cuda/context.hpp"
#include "merlin/cuda/event.hpp"
#include "merlin/cuda/graph.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/cuda/event.hpp"

#include "merlin/logger.hpp"
#include "merlin/utils.hpp"

#include <cinttypes>

__global__ void foo(void) {
    CUDAOUT("A message.\n");
}

__global__ void print_array(double * array) {
    CUDAOUT("Print array element: %f.\n", array[merlin::flatten_kernel_index()]);
}

void str_callback(::cudaStream_t stream, ::cudaError_t status, void * data) {
    int & a = *(static_cast<int *>(data));
    MESSAGE("Callback argument: %d\n", a);
}

void graph_callback(void * data) {
    MESSAGE("Graph callback function called.\n");
}

int main(void) {
    merlin::cuda::print_all_gpu_specification();
    std::uint64_t stack_size = merlin::cuda::Device::limit(merlin::cuda::DeviceLimit::StackSize);
    MESSAGE("Stack size: %" PRIu64 ".\n", stack_size);
    merlin::cuda::test_all_gpu();

    merlin::cuda::Context new_ctx(0, merlin::cuda::ContextSchedule::BlockSync);
    merlin::cuda::Context c = merlin::cuda::Context::get_current();
    MESSAGE("Current context: %s.\n", c.str().c_str());

    merlin::cuda::Event ev(merlin::cuda::EventCategory::BlockingSyncEvent | merlin::cuda::EventCategory::DisableTimingEvent);

    merlin::cuda::Stream s(merlin::cuda::StreamSetting::NonBlocking);
    MESSAGE("Default stream: %s.\n", s.str().c_str());
    int data = 1;
    s.check_cuda_context();
    merlin::cuda::begin_capture_stream(s);
    ::foo<<<2, 2, 0, reinterpret_cast<cudaStream_t>(s.get_stream_ptr())>>>();
    merlin::cuda::Graph g = merlin::cuda::end_capture_stream(s);
    g.execute(s);
    s.add_callback(str_callback, &data);
    s.synchronize();

    merlin::cuda::Graph dynamic_graph(0);
    auto [mem_alloc_node, data_ptr_void] = dynamic_graph.add_mem_alloc_node(4*sizeof(double),
                                                                            merlin::cuda::begin_graphnode);
    double * data_ptr = reinterpret_cast<double *>(data_ptr_void);
    double data_cpu[4] = {5.8, 4.3, 2.5, 1.6};
    auto memcpy_node = dynamic_graph.add_memcpy_node(data_ptr, data_cpu, 2*sizeof(double),
                                                     merlin::cuda::MemcpyKind::HostToDevice, {mem_alloc_node});
    auto kernel_node = dynamic_graph.add_kernel_node(&print_array, 1, 2, 0, {memcpy_node}, data_ptr);
    auto memcpy_node2 = dynamic_graph.add_memcpy_node(&data_ptr[2], &data_cpu[2], 2*sizeof(double),
                                                      merlin::cuda::MemcpyKind::HostToDevice, {mem_alloc_node});
    data_ptr += 2;
    auto kernel_node2 = dynamic_graph.add_kernel_node(&print_array, 1, 2, 0, {memcpy_node2}, data_ptr);
    auto event_record_node = dynamic_graph.add_event_record_node(ev, {kernel_node});
    auto event_wait_node = dynamic_graph.add_event_wait_node(ev, {kernel_node2});
    auto memfree_node = dynamic_graph.add_memfree_node(data_ptr-2, {kernel_node, kernel_node2});
    auto graph_callback_node = dynamic_graph.add_host_node(&graph_callback, {memfree_node});
    dynamic_graph.export_to_dot("output.dot");
    dynamic_graph.execute(s);
    s.synchronize();

}
