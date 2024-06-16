// Copyright 2022 quocdang1998
#include "merlin/cuda/graph.hpp"

#include "merlin/logger.hpp"  // cuda_compile_error, FAILURE

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// GraphNode
// ---------------------------------------------------------------------------------------------------------------------

// Empty beginning of a graph
const std::vector<cuda::GraphNode> cuda::begin_graphnode;

#ifndef __MERLIN_CUDA__

// Wrapper of the function adding CUDA callback to graph
cuda::GraphNode cuda::add_callback_to_graph(std::uintptr_t graph_ptr, cuda::GraphCallback functor,
                                            const std::vector<cuda::GraphNode> & deps, void * arg) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return cuda::GraphNode(0);
}

#endif  // __MERLIN_CUDA__

// ---------------------------------------------------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// Destroy current CUDA graph instance
void cuda::Graph::destroy_graph(void) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
}

// Constructor
cuda::Graph::Graph(int flag) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
}

// Copy constructor
cuda::Graph::Graph(const cuda::Graph & src) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
}

// Copy assignment
cuda::Graph & cuda::Graph::operator=(const cuda::Graph & src) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return *this;
}

// Move constructor
cuda::Graph::Graph(cuda::Graph && src) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
}

// Move assignment
cuda::Graph & cuda::Graph::operator=(cuda::Graph && src) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return *this;
}

// Get number of nodes in a graph
std::uint64_t cuda::Graph::get_num_nodes(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return 0;
}

// Get node list
std::vector<cuda::GraphNode> cuda::Graph::get_node_list(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return std::vector<cuda::GraphNode>();
}

// Get number of edges
std::uint64_t cuda::Graph::get_num_edges(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return 0;
}

// Get edge list
std::vector<std::array<cuda::GraphNode, 2>> cuda::Graph::get_edge_list(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return std::vector<std::array<cuda::GraphNode, 2>>();
}

// Add memory allocation node
std::tuple<cuda::GraphNode, void *> cuda::Graph::add_malloc_node(std::uint64_t size,
                                                                 const std::vector<cuda::GraphNode> & deps) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return std::tuple<cuda::GraphNode, void *>();
}

// Add memcpy node
cuda::GraphNode cuda::Graph::add_memcpy_node(void * dest, const void * src, std::uint64_t size,
                                             cuda::MemcpyKind copy_flag, const std::vector<cuda::GraphNode> & deps) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return cuda::GraphNode(0);
}

// Add CUDA deallocation node
cuda::GraphNode cuda::Graph::add_memfree_node(void * ptr, const std::vector<cuda::GraphNode> & deps) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return cuda::GraphNode(0);
}

// Add CUDA event record node
cuda::GraphNode cuda::Graph::add_event_record_node(const cuda::Event & event,
                                                   const std::vector<cuda::GraphNode> & deps) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return cuda::GraphNode(0);
}

// Add CUDA event wait node
cuda::GraphNode cuda::Graph::add_event_wait_node(const cuda::Event & event, const std::vector<cuda::GraphNode> & deps) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return cuda::GraphNode(0);
}

// Add CUDA child graph node
cuda::GraphNode cuda::Graph::add_child_graph_node(const cuda::Graph & child_graph,
                                                  const std::vector<cuda::GraphNode> & deps) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return cuda::GraphNode(0);
}

// Export graph into DOT file
void cuda::Graph::export_to_dot(const std::string & filename) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
}

// Execute a graph (add detecting errored node)
void cuda::Graph::execute(const cuda::Stream & stream) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
}

// Destructor
cuda::Graph::~Graph(void) {}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
