// Copyright 2022 quocdang1998
#include "merlin/cuda/graph.hpp"

#include <cstddef>  // std::size_t
#include <cstring>  // std::memset

#include "merlin/cuda/device.hpp"  // merlin::cuda::Device
#include "merlin/cuda/event.hpp"   // merlin::cuda::Event
#include "merlin/cuda/stream.hpp"  // merlin::cuda::Stream
#include "merlin/logger.hpp"       // merlin::Fatal, merlin::Warning, merlin::cuda_runtime_error

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Wrapper of the function adding CUDA callback to graph
cuda::GraphNode cuda::add_callback_to_graph(std::uintptr_t graph_ptr, cuda::GraphCallback functor,
                                            const cuda::GraphNodeList & deps, void * arg) {
    ::cudaGraphNode_t graph_node;
    ::cudaHostNodeParams function_params;
    function_params.fn = functor;
    function_params.userData = arg;
    ::cudaError_t err_ = ::cudaGraphAddHostNode(&graph_node, reinterpret_cast<::cudaGraph_t>(graph_ptr),
                                                reinterpret_cast<const ::cudaGraphNode_t *>(deps.data()), deps.size(),
                                                &function_params);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Add host function node to graph failed with message \"%s\".\n",
                                  ::cudaGetErrorString(err_));
    }
    return cuda::GraphNode(reinterpret_cast<std::uintptr_t>(graph_node));
}

// ---------------------------------------------------------------------------------------------------------------------
// GraphNode
// ---------------------------------------------------------------------------------------------------------------------

// Get node type
cuda::NodeType cuda::GraphNode::get_node_type(void) const {
    ::cudaGraphNodeType type;
    ::cudaError_t err_ = ::cudaGraphNodeGetType(reinterpret_cast<::cudaGraphNode_t>(this->node_id), &type);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("CUDA get node type failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    return static_cast<cuda::NodeType>(type);
}

// ---------------------------------------------------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------------------------------------------------

// Destroy current CUDA graph instance
void cuda::Graph::destroy_graph(void) {
    if (this->graph_ != 0) {
        ::cudaError_t err_ = ::cudaGraphDestroy(reinterpret_cast<::cudaGraph_t>(this->graph_));
        if (err_ != 0) {
            Fatal<cuda_runtime_error>("CUDA destroy graph failed with message \"%s\".\n", ::cudaGetErrorString(err_));
        }
        this->graph_ = 0;
    }
}

// Constructor
cuda::Graph::Graph(int flag) {
    ::cudaError_t err_;
    switch (flag) {
        case -1 : {  // default constructor
            break;
        }
        case 0 : {  // construct an empty graph
            ::cudaGraph_t graph_;
            err_ = ::cudaGraphCreate(&graph_, 0);
            if (err_ != 0) {
                Fatal<cuda_runtime_error>("CUDA create graph failed with message \"%s\".\n",
                                          ::cudaGetErrorString(err_));
            }
            this->graph_ = reinterpret_cast<std::uintptr_t>(graph_);
            break;
        }
        default : {  // error unknown argument
            Fatal<std::invalid_argument>("Expected 0 (new empty graph) or -1 (NULL graph), got %d.\n", flag);
            break;
        }
    }
}

// Copy constructor
cuda::Graph::Graph(const cuda::Graph & src) {
    ::cudaGraph_t graph_, graph_src = reinterpret_cast<::cudaGraph_t>(src.graph_);
    ::cudaError_t err_ = ::cudaGraphClone(&graph_, graph_src);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("CUDA clone graph failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    this->graph_ = reinterpret_cast<std::uintptr_t>(graph_);
}

// Copy assignment
cuda::Graph & cuda::Graph::operator=(const cuda::Graph & src) {
    // Destroy current isntance
    this->destroy_graph();
    // Clone graph
    ::cudaGraph_t graph_, graph_src = reinterpret_cast<::cudaGraph_t>(src.graph_);
    ::cudaError_t err_ = ::cudaGraphClone(&graph_, graph_src);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("CUDA clone graph failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    this->graph_ = reinterpret_cast<std::uintptr_t>(graph_);
    return *this;
}

// Move constructor
cuda::Graph::Graph(cuda::Graph && src) {
    this->graph_ = src.graph_;
    src.graph_ = 0;
}

// Move assignment
cuda::Graph & cuda::Graph::operator=(cuda::Graph && src) {
    // Destroy current isntance
    this->destroy_graph();
    // Move graph pointer
    this->graph_ = src.graph_;
    src.graph_ = 0;
    return *this;
}

// Get number of nodes in a graph
std::uint64_t cuda::Graph::get_num_nodes(void) const {
    std::size_t num_nodes;
    ::cudaError_t err_ = ::cudaGraphGetNodes(reinterpret_cast<::cudaGraph_t>(this->graph_), nullptr, &num_nodes);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Get number of nodes of CUDA graph failed with message \"%s\".\n",
                                  ::cudaGetErrorString(err_));
    }
    return std::uint64_t(num_nodes);
}

// Get node list
cuda::GraphNodeList cuda::Graph::get_node_list(void) const {
    std::size_t num_nodes = this->get_num_nodes();
    cuda::GraphNodeList node_list(num_nodes);
    ::cudaError_t err_ = ::cudaGraphGetNodes(reinterpret_cast<::cudaGraph_t>(this->graph_),
                                             reinterpret_cast<::cudaGraphNode_t *>(node_list.data()), &num_nodes);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Get node list of CUDA graph failed with message \"%s\".\n",
                                  ::cudaGetErrorString(err_));
    }
    return node_list;
}

// Get number of edges
std::uint64_t cuda::Graph::get_num_edges(void) const {
    std::size_t num_edges;
    ::cudaError_t err_ = ::cudaGraphGetEdges(reinterpret_cast<::cudaGraph_t>(this->graph_), nullptr, nullptr,
                                             &num_edges);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Get number of edges of CUDA graph failed with message \"%s\".\n",
                                  ::cudaGetErrorString(err_));
    }
    return std::uint64_t(num_edges);
}

// Get edge list
cuda::GraphEdgeList cuda::Graph::get_edge_list(void) const {
    // allocate memory
    std::size_t num_edges = this->get_num_edges();
    vector::DynamicVector<::cudaGraphNode_t> nodes_from(num_edges), nodes_to(num_edges);
    // get edge list
    ::cudaError_t err_ = ::cudaGraphGetEdges(reinterpret_cast<::cudaGraph_t>(this->graph_), nodes_from.data(),
                                             nodes_to.data(), &num_edges);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Get edge list of CUDA graph failed with message \"%s\".\n",
                                  ::cudaGetErrorString(err_));
    }
    // transform result
    cuda::GraphEdgeList edge_list(num_edges);
    for (std::uint64_t i = 0; i < num_edges; i++) {
        edge_list[i][0].node_id = reinterpret_cast<std::uintptr_t>(nodes_from[i]);
        edge_list[i][1].node_id = reinterpret_cast<std::uintptr_t>(nodes_to[i]);
    }
    return edge_list;
}

// Add memory allocation node
std::pair<cuda::GraphNode, void *> cuda::Graph::add_malloc_node(std::uint64_t size, const cuda::GraphNodeList & deps) {
    ::cudaGraphNode_t graph_node;
    ::cudaMemAllocNodeParams node_params;
    std::memset(&node_params, 0, sizeof(::cudaMemAllocNodeParams));
    node_params.bytesize = size;
    node_params.poolProps.allocType = ::cudaMemAllocationTypePinned;
    node_params.poolProps.location.id = cuda::Device::get_current_gpu().id();
    node_params.poolProps.location.type = ::cudaMemLocationTypeDevice;
    ::cudaError_t err_ = ::cudaGraphAddMemAllocNode(&graph_node, reinterpret_cast<::cudaGraph_t>(this->graph_),
                                                    reinterpret_cast<const ::cudaGraphNode_t *>(deps.data()),
                                                    deps.size(), &node_params);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Add memory allocation node to graph failed with message \"%s\".\n",
                                  ::cudaGetErrorString(err_));
    }
    return std::pair<cuda::GraphNode, void *>(cuda::GraphNode(reinterpret_cast<std::uintptr_t>(graph_node)),
                                              node_params.dptr);
}

// Add memcpy node
cuda::GraphNode cuda::Graph::add_memcpy_node(void * dest, const void * src, std::uint64_t size,
                                             cuda::MemcpyKind copy_flag, const cuda::GraphNodeList & deps) {
    ::cudaGraphNode_t graph_node;
    ::cudaError_t err_ = ::cudaGraphAddMemcpyNode1D(&graph_node, reinterpret_cast<::cudaGraph_t>(this->graph_),
                                                    reinterpret_cast<const ::cudaGraphNode_t *>(deps.data()),
                                                    deps.size(), dest, src, size,
                                                    static_cast<::cudaMemcpyKind>(copy_flag));
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Add memcpy node to graph failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    return cuda::GraphNode(reinterpret_cast<std::uintptr_t>(graph_node));
}

// Add CUDA deallocation node
cuda::GraphNode cuda::Graph::add_memfree_node(void * ptr, const cuda::GraphNodeList & deps) {
    ::cudaGraphNode_t graph_node;
    ::cudaError_t err_ = ::cudaGraphAddMemFreeNode(&graph_node, reinterpret_cast<::cudaGraph_t>(this->graph_),
                                                   reinterpret_cast<const ::cudaGraphNode_t *>(deps.data()),
                                                   deps.size(), ptr);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Add memfree node to graph failed with message \"%s\".\n",
                                  ::cudaGetErrorString(err_));
    }
    return cuda::GraphNode(reinterpret_cast<std::uintptr_t>(graph_node));
}

// Add CUDA event record node
cuda::GraphNode cuda::Graph::add_event_record_node(const cuda::Event & event, const cuda::GraphNodeList & deps) {
    ::cudaGraphNode_t graph_node;
    ::cudaError_t err_ = ::cudaGraphAddEventRecordNode(&graph_node, reinterpret_cast<::cudaGraph_t>(this->graph_),
                                                       reinterpret_cast<const ::cudaGraphNode_t *>(deps.data()),
                                                       deps.size(),
                                                       reinterpret_cast<::cudaEvent_t>(event.get_event_ptr()));
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Add event record node to graph failed with message \"%s\".\n",
                                  ::cudaGetErrorString(err_));
    }
    return cuda::GraphNode(reinterpret_cast<std::uintptr_t>(graph_node));
}

// Add CUDA event wait node
cuda::GraphNode cuda::Graph::add_event_wait_node(const cuda::Event & event, const cuda::GraphNodeList & deps) {
    ::cudaGraphNode_t graph_node;
    ::cudaError_t err_ = ::cudaGraphAddEventWaitNode(&graph_node, reinterpret_cast<::cudaGraph_t>(this->graph_),
                                                     reinterpret_cast<const ::cudaGraphNode_t *>(deps.data()),
                                                     deps.size(),
                                                     reinterpret_cast<::cudaEvent_t>(event.get_event_ptr()));
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Add event wait node to graph failed with message \"%s\".\n",
                                  ::cudaGetErrorString(err_));
    }
    return cuda::GraphNode(reinterpret_cast<std::uintptr_t>(graph_node));
}

// Add CUDA child graph node
cuda::GraphNode cuda::Graph::add_child_graph_node(const cuda::Graph & child_graph, const cuda::GraphNodeList & deps) {
    ::cudaGraphNode_t graph_node;
    ::cudaError_t err_ = ::cudaGraphAddChildGraphNode(&graph_node, reinterpret_cast<::cudaGraph_t>(this->graph_),
                                                      reinterpret_cast<const ::cudaGraphNode_t *>(deps.data()),
                                                      deps.size(), reinterpret_cast<::cudaGraph_t>(child_graph.graph_));
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Add child graph node to graph failed with message \"%s\".\n",
                                  ::cudaGetErrorString(err_));
    }
    return cuda::GraphNode(reinterpret_cast<std::uintptr_t>(graph_node));
}

// Export graph into DOT file
void cuda::Graph::export_to_dot(const std::string & filename) {
    ::cudaError_t err_ = ::cudaGraphDebugDotPrint(reinterpret_cast<::cudaGraph_t>(this->graph_), filename.c_str(), 0);
}

// Execute a graph (add detecting errored node)
void cuda::Graph::execute(const cuda::Stream & stream) {
    ::cudaGraphExec_t exec_graph;
    char log_buffer[256];
    std::memset(log_buffer, 0, sizeof(log_buffer));
    ::cudaError_t err_ = ::cudaGraphInstantiate(&exec_graph, reinterpret_cast<::cudaGraph_t>(this->graph_), nullptr,
                                                log_buffer, sizeof(log_buffer));
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Create executable graph failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    if (log_buffer[0]) {  // not a null started string
        Warning("Instantiate graph executable failed with error \"%s\"\n", log_buffer);
    }
    cuda::CtxGuard guard(stream.get_gpu());
    err_ = ::cudaGraphLaunch(exec_graph, reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr()));
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Launch graph failed with error: \"%s\"\n", ::cudaGetErrorString(err_));
    }
}

// Destructor
cuda::Graph::~Graph(void) { this->destroy_graph(); }

}  // namespace merlin
