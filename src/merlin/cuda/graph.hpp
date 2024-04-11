// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_GRAPH_HPP_
#define MERLIN_CUDA_GRAPH_HPP_

#include <array>    // std::array
#include <cstdint>  // std::uintptr_t
#include <string>   // std::string
#include <tuple>    // std::tuple

#include "merlin/cuda/declaration.hpp"   // merlin::cuda::Graph
#include "merlin/cuda/enum_wrapper.hpp"  // merlin::cuda::MemcpyKind, merlin::cuda::NodeType
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/vector.hpp"             // merlin::Vector

namespace merlin {

namespace cuda {

/** @brief CUDA graph node.*/
struct GraphNode {
    /** @brief Default constructor.*/
    GraphNode(void) = default;
    /** @brief Constructor from pointer.*/
    GraphNode(std::uintptr_t ptr) : graphnode_ptr(ptr) {}

    /** @brief Get node type.*/
    cuda::NodeType get_node_type(void) const;

    /** @brief Destructor.*/
    ~GraphNode(void) = default;

    /** @brief Pointer to CUDA graph node object.*/
    std::uintptr_t graphnode_ptr = 0;
};

static inline Vector<cuda::GraphNode> begin_graphnode = Vector<cuda::GraphNode>();

#ifdef __NVCC__
typedef cudaHostFn_t CudaHostFunction;
#else
typedef void (*CudaHostFunction)(void *);
#endif  // __NVCC__

}  // namespace cuda

/** @brief CUDA Graph.*/
class cuda::Graph {
  public:
    /// @name Constructors
    /// @{
    /** @brief Constructor a graph.
     *  @param flag Creation flag. ``-1`` means a default (empty) constructor, ``0`` means creating a new CUDA graph.
     */
    MERLIN_EXPORTS Graph(int flag = -1);
    /** @brief Constructor from pointer.*/
    Graph(std::uintptr_t graph_ptr) : graph_ptr_(graph_ptr) {}
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    MERLIN_EXPORTS Graph(const cuda::Graph & src);
    /** @brief Copy assignment.*/
    MERLIN_EXPORTS cuda::Graph & operator=(const cuda::Graph & src);
    /** @brief Move constructor.*/
    MERLIN_EXPORTS Graph(cuda::Graph && src);
    /** @brief Move assignment.*/
    MERLIN_EXPORTS cuda::Graph & operator=(cuda::Graph && src);
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get pointer of CUDA graph object.*/
    std::uintptr_t get_graph_ptr(void) const { return this->graph_ptr_; }
    /** @brief Get number of nodes in a graph.*/
    MERLIN_EXPORTS std::uint64_t get_num_nodes(void) const;
    /** @brief Get node list.*/
    MERLIN_EXPORTS Vector<cuda::GraphNode> get_node_list(void) const;
    /** @brief Get number of edges.*/
    MERLIN_EXPORTS std::uint64_t get_num_edges(void) const;
    /** @brief Get edge list.
     *  @return A vector of pairs of graph nodes. The first node is origin of the edge, the second is destination.
     */
    MERLIN_EXPORTS Vector<std::array<cuda::GraphNode, 2>> get_edge_list(void) const;
    /// @}

    /// @name Add nodes to graph
    /// @{
    /** @brief Add memory allocation node.
     *  @param size Size (in bytes) to allocate.
     *  @param deps %Vector of nodes on which the node depends.
     *  @return Tuple of added graph node and pointer to allocated data.
     */
    MERLIN_EXPORTS std::tuple<cuda::GraphNode, void *> add_mem_alloc_node(std::uint64_t size,
                                                                          const Vector<cuda::GraphNode> & deps);
    /** @brief Add memory copy node.
     *  @param dest Pointer to destination array.
     *  @param src Pointer to source destination.
     *  @param size Number bytes to copy.
     *  @param copy_flag Copy flag.
     *  @param deps %Vector of nodes on which the node depends.
     */
    MERLIN_EXPORTS cuda::GraphNode add_memcpy_node(void * dest, const void * src, std::uint64_t size,
                                                   cuda::MemcpyKind copy_flag, const Vector<cuda::GraphNode> & deps);
    /** @brief Add CUDA kernel node.
     *  @param kernel Pointer to function (functor) on GPU to be executed.
     *  @param n_blocks Number of blocks in the grid.
     *  @param n_threads Number of threads per block.
     *  @param shared_mem Size in bytes of shared memory.
     *  @param deps %Vector of nodes on which the node depends.
     *  @param args List of arguments to provide to the kernel.
     */
    template <typename Function, typename... Args>
    cuda::GraphNode add_kernel_node(Function * kernel, std::uint64_t n_blocks, std::uint64_t n_threads,
                                    std::uint64_t shared_mem, const Vector<cuda::GraphNode> & deps, Args &&... args);
    /** @brief Add CUDA host node.
     *  @param functor Pointer to CPU function take in a pointer to ``void`` argument.
     *  @param deps %Vector of nodes on which the node depends.
     *  @param arg Pointer to argument to pass to the function.
     */
    MERLIN_EXPORTS cuda::GraphNode add_host_node(cuda::CudaHostFunction functor, const Vector<cuda::GraphNode> & deps,
                                                 void * arg = nullptr);
    /** @brief Add CUDA deallocation node.
     *  @param ptr GPU pointer to be freed.
     *  @param deps %Vector of nodes on which the node depends.
     */
    MERLIN_EXPORTS cuda::GraphNode add_memfree_node(void * ptr, const Vector<cuda::GraphNode> & deps);
    /** @brief Add CUDA event record node.
     *  @param event CUDA event to be recorded.
     *  @param deps %Vector of nodes on which the node depends.
     */
    MERLIN_EXPORTS cuda::GraphNode add_event_record_node(const cuda::Event & event,
                                                         const Vector<cuda::GraphNode> & deps);
    /** @brief Add CUDA event wait node.
     *  @param event CUDA event to be synchronized.
     *  @param deps %Vector of nodes on which the node depends.
     */
    MERLIN_EXPORTS cuda::GraphNode add_event_wait_node(const cuda::Event & event, const Vector<cuda::GraphNode> & deps);
    /** @brief Add CUDA child graph node.
     *  @param child_graph CUDA graph as child graph to be added.
     *  @param deps %Vector of nodes on which the node depends.
     */
    MERLIN_EXPORTS cuda::GraphNode add_child_graph_node(const cuda::Graph & child_graph,
                                                        const Vector<cuda::GraphNode> & deps);
    /// @}

    /// @name Operation on CUDA Graph
    /// @{
    /** @brief Execute a graph on a CUDA stream.
     *  @warning This function will lock the mutex.
     *  @param stream %Stream on which the graph is launched.
     *  @note %Stream capture must be finished.
     */
    MERLIN_EXPORTS void execute(const cuda::Stream & stream);
    /** @brief Export graph into DOT file.*/
    MERLIN_EXPORTS void export_to_dot(const std::string & filename);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Graph(void);
    /// @}

  protected:
    /** @brief Pointer to CUDA graph object.*/
    std::uintptr_t graph_ptr_ = 0;

  private:
    /** @brief Destroy current CUDA graph instance.*/
    void destroy_graph(void);
};

}  // namespace merlin

#include "merlin/cuda/graph.tpp"

#endif  // MERLIN_CUDA_GRAPH_HPP_
