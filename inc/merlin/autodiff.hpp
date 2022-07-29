// Copyright 2022 quocdang1998
#ifndef MERLIN_AUTODIFF_HPP_
#define MERLIN_AUTODIFF_HPP_

#include <cstdio>
#include <vector>

#include "merlin/array.hpp"

namespace merlin {

/** @brief Abstract class representing a node in the dataflow graph.*/
class AutoNode {
public:
    /** @brief Default destructor.*/
    virtual ~AutoNode(void);

    /** @brief Reference to list of AutoNode-s connected to the current instance */
    std::vector<AutoNode*> & connected_nodes(void) {
        return this->connected_nodes_;
    }
    /** @brief Constant reference to list of AutoNode-s connected to the current instance */
    const std::vector<AutoNode*>& connected_nodes (void) const {
        return this->connected_nodes_;
    }

    /** @brief Value of the current node.
    
    The value hold by the object. Return in form of an array.
    
    Since this class is the base class for all other types of nodes, the value is the
    array \f$[0.0]\f$
    */
    virtual Array value(void) {
        return Array(0.0);
    }
    /*
    virtual Array gradient(void) {
        return Array(0.0);
    }
    */

protected:
    /** @brief List of pointers to nodes connected to the object.*/
    std::vector<AutoNode*> connected_nodes_;
};

class Constant : public AutoNode {
public:
    Constant(float value);
    Constant(const Array& value) : data_(value) {}

    virtual Array value(void) {
        return this->data_;
    }

protected:
    Array data_;
};
/*
class Variable : public Constant {
 public:
    
};
*/
}  // namespace merlin

#endif  // MERLIN_AUTODIFF_HPP_
