#ifndef MERLIN_ARRAY_HPP
#define MERLIN_ARRAY_HPP

#include <vector>

namespace merlin {

/*! \brief Multi-dimensional array.

    This class make the interface between C array and Numpy array.*/
class Array {

    /*! \brief Multi-dimensional array iterator.*/
    class iterator {
      public:
        /*! \brief Constructor from a vector of index.*/
        iterator(const std::vector<unsigned int> & it) : it_(it) {};

        /*! \brief Increment operator.
            
            Increase the index of the last dimension by 1. If the maximum is
            reached, set the index to zero and increment the next dimension.
            
            Example: \f$(0, 0) \rightarrow (0, 1) \rightarrow (0, 2) \rightarrow
            (1, 0) \rightarrow (1, 1) \rightarrow (1, 2)\f$.*/
        void inc(const std::vector<unsigned int> & dims);

        /*! \brief Index vector.*/
        std::vector<unsigned int> it_;
    };

  public:
    /*! \brief Create array from NumPy array.*/
    Array(double * data, unsigned int ndim, unsigned int * dims, unsigned int * strides, bool copy = true);
    /*! \brief Destructor.*/
    ~Array(void);

    /*! \brief Indicate array is copied or assigned to another array.
    
    If the array is copy version, delete operator is called when the array is
    destroyed to avoid memory leak.*/
    bool is_copy;
    /*! \brief Size of the array.
    
    Product of the size of each dimension.*/
    unsigned int size();
    /*! \brief Begin iterator.
    
    Vector of index \f$(0, 0, ..., 0)\f$.*/
    Array::iterator begin(void);
    /*! \brief End iterator.
    
    Vector of index \f$(d_0 - 1, d_1 - 1, ..., d_{n-1} - 1)\f$.*/
    Array::iterator end(void);
    friend bool operator< (const iterator & left, const iterator & right);
    double & operator[] (const std::vector<unsigned int> & index);

  private:
    double * data_;
    unsigned int ndim_;
    std::vector<unsigned int> dims_;
    std::vector<unsigned int> strides_;

};

}  // namespace merlin

#endif