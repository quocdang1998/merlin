#ifndef MERLIN_ARRAY_HPP
#define MERLIN_ARRAY_HPP

#include <vector>

namespace merlin {

class Array {

    class iterator {
      public:
        iterator(const std::vector<unsigned int> & it) : it_(it) {};

        void inc(const std::vector<unsigned int> & dims);

        std::vector<unsigned int> it_;
    };

  public:
    /*! \brief Create array from NumPy array.*/
    Array(double * data, unsigned int ndim, unsigned int * dims, unsigned int * strides, bool copy = true);
    ~Array(void);

    bool is_copy;
    unsigned int size();
    Array::iterator begin(void);
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