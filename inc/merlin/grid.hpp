#ifndef MERLIN_GRID_HPP
#define MERLIN_GRID_HPP

#include <vector>

namespace merlin {

class GridND {
  public:
    /*! \brief Construct an empty grid from a given number of n-dim points.*/
    GridND (unsigned int ndim, unsigned int npoint);
    /*! \brief Deep copy constructor.*/
    GridND (const GridND & gridnd);
    /*! \brief Deep copy assignment.*/
    GridND & operator= (const GridND & gridnd);
    /*! \brief Move constructor.*/
    GridND (GridND && gridnd);
    /*! \brief Move assignment.*/
    GridND & operator= (GridND && gridnd);
    /*! \brief Default destructor.*/
    virtual ~GridND(void);

  protected:
    /*! \brief Number of dimension of the grid.*/
    unsigned int ndim_;
    /*! \brief Number of points in the grid.*/
    unsigned int npoint_;
    /*! \brief Pointer to a 2D C-contiguous array of size (npoint, ndim).
    
    This 2D table store the value of each n-dimensional point as a row vector.*/.
    double * data_;
};

}  // namespace merlin

#endif