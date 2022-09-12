// Copyright 2022 quocdang1998
#ifndef MERLIN_PARCEL_HPP_
#define MERLIN_PARCEL_HPP_

#include <fstream>
#include <string>

#include "merlin/nddata.hpp"  // merlin::NdData

namespace merlin {

class Stock : public NdData {
  public:
    /** @brief Default constructor.*/
    Stock(void) = default;
    /** @brief Constructor from a filename.*/
    explicit Stock(const std::string & filename);

    Array to_array(void);
    void dumped(const Array & src);

    /** @brief Destructor.*/
    ~Stock(void);

  protected:
    std::fstream file_stream_;

};

}  // namespace merlin

#endif  // MERLIN_PARCEL_HPP_
