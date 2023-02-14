// Copyright 2022 quocdang1998
#ifndef MERLIN_ENV_HPP_
#define MERLIN_ENV_HPP_

#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin {

/** @brief Execution environment of merlin.*/
class MERLIN_EXPORTS Environment {
  public:
    /** @brief Default constructor.*/
    Environment(void);

    void set_inited(bool value);
    void print_inited(void);

    /** @brief Check if the environment is initialized or not.*/
    static bool is_inited;
};

}  // namespace merlin

#endif  // MERLIN_ENV_HPP_
