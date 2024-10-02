// Copyright 2022 quocdang1998
#ifndef MERLIN_HPP_
#define MERLIN_HPP_

// Low level API
// -------------

#include "merlin/assume.hpp"
#include "merlin/avx.hpp"
#include "merlin/config.hpp"
#include "merlin/exports.hpp"
#include "merlin/filelock.hpp"
#include "merlin/memory.hpp"
#include "merlin/thread_divider.hpp"

// Environment
// -----------
#include "merlin/color.hpp"
#include "merlin/env.hpp"
#include "merlin/logger.hpp"

// Vector API
// ----------
#include "merlin/vector.hpp"
#include "merlin/utils.hpp"
#include "merlin/permutation.hpp"

// Array API
// ---------

#include "merlin/array/nddata.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/array/stock.hpp"

// Grid API
// --------

#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/grid/regular_grid.hpp"

// CUDA API
// --------

#include "merlin/cuda/device.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/cuda/event.hpp"
#include "merlin/cuda/event.hpp"
#include "merlin/cuda/graph.hpp"
#include "merlin/cuda/copy_helpers.hpp"

// Canonical Polyadic Decomposition
// --------------------------------

#include "merlin/candy/model.hpp"
#include "merlin/candy/gradient.hpp"
#include "merlin/candy/loss.hpp"
#include "merlin/candy/optimizer.hpp"
#include "merlin/candy/randomizer.hpp"
#include "merlin/candy/trial_policy.hpp"
#include "merlin/candy/trainer.hpp"

#endif  // MERLIN_HPP_
