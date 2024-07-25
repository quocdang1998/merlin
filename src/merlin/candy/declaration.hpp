// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_DECLARATION_HPP_
#define MERLIN_CANDY_DECLARATION_HPP_

namespace merlin::candy {

/** @brief Loss function used for training canonical model.*/
enum class TrainMetric : unsigned int {
    /** @brief Relative square error, skipping all data points that are not normal (``0``, ``inf`` or ``nan``).*/
    RelativeSquare = 0x00,
    /** @brief Absolute square error, skipping all data points that are not finite (``inf`` or ``nan``).*/
    AbsoluteSquare = 0x01
};

class Gradient;
class Launcher;
class Model;
class Optimizer;
class Trainer;
class TrialPolicy;
}  // namespace merlin::candy

#endif  // MERLIN_CANDY_DECLARATION_HPP_
