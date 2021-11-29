#ifndef INVERTEDPENDULUMSIM_H
#define INVERTEDPENDULUMSIM_H

#include <array>
#include <vector>

using State = std::array<double, 4>;
using TimeAndState = std::tuple<double, State, double>;
using TrajectoryTuple = std::tuple<double, double, double>;

std::vector<TrajectoryTuple> executeSimulation();

#endif // INVERTEDPENDULUMSIM_H
