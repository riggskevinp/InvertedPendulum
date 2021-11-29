//
// Created by kr63454 on 11/28/2021.
//

#ifndef INVERTEDPENDULUMCARTSIMULATION_SYSTEMPARAMETERS_H
#define INVERTEDPENDULUMCARTSIMULATION_SYSTEMPARAMETERS_H

#include <string>

struct Cart{
public:
    double mass;
};

struct Pendulum{
public:
    double mass;
    double length;
    double initialAngle;
    double momentOfInertiaAboutCenter;
};


struct InvertedPendulumSystemParameters{
public:
    Cart cart;
    Pendulum pendulum;
    double friction;
    double gravity;
};

class SystemParameters{
public:
    SystemParameters(const std::string inputXmlFile);
    InvertedPendulumSystemParameters params;

};


#endif //INVERTEDPENDULUMCARTSIMULATION_SYSTEMPARAMETERS_H
