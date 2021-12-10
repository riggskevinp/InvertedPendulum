#ifndef INVERTEDPENDULUMCARTSIMULATION_INVERTEDPENDULUMLINEARDYNAMICS_H
#define INVERTEDPENDULUMCARTSIMULATION_INVERTEDPENDULUMLINEARDYNAMICS_H

#include "Eigen/Dense"
#include "SystemParameters.h"


class InvertedPendulumLinearDynamics
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    InvertedPendulumLinearDynamics(const SystemParameters& system_params);

    Eigen::Matrix4d AMatrix() { return _a; }
    Eigen::Matrix<double, 4, 1> BMatrix() { return _b; }
    Eigen::Matrix<double, 2, 4> CMatrix() { return _c; }
    Eigen::Matrix<double, 2, 1> DMatrix() { return _d; }
    static const Eigen::Matrix<double, 1, 4>& GetKMatrix();
    Eigen::Matrix<double, 1, 4> GetKMatrixCuda();

private:
    Eigen::Matrix4d _a;
    Eigen::Matrix<double, 4, 1> _b;
    Eigen::Matrix<double, 2, 4> _c;
    Eigen::Matrix<double, 2, 1> _d;
    Eigen::Matrix<double, 4, 4> _q {
        {5000, 0,   0, 0},
        {   0, 0,   0, 0},
        {   0, 0, 100, 0},
        {   0, 0,   0, 0},
    };

};




#endif //INVERTEDPENDULUMCARTSIMULATION_INVERTEDPENDULUMLINEARDYNAMICS_H
