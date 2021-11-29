#include "InvertedPendulumLinearDynamics.h"

InvertedPendulumLinearDynamics::InvertedPendulumLinearDynamics(const SystemParameters& system_params)
{
    auto g = system_params.params.gravity;
    auto m_c = system_params.params.cart.mass;
    auto m_p = system_params.params.pendulum.mass;
    auto length = system_params.params.pendulum.length;
    auto I = system_params.params.pendulum.momentOfInertiaAboutCenter;

    auto b = system_params.params.friction;

    auto lengthSquared = pow(length, 2.0);
    auto p = (I*(m_c + m_p)) + (m_c*m_p*(lengthSquared));

    _a.setZero();
    _a(0, 1) = 1;
    _a(1, 1) = -((I + (m_p*lengthSquared))*b) / p;
    _a(1, 2) = (pow(m_p, 2.0))*g*(lengthSquared) / p;
    _a(2, 3) = 1;
    _a(3, 1) = -(m_p*length*b) / p;
    _a(3, 2) = m_p*g*length*(m_c + m_p) / p;

    _b.setZero();
    _b(1, 0) = ((I + (m_p*lengthSquared))) / p;
    _b(3, 0) = (m_p * length) / p;

    _c.setZero();
    _c(0, 0) = 1.0;
    _c(1, 2) = 1.0;

    _d.setZero();
}

// stackoverflow.com/questions/25999407/initialize-a-constant-eigen-matrix-in-a-header-file
const Eigen::Matrix<double, 1, 4>& InvertedPendulumLinearDynamics::GetKMatrix()
{
    static const struct Once
            {
        Eigen::Matrix<double, 1, 4> K;

        Once()
        {
            // This is for friction = 0.1
            // K << -70.711, -37.834, 105.530, 20.924;

            // friction 0.0
            K << -70.711, -37.734, 105.528, 20.923;
        }
            } once;
    return once.K;
}
