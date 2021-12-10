// Sources:
// https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
// https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace
// http://www.taumuon.co.uk/2016/02/lqr-control-of-inverted-pendulum-in-c-using-eigen.html
// https://github.com/taumuon/robotics/tree/master/underactuated_notes/CartPoleLQR/CartPoleLQR

#include <iostream>
#include <cmath>
#include <memory>

#include "Eigen/Dense"

#include "InvertedPendulumSim.h"
#include "InvertedPendulumLinearDynamics.h"
#include "PendulumGPU/pendulumCudaLib.cuh"


double ControlLQR(const State& state)
{
    auto K = InvertedPendulumLinearDynamics::GetKMatrix();
    auto stateV = Eigen::Vector4d::Map(state.data());
    auto u = (K * stateV).value();
    return u;
}

auto GetTimeStepFunctionCartPoleLinear_Eigen(SystemParameters system_params)
{
    InvertedPendulumLinearDynamics cart_pole_linear(system_params);

    auto A = cart_pole_linear.AMatrix();
    auto B = cart_pole_linear.BMatrix();
    auto C = cart_pole_linear.CMatrix();
    auto D = cart_pole_linear.DMatrix();

    return [=](auto state, auto delta_time, auto u)
    {
        Eigen::Vector4d state_vector(state.data());

        Eigen::Matrix<double, 4, 1> state_dot = (A * state_vector) - (B * u);

        auto new_state = state_vector + (state_dot * delta_time);
        return State{ new_state[0], new_state[1], new_state[2], new_state[3]};
    };
}

auto GetTimeStepFunctionCartPoleLinear_Cuda(SystemParameters system_params)
{
    InvertedPendulumLinearDynamics cart_pole_linear(system_params);

    auto A = cart_pole_linear.AMatrix();
    auto B = cart_pole_linear.BMatrix();
    auto C = cart_pole_linear.CMatrix();
    auto D = cart_pole_linear.DMatrix();

    return [=](auto state, auto delta_time, auto u)
    {
        // State vector, 4 elements
        Eigen::Vector4d state_vector(state.data());
        // Change in State = A * State - B * control inputs
        Eigen::Matrix<double, 4, 1> state_dot = (A * state_vector) - (B * u);

        // New State = original state + change in state * time change

        double new_state[4];
        double state_change[4];
        for(auto i = 0; i < 4; i++){
            new_state[i] = state_vector(i);
            state_change[i] = state_dot(i);
        }
        //auto new_state = state_vector + (state_dot * delta_time);
        PendulumCudaLib::updateState(new_state, state_change, delta_time);

        return State{ new_state[0], new_state[1], new_state[2], new_state[3]};
    };
}

template <typename FunctorUpdateState, typename FunctorControlInput>
std::vector<TimeAndState> SimulateTrajectory(const State& initial_state,
                                              int N,
                                              double delta_time,
                                             FunctorUpdateState update_state,
                                             FunctorControlInput control_input)
{
    std::vector<TimeAndState> result;

    result.push_back(TimeAndState(0.0, initial_state, 0.0));
    
    auto state = initial_state;
    for (int i = 1; i < N; ++i)
    {
        auto u = control_input(state);
        state = update_state(state, delta_time, u);

        // TODO: boost.odeint
        auto time = i * delta_time;

        result.push_back(TimeAndState(time, state, u));
    }

    return result;
}

std::vector<TrajectoryTuple> executeSimulation()
{
    SystemParameters system_params = SystemParameters("parameters.xml");

    auto initial_theta = system_params.params.pendulum.initialAngle;
    auto initial_state = State{ 0.0, 0.0, initial_theta, 0.0 };

    //auto time_stepper = GetTimeStepFunctionCartPoleLinear_Eigen(system_params);
    auto time_stepper = GetTimeStepFunctionCartPoleLinear_Cuda(system_params);
    auto trajectory = SimulateTrajectory(initial_state, 50000, 0.0001, time_stepper, ControlLQR);

    auto count = 0;
    std::cout << "# t x theta u" << std::endl;
    std::vector<TrajectoryTuple> ret;
    for (auto item : trajectory)
    {
        if (count++ % 100 != 0) { continue; }
        // if (count++ > 1000) { break; }

        auto time = std::get<0>(item);
        auto x = std::get<1>(item)[0];
        auto theta = std::get<1>(item)[2];
        auto u = std::get<2>(item);
        std::cout << time << " " << x << " " << theta << " " << u << std::endl;
        ret.push_back(TrajectoryTuple(time, x, theta));
    }
    // Testing runtime K calculation, currently not working
    //InvertedPendulumLinearDynamics cart_pole_linear(system_params);
    //std::cout << cart_pole_linear.GetKMatrixCuda() << std::endl;

    // std::cin.get();
    return ret;
}
