# Inverted Pendulum Simulation

## Background

It was strangely difficult for me to come up with a gpgpu/hetergenous computing application I was interested in.
I looked at raytracing, path planning and some computational geometry algorithms, among others, but didn't find something I could cleanly implement.
After looking back at some introduction to robotics coursework, I decided to look into writing accelerated versions of some of the programs.
The main opportunity that presented itself was the dynamic Linear Quadratic Regulator control strategy as it has a lot of matrix operations.
There were other opportunities such as the extended Kalman Filter used for localization.
The nice part of that application is it can build up very large matrices depending on how many landmarks you are using.

I chose the LQR.

First I looked for existing solutions and found a good jumping off point through a [blog post from Gary Evans](http://www.taumuon.co.uk/2016/02/lqr-control-of-inverted-pendulum-in-c-using-eigen.html)
Gary implemented an LQR for an inverted pendulum system using the content from an MIT MOOC, [System Modeling](https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling)
[State-Space Methods for Controller Design](https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace#6),which appears to also have contributions from Mathworks, UMich and Carnegie Mellon.

## Goals
- [x] Use cmake to enable easier crossplatform developement
- [x] Add unit test framework to enable faster iterations
- [x] Remove hard coding of system parameters to enable users to experiment
    - [x] Refactor to use xml configs
    - [x] Calculate dependent variables dynamically
    - [] Allow modification of playback speed, currently set at compile time
- [x] Add a gui/visualization so that users can see the performance of their system
- [x] Use Eigen for explicit vectorization to make matrix operations fast
- [] Calculate control K dynamically
- [] Calculate linearized system dynamically
- [] Look to incorporate cuda/opencl to calculate K and do other matrix operations
- [] Switch from LQR to dLQR
- [] If there are multiple implementations of the same calculations, provide timing analysis
- [] Possibly incorporate WASM and WASM hardware acceleration
- [] Overall, clean everything up.  It's pretty sloppy right now.

## Current System

### Visualization

- Utilizes Qt
    - Specifically QGraphicsView and QGraphicsScene
    - The main point of interest is the mechanism for advancing the scene.
    QTimer has a signal, timeout, that is connected to the slot, advance, of the QGraphicsObjects.
    Setting the timeout duration, in milliseconds, is how we can control the playback speed.
    Currently, the simulation creates a state for ever millisecond.

Static capture of current visualization:

    ![Initial ](PoCScreenCapture.PNG)

### Simulation

- Broken into the main InvertedPendulumSim and InvertedPendulumLinearDynamics.
- SystemParameters currently holds a minor portion of the simulation logic as it calculates the moment of inertia about the center of the pole(pendulum.)

### IO

- Currently only the simulation uses parameters from the xml
- Need to pull SystemParameters up a level and out of Simulation

###