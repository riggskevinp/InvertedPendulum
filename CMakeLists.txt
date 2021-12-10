cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(InvertedPendulumCartSimulation LANGUAGES C CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
#add_compile_options(-std=c++11)

include_directories(eigen eigen/Eigen)
include_directories(pugixml/src)


find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)



add_executable(simulation
main.cpp
mainwindow.cpp
mainwindow.h
mainwindow.ui
robot.cpp
robot.h
InvertedPendulumSim.cpp
InvertedPendulumSim.h
SystemParameters.h
SystemParameters.cpp
InvertedPendulumLinearDynamics.h
InvertedPendulumLinearDynamics.cpp
pugixml/src/pugixml.hpp
pugixml/src/pugixml.cpp
)

set_target_properties(simulation
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(PendulumGPU)

target_link_libraries(simulation
    PRIVATE
        Qt${QT_VERSION_MAJOR}::Widgets
    PUBLIC
        PendulumCuda)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/parameters.xml ${CMAKE_CURRENT_BINARY_DIR}/parameters.xml COPYONLY)


#target_link_libraries(simulation
#        InvertedPendulumLinearDynamics
#        )

#Comment out to disable testing
enable_testing()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
