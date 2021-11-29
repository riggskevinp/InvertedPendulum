cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)
include_directories(..)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/parametersTest.xml ${CMAKE_CURRENT_BINARY_DIR}/parametersTest.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/parametersTestMissingCartMass.xml ${CMAKE_CURRENT_BINARY_DIR}/parametersTestMissingCartMass.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/parametersTestMissingFriction.xml ${CMAKE_CURRENT_BINARY_DIR}/parametersTestMissingFriction.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/parametersTestFrictionEmpty.xml ${CMAKE_CURRENT_BINARY_DIR}/parametersTestFrictionEmpty.xml COPYONLY)

add_executable(SimTest
        InvertedPendulumSimTest.cpp
        ../SystemParameters.h
        ../SystemParameters.cpp
        ../InvertedPendulumLinearDynamics.h
        ../InvertedPendulumLinearDynamics.cpp
        ../pugixml/src/pugixml.hpp
        ../pugixml/src/pugixml.cpp
        )

gtest_discover_tests(SimTest)


target_link_libraries(SimTest
        gtest
        gtest_main
)

