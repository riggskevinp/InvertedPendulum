cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(PendulumCuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
#add_compile_options(-std=c++11)



add_library(PendulumCuda
        pendulumCudaLib.cuh
        pendulumCudaLib.cu
        )


#Comment out to disable testing
#enable_testing()

#if(BUILD_TESTING)
#    add_subdirectory(tests)
#endif()
