#include <stdio.h>
#include <iostream>
#include <chrono>

#include "pendulumCudaLib.cuh"

template<class F>
__global__ void exKernel(F f,double *device_a, double *device_b, double time_change){
    int threadId = threadIdx.x + blockIdx.x * blockDim.x ;
    f(device_a, device_b, time_change, threadId);
}

template<typename F>
void PendulumCudaLib::executeArrayFunction(F fn, double* state, double* stateChange, double time_change, int size)
{
    const int numThreads = 4;
    const int blockSize = 4;
    const int numBlocks = numThreads/blockSize;
    double *host_state, *host_stateChange;
    double *device_state, *device_stateChange;
    //int sizeOfArray = 1024*1024;

    cudaMalloc(( void**)& device_state, numThreads * sizeof(double));
    cudaMalloc(( void**)& device_stateChange, numThreads * sizeof(double));


    cudaHostAlloc((void **)&host_state, numThreads * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_stateChange, numThreads * sizeof(double), cudaHostAllocDefault);


    for(int index = 0; index < size; index++)
    {
        host_state[index] = state[index];
        host_stateChange[index] = stateChange[index];
    }

    cudaMemcpy(device_state, host_state, numThreads * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(device_stateChange, host_stateChange, numThreads * sizeof(double), cudaMemcpyHostToDevice);

    /*Kernel call*/

    exKernel<<<1, 4>>>(fn, device_state, device_stateChange, time_change);

    cudaMemcpy(host_state, device_state, numThreads * sizeof(double), cudaMemcpyDeviceToHost);

    for(int index = 0; index < size; index++)
    {
        state[index] = host_state[index];
    }

    cudaFreeHost(host_state);
    cudaFreeHost(host_stateChange);
    cudaFree(device_state);
    cudaFree(device_stateChange);
}


void PendulumCudaLib::updateState(double* state, double* state_change, double time_change)
{
    auto updateState = [] __device__ (double* state, double* state_change, double time_change, int index){
        state[index]= state[index] + state_change[index] * 0.0001;
    };
    PendulumCudaLib::executeArrayFunction(updateState, state, state_change, time_change, 4);
}
