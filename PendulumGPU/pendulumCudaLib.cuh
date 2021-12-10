#ifndef PENDULUMCUDALIB_CUH
#define PENDULUMCUDALIB_CUH

namespace PendulumCudaLib{
    /*
        __device__ void addArraysElementWise(int *device_a, int *device_b, int *device_result, int ind){
            //int threadId = threadIdx.x + blockIdx.x * blockDim.x ;
            device_result[ind]= device_a[ind]+device_b[ind];
        }
*/


    template<typename F>
    void executeArrayFunction(F fn, double* state, double* stateChange, double timeChange, int size);

    void updateState(double* state, double* state_change, double time_change);


}
#endif
