#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

#pragma GCC diagnostic warning "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>

//Link HIP
#include "hip/hip_runtime.h"

//Links for dev
#include <thread>
#include <vector>
#include <array>
#include <typeinfo>
#include <iostream>
#include <mutex>
#include <sched.h>
#include <pthread.h>
#include <algorithm> 
#include <string>
#include <utility>
#include <functional>
#include <future>
#include <cassert>
#include <chrono>
#include <type_traits>
#include <list>
#include <ranges>

//Links Specx
#include "SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"
#include "Utils/SpTimer.hpp"
#include "Utils/small_vector.hpp"
#include "Utils/SpConsumerThread.hpp"
#include "SpComputeEngine.hpp"
#include "Speculation/SpSpeculativeModel.hpp"

#include "Tools.hpp"

#define BLOCK_SIZE 128


void __device__ vector_add_device(const double vecA, const double vecB, double &vecC)
{
    vecC = vecA + vecB;
}

void __global__ vector_add(const double *vecA, const double *vecB, double *vecC, const int nb)
{
    const int i = hipBlockDim_x*hipBlockIdx_x+hipThreadIdx_x;

    if (i < nb)
        vector_add_device(vecA[i], vecB[i], vecC[i]); 
}


void __device__ vector_scalar_device(double coeff,double &vecA)
{
    vecA = coeff*vecA;
}

void __global__ vector_scalar(double coeff,double *vecA, const int nb)
{
    const int i = hipBlockDim_x*hipBlockIdx_x+hipThreadIdx_x;
    if (i < nb)
        vector_scalar_device(coeff,vecA[i]); 
}


void check_solution(double coeff,double* a_in,double* b_in,double* c_in,int nb)
{
  printf("[INFO] :");
	int errors = 0;
  	for (int i = 0; i < nb; i++) {
	    if (c_in[i] != coeff*(a_in[i] + b_in[i])) { errors++; }
	}
  	if (errors!=0) { printf("FAILED: %d errors\n",errors); } else { printf ("WELL DONE PASSED! :-)\n"); }
}

void write_vector(std::string ch,double* v,int nb)
{
  std::cout<<"[INFO] :"<<ch<<"> ";
	for (int i = 0; i < nb; i++) { std::cout<<int(v[i]); }
  std::cout<<std::endl;
}



void Test001()
{
    int nbThreads=3;
    int nbElements=10;
    double c0=2.0; 
    double c1=1.5; 
    int size_array = sizeof(double) * nbElements;

    double *h_vecA = (double *)malloc(size_array);
    double *h_vecB = (double *)malloc(size_array);
    double *h_vecC = (double *)malloc(size_array);

    for (int i = 0; i < nbElements; i++)
    {
        h_vecA[i] = 1;
        h_vecB[i] = 2;
        h_vecC[i] = 0;
    }

    write_vector("Vector A",h_vecA,nbElements);
    write_vector("Vector B",h_vecB,nbElements);

    auto FC1=[size_array,nbElements](const double* vh_vecA,const double* vh_vecB,double* vh_vecC0) {  
          double *d_vecA,*d_vecB,*d_vecC;
          hipMalloc((void **)&d_vecA, size_array);
          hipMalloc((void **)&d_vecB, size_array);
          hipMalloc((void **)&d_vecC, size_array);
          hipMemcpy(d_vecA, vh_vecA, size_array, hipMemcpyHostToDevice);
          hipMemcpy(d_vecB, vh_vecB, size_array, hipMemcpyHostToDevice);
          int grid_size = (nbElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
          hipLaunchKernelGGL(vector_add, grid_size,BLOCK_SIZE, 0, 0, d_vecA, d_vecB, d_vecC, nbElements);
          hipMemcpy(vh_vecC0, d_vecC, size_array, hipMemcpyDeviceToHost);
          hipFree(d_vecA); hipFree(d_vecB); hipFree(d_vecC);
        return true;
    };

    auto FC2=[size_array,nbElements](const double& vcoeff,double* vh_vecC0) {  
          double *d_vecC;
          hipMalloc((void **)&d_vecC, size_array);
          hipMemcpy(d_vecC, vh_vecC0, size_array, hipMemcpyHostToDevice);
          int grid_size = (nbElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
          hipLaunchKernelGGL(vector_scalar, grid_size,BLOCK_SIZE, 0, 0,vcoeff,d_vecC, nbElements);
          hipMemcpy(vh_vecC0, d_vecC, size_array, hipMemcpyDeviceToHost);
          hipFree(d_vecC);
        return true;
    };


    SpRuntime runtime(nbThreads);
      runtime.task(SpRead(h_vecA),SpRead(h_vecB),SpWrite(h_vecC),FC1);
      runtime.task(SpRead(c0),SpWrite(h_vecC),FC2);
      runtime.task(SpRead(c1),SpWrite(h_vecC),FC2);
      //...//
    runtime.waitAllTasks();
    runtime.stopAllThreads();

    runtime.generateDot("Test_hip_AMD.dot", true);
    runtime.generateTrace("Test_hip_AMD.svg");  

    write_vector("Vector C = "+std::to_string(c0*c1)+" * (A + B)",h_vecC,nbElements);


    check_solution(c0*c1,h_vecA,h_vecB,h_vecC,nbElements);

    free(h_vecA); free(h_vecB); free(h_vecC);
}


int main() {
  Test001();
  return 0;
}


