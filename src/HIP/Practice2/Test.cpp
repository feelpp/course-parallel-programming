#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"






#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif


//#define WIDTH     1024
//#define HEIGHT    1024

#define WIDTH     16
#define HEIGHT    16


#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1



void __device__ vector_add_device(const double vecA, const double vecB, double &vecC)
{
    vecC = vecA + vecB;
}

void __global__ vector_add(const double *vecA, const double *vecB, double *vecC, const int nb)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nb)
        vector_add_device(vecA[i], vecB[i], vecC[i]); 
}


__global__ void vector_add(double* __restrict__ a, const double* __restrict__ b, const double* __restrict__ c, int width, int height) 
  {
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
      int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
      int i = y * width + x;
      if ( i < (width * height)) {
        a[i] = b[i] + c[i];
      }
  }

#if 0
__kernel__ void vector_add(double* a, const double* b, const double* c, int width, int height) {
 
  int x = blockDimX * blockIdx.x + threadIdx.x;
  int y = blockDimY * blockIdy.y + threadIdx.y;

  int i = y * width + x;
  if ( i < (width * height)) {
    a[i] = b[i] + c[i];
  }
}
#endif

void check_solution(double* a_in,double* b_in,double* c_in,int nb)
{
	int errors = 0;
  	for (int i = 0; i < nb; i++) {
	    if (a_in[i] != (b_in[i] + c_in[i])) { errors++; }
	}
  	if (errors!=0) { printf("FAILED: %d errors\n",errors); } else { printf ("PASSED!\n"); }
}

void write_vector(char *ch,double* v,int nb)
{
	printf("%s\n",ch);
	for (int i = 0; i < nb; i++) { printf("%i",int(v[i])); }
	printf("\n");
}


int main() {
  
  double* hostCpu_A;
  double* hostCpu_B;
  double* hostCpu_C;

  double* deviceGPU_A;
  double* deviceGPU_B;
  double* deviceGPU_C;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  std::cout << " System minor " << devProp.minor << std::endl;
  std::cout << " System major " << devProp.major << std::endl;
  std::cout << " agent prop name " << devProp.name << std::endl;
 
  hostCpu_A = (double*)malloc(NUM * sizeof(double));
  hostCpu_B = (double*)malloc(NUM * sizeof(double));
  hostCpu_C = (double*)malloc(NUM * sizeof(double));
  
  for (int i = 0; i < NUM; i++) {
	  hostCpu_A[i] = 0;
	  hostCpu_B[i] = 1;
    hostCpu_C[i] = 2;
  }
  
  HIP_ASSERT(hipMalloc((void**)&deviceGPU_A, NUM * sizeof(double)));
  HIP_ASSERT(hipMalloc((void**)&deviceGPU_B, NUM * sizeof(double)));
  HIP_ASSERT(hipMalloc((void**)&deviceGPU_C, NUM * sizeof(double)));
  
  HIP_ASSERT(hipMemcpy(deviceGPU_B, hostCpu_B, NUM*sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceGPU_C, hostCpu_C, NUM*sizeof(double), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(vector_add, 
                  dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  deviceGPU_A ,deviceGPU_B ,deviceGPU_C ,WIDTH ,HEIGHT);


  HIP_ASSERT(hipMemcpy(hostCpu_A, deviceGPU_A, NUM*sizeof(double), hipMemcpyDeviceToHost));

  // BEGIN::CTRL
  check_solution(hostCpu_A,hostCpu_B,hostCpu_C,NUM);
  write_vector("Vector A",hostCpu_A,NUM);
  // END::CTRL


  HIP_ASSERT(hipFree(deviceGPU_A));
  HIP_ASSERT(hipFree(deviceGPU_B));
  HIP_ASSERT(hipFree(deviceGPU_C));

  free(hostCpu_A);
  free(hostCpu_B);
  free(hostCpu_C);

  //hipResetDefaultAccelerator();

  return 0;
}

