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


void check_solution(double* a_in,double* b_in,double* c_in,int nb)
{
	int errors = 0;
  	for (int i = 0; i < nb; i++) {
	    if (c_in[i] != (a_in[i] + b_in[i])) { errors++; }
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
  
  const int NX = 25600004;
	int size_array = sizeof(double) * NX;

  const double a = 1.0;
  const double b = 2.0;
  const double c = 0.0;

    double *h_vecA = (double *)malloc(size_array);
    double *h_vecB = (double *)malloc(size_array);
    double *h_vecC = (double *)malloc(size_array);

    for (int i = 0; i < NX; i++)
    {
        h_vecA[i] = a;
        h_vecB[i] = b;
    }

    double *d_vecA, *d_vecB, *d_vecC;
    hipMalloc((void **)&d_vecA, size_array);
    hipMalloc((void **)&d_vecB, size_array);
    hipMalloc((void **)&d_vecC, size_array);

    hipMemcpy(d_vecA, h_vecA, size_array, hipMemcpyHostToDevice);
    hipMemcpy(d_vecB, h_vecB, size_array, hipMemcpyHostToDevice);

    const int block_size = 128;
    int grid_size = (NX + block_size - 1) / block_size;

    std::cout<<"block_size ="<<block_size<<"\n";
    std::cout<<"grid_size  ="<<grid_size<<"\n";
    std::cout<<"\n";

    hipLaunchKernelGGL(vector_add, grid_size, block_size, 0, 0, d_vecA, d_vecB, d_vecC, NX);
    hipMemcpy(h_vecC, d_vecC, size_array, hipMemcpyDeviceToHost);


    check_solution(h_vecA,h_vecB,h_vecC,NX);


    free(h_vecA);
    free(h_vecB);
    free(h_vecC);
    hipFree(d_vecA);
    hipFree(d_vecB);
    hipFree(d_vecC);

  return 0;
}

