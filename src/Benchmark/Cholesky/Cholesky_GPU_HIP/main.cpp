
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
//#include "hip/hip_runtime.h"
//#include "roctx.h"
//#include "roctracer_ext.h"


//Links for dev
#include <thread>
#include <vector>
#include <array>
#include <typeinfo>
#include <iostream>
#include <mutex>
#include <sched.h>
#include <pthread.h>
#include <algorithm> //for Each_fors
#include <string>
#include <utility>
#include <functional>
#include <future>
#include <cassert>
#include <chrono>
#include <type_traits>
#include <list>
#include <ranges>
#include <atomic> 



//Links Specx
#include "SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"
#include "Utils/SpTimer.hpp"
#include "Utils/small_vector.hpp"
#include "Utils/SpConsumerThread.hpp"

//Links Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>




//#include "/home/u2/lemoine/SpecxProjects/external/Test13/na.hpp"
//#include "/home/u2/lemoine/SpecxProjects/external/Test13/Tools.hpp"
//#include "/home/u2/lemoine/SpecxProjects/external/Test13/TIT.hpp"

#include "na.hpp"
#include "Tools.hpp"
//#include "TITb.hpp"
#include "Taskflow_HPC.hpp"

#include <execution> //C++20
//#include <coroutine> //C++20
//#include "CoroutineScheduler.hpp" //C++20


#include <cmath>

#define HIP_CHECK(command) {               \
  hipError_t status = command;             \
  if (status!=hipSuccess) {                \
    std::cerr <<"Error: HIP reports "<< hipGetErrorString(status)<< std::endl; \
    std::abort(); } }


#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif

//#define BLOCK_SIZE 128
//#define BLOCK_SIZE 1024

 typedef struct {
 	  unsigned int num_columns;
   	  unsigned int num_rows;
 	  unsigned int pitch; 
 	  double* elements;
  } Matrix;







/*********************************************************************************************************************************************************/



/*********************************************************************************************************************************************************/
// BEGIN::INTRODUCTION
int check_if_symmetric                 (const Matrix M); 
int check_if_diagonal_dominant         (const Matrix M);
Matrix create_positive_definite_matrix (unsigned int, unsigned int);
Matrix allocate_matrix                 (int num_rows, int num_columns, int init);

void writeMatrix                       (const Matrix);
void copy_matrix_to_device             (Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device           (Matrix Mhost,   const Matrix Mdevice);
Matrix allocate_matrix_on_gpu          (const Matrix M);
// END::INTRODUCTION
/*********************************************************************************************************************************************************/



/*********************************************************************************************************************************************************/
// BEGIN::HIP AMD GPU

__global__ void chol_kernel(double * U, int ops_per_thread, int m) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i, j, k;
    unsigned int num_rows = m;
    for (k = 0; k < num_rows; k++) {
        if (tx == 0) {
            U[k * num_rows + k] = sqrt(U[k * num_rows + k]);
            for (j = (k + 1); j < num_rows; j++) {
                U[k * num_rows + j] /= U[k * num_rows + k];
            }
        }
        __syncthreads();
        int istart = ( k + 1 )  +  tx * ops_per_thread;
        int iend = istart + ops_per_thread;      
        for (i = istart; i < iend; i++) {
            for (j = i; j < num_rows; j++) {
                U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
            }
        }
        __syncthreads();
    }
    __syncthreads();
    int istart = tx * ops_per_thread;
    int iend   = istart + ops_per_thread;
    for (i = istart; i < iend; i++) {
        for (j = 0; j < i; j++) {
            U[i * num_rows + j] = 0.0;
        }
    }
}


__global__ void chol_kernel_optimized_div(double * U, int k, int stride, int m) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j;
    unsigned int num_rows = m;
    if (tx == 0) { U[k * num_rows + k] = sqrt(U[k * num_rows + k]); }
    int offset  = (k + 1); 
    int jstart  = threadIdx.x + offset;
    int jstep   = stride;
    int jtop    = num_rows - 1;
    int jbottom = (k + 1);
    if (blockIdx.x == 0) {
        for (j = jstart; (j >= jbottom) && (j <= jtop); j += jstep) {
            U[k * num_rows + j] /= U[k * num_rows + k]; 
        }
    }
}

__global__ void chol_kernel_optimized(double * U, int k, int stride, int m) {
    unsigned int j;
    unsigned int num_rows = m; 
    int i       = blockIdx.x + (k + 1);
    int offset  = i;
    int jstart  = threadIdx.x + offset;
    int jstep   = stride;
    int jtop    = num_rows - 1;
    int jbottom = i;
    for (j = jstart; (j >= jbottom) && (j <= jtop); j += jstep) {
        U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
    }
}

// END::HIP AMD GPU
/*********************************************************************************************************************************************************/


/*********************************************************************************************************************************************************/
//BEGIN::TOOLS MEMORY TRANSFER HIP AMD GPU

Matrix allocate_matrix(int num_rows, int num_columns, int init) {
    Matrix M;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;

    M.elements = (double *) malloc(size * sizeof (double));
    for (unsigned int i = 0; i < size; i++) {
        if (init == 0) M.elements[i] = 0;
        else
            M.elements[i] = (double) rand() / (double) RAND_MAX;
    }
    return M;
}


Matrix allocate_matrix_on_gpu(const Matrix M){
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(double);
    hipMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}


void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(double);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    hipMemcpy(Mdevice.elements, Mhost.elements, size, hipMemcpyHostToDevice);
}

void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice){
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(double);
    hipMemcpy(Mhost.elements, Mdevice.elements, size, hipMemcpyDeviceToHost);
}
//END::TOOLS MEMORY TRANSFER HIP AMD GPU
/*********************************************************************************************************************************************************/

/*********************************************************************************************************************************************************/
//BEGIN:: BUILD INIT MATRIX

Matrix create_positive_definite_matrix(unsigned int num_rows, unsigned int num_columns)
{
	Matrix M;
	M.num_columns = M.pitch = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (double *)malloc(size * sizeof(double));

	// Step 1: Create a matrix with random numbers between [-.5 and .5]
	printf("Creating a %d x %d matrix with random numbers between [-.5, .5]...", num_rows, num_columns);
	unsigned int i;
	unsigned int j;
	for(i = 0; i < size; i++)
		M.elements[i] = ((double)rand()/(double)RAND_MAX) - 0.5;
       	printf("done. \n");
	// writeMatrix(M);
	// getchar();

	// Step 2: Make the matrix symmetric by adding its transpose to itself
	printf("Generating the symmetric matrix...");
	Matrix transpose;
	transpose.num_columns = transpose.pitch = num_columns;
	transpose.num_rows = num_rows; 
	size = transpose.num_rows * transpose.num_columns;
	transpose.elements = (double *)malloc(size * sizeof(double));

	for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			transpose.elements[i * M.num_rows + j] = M.elements[j * M.num_columns + i];
	// writeMatrix(transpose);

	for(i = 0; i < size; i++)
		M.elements[i] += transpose.elements[i];
	if (check_if_symmetric(M))
		printf("done. \n");
	else{ 
		printf("error. \n");
		free(M.elements);
		M.elements = NULL;
	}
	// Step 3: Make the diagonal entries large with respect to the row and column entries
	printf("Generating the positive definite matrix...");
	for(i = 0; i < num_rows; i++)
		for(j = 0; j < num_columns; j++){
			if(i == j) 
				M.elements[i * M.num_rows + j] += 0.5 * M.num_rows;
		}
	if(check_if_diagonal_dominant(M))
		printf("done. \n");
	else{
		printf("error. \n");
		free(M.elements);
		M.elements = NULL;
	}
	free(transpose.elements);
	return M;
}


void writeMatrix(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
			printf("%f ", M.elements[i*M.num_rows + j]);
		printf("\n");
	} 
	printf("\n");
}

void saveMatrix(const Matrix M, char *filename) 
{
    FILE* FICH = fopen(filename,"w");
    for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++)
            fprintf(FICH,"%f ", M.elements[i * M.num_rows + j]);
        fprintf(FICH,"\n");
    }
    fprintf(FICH,"\n");
    fclose(FICH);
}

void readMatrix(const Matrix M, char *filename) 
{

}

int check_if_symmetric(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++)
		for(unsigned int j = 0; j < M.num_columns; j++)
			if(M.elements[i * M.num_rows + j] != M.elements[j * M.num_columns + i]) return 0;
	return 1;
}

int check_if_diagonal_dominant(const Matrix M)
{
	float diag_element;
	float sum;
	for(unsigned int i = 0; i < M.num_rows; i++){
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for(unsigned int j = 0; j < M.num_columns; j++){
			if(i != j) sum += abs(M.elements[i * M.num_rows + j]);
		}
		if(diag_element <= sum) return 0;
	}
	return 1;
}

Matrix matrix_multiply(const Matrix A, const Matrix B) 
{
    Matrix C;
    C.num_columns = C.pitch = A.num_columns;
    C.num_rows = A.num_rows;
    unsigned int size = C.num_rows * C.num_columns;
    C.elements = (double *) malloc(size * sizeof (double));

    for (unsigned int i = 0; i < A.num_columns; i++)
        for (unsigned int j = 0; j < B.num_rows; j++) {
            double sum = 0.0f;
            for (unsigned int k = 0; k < A.num_columns; k++) {
                double a = A.elements[i * A.num_columns + k];
                double b = B.elements[k * B.num_rows + j];
                sum += a * b;
            }
            C.elements[i * B.num_rows + j] = (double) sum;
        }
    return C;
}

Matrix matrix_tanspose(const Matrix M) 
{
  Matrix R= allocate_matrix(M.num_columns,M.num_rows,0);
  int i,j;
  for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			R.elements[i * M.num_rows + j] = M.elements[j * M.num_columns + i];
  return R;
}

//END:: BUILD INIT MATRIX
/*********************************************************************************************************************************************************/


/*********************************************************************************************************************************************************/
//BEGIN::Cholesky Factoristation

Matrix getCholeskySerial(Matrix A)
{
  int i,j,k;
  int n = A.num_rows;
  Matrix L = allocate_matrix(n,n,0);
  for (i = 0; i < n; i++)
      for (j = 0; j < (i+1); j++) {
            double s = 0;
            for (k = 0; k < j; k++)
                s += L.elements[i * n + k] * L.elements[j * n + k];
            L.elements[i * n + j] = (i == j) ? sqrt(A.elements[i * n + i] - s) : (1.0 / L.elements[j * n + j] * (A.elements[i * n + j] - s));
  }
  Matrix U=matrix_tanspose(L);
  return U;
}


Matrix getCholeskyGPUVers1(Matrix A)
{
	int matrixSize=A.num_rows;
	Matrix U= allocate_matrix(matrixSize,matrixSize,0);
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);
	int num_blocks = 1;
	int threads_per_block = 512;
	int ops_per_thread = matrixSize/(threads_per_block * num_blocks);
	hipEventRecord(start, 0);
	Matrix gpu_u = allocate_matrix_on_gpu(U);

	copy_matrix_to_device(gpu_u, A );
	dim3 thread_block(threads_per_block, 1, 1);
	dim3 grid(num_blocks, 1);

	hipLaunchKernelGGL(chol_kernel,grid, thread_block,0,0,gpu_u.elements,ops_per_thread,matrixSize); 
	hipDeviceSynchronize();
	copy_matrix_from_device(U,gpu_u);
	hipEventRecord(stop, 0);
	hipEventSynchronize(stop);
	hipFree(gpu_u.elements);
  
	int i, j;
	for (i = 0; i < matrixSize; i++)
		for (j = 0; j < i; j++)
			U.elements[i * matrixSize + j] = 0.0;
	return U;
}


Matrix getCholeskyGPUVers2(Matrix A)
{
	int matrixSize=A.num_rows;
    Matrix U= allocate_matrix(matrixSize,matrixSize,0);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    int threads_per_block = 256; 
    int stride = threads_per_block;    
    hipEventRecord(start, 0);    
    Matrix gpu_u = allocate_matrix_on_gpu(U);
    copy_matrix_to_device(gpu_u, A);
    int k;
    for (k = 0; k < matrixSize; k++) {
        int isize = (matrixSize - 1) - (k + 1) + 1;
        int num_blocks = isize;
        if (num_blocks <= 0) { num_blocks = 1; }
        dim3 thread_block(threads_per_block, 1, 1);
        dim3 grid(num_blocks, 1);
        hipLaunchKernelGGL(chol_kernel_optimized_div,grid, thread_block,0,0,gpu_u.elements,k,stride,matrixSize); 
        hipLaunchKernelGGL(chol_kernel_optimized,grid, thread_block,0,0,gpu_u.elements,k,stride,matrixSize); 
    }
    copy_matrix_from_device(U, gpu_u);  				 
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipFree(gpu_u.elements);

    //As the final step, zero out the lower triangular portion of U
    int i, j;
    for (i = 0; i < matrixSize; i++)
        for (j = 0; j < i; j++)
            U.elements[i * matrixSize + j] = 0.0;
		
    return U;
}

//END::Cholesky Factoristation
/*********************************************************************************************************************************************************/


/*********************************************************************************************************************************************************/
//BEGIN::pthread Cholesky Factoristation

typedef struct chol_pthread_args
{
	Matrix A,U;
	int copyi_start,copyi_end;
	int zeroi_start,zeroi_end;
	pthread_barrier_t * barrier;
	int id;
    int nbTh;
} chol_pthread_args;


//Range splitter helper function
void range_splitter(int size, int num_threads, int * items_per_thread, int * items_last_thread)
{
	//Divide up total size by number of threads
	//How many are left over?
	int elems_left_over = size%num_threads;
	int elements_per_thread = size/num_threads;
	int last_thread_elements = elements_per_thread;

	if(elems_left_over !=0)
	{
		last_thread_elements = elements_per_thread+elems_left_over;
	}

	//Double check because math is hard
	if( (((num_threads-1)*elements_per_thread) + last_thread_elements) != size || (last_thread_elements<0))
	{
		printf("AH! MATH! threads:%d elementsperthread:%d lastthreadelm:%d size:%d leftover:%d\n", num_threads,elements_per_thread,last_thread_elements,size,elems_left_over);
		exit(-1);
	}
	*items_per_thread = elements_per_thread;
	*items_last_thread = last_thread_elements;
}


void range_maker(int items_per_thread,int items_last_thread, int num_threads, int index, int offset, int is_last_thread, int * start, int * end)
{
    *start=items_per_thread*index + offset;
	if (is_last_thread==1) { *end = *start + items_last_thread;   }
	else                   { *end = *start + items_per_thread -1; }
}

void populate_thread_args(chol_pthread_args * arg_list,Matrix A, Matrix U, pthread_barrier_t * barrier,int NbThread)
{
	//Matrix size
	unsigned int size = A.num_rows * A.num_columns;

	//Copy
	int copyisize = size;
	int copyi_items_per_thread, copyi_items_last_thread;
	range_splitter(copyisize,NbThread, &copyi_items_per_thread, &copyi_items_last_thread);

	//Zero out
	int zeroisize = U.num_rows;
	int zeroi_items_per_thread, zeroi_items_last_thread;
	range_splitter(zeroisize,NbThread, &zeroi_items_per_thread, &zeroi_items_last_thread);

	//Zero offset for both sets of work
	int offset = 0;

	//Loop through threads
	int i;
	for(i=0;i<NbThread; i++)
	{
		//Easy ones for all threads
		arg_list[i].A=A;
		arg_list[i].U=U;
		arg_list[i].barrier = barrier;
		arg_list[i].id = i;
        arg_list[i].nbTh = NbThread;

		if(i == (NbThread-1))
		{
			//Last thread
			range_maker(copyi_items_per_thread,copyi_items_last_thread,NbThread, i, offset,1, &(arg_list[i].copyi_start), &(arg_list[i].copyi_end));
			range_maker(zeroi_items_per_thread,zeroi_items_last_thread,NbThread, i, offset,1, &(arg_list[i].zeroi_start), &(arg_list[i].zeroi_end));
		}
		else
		{
			//Regular threads
			range_maker(copyi_items_per_thread,copyi_items_last_thread,NbThread, i, offset,0, &(arg_list[i].copyi_start), &(arg_list[i].copyi_end));
			range_maker(zeroi_items_per_thread,zeroi_items_last_thread,NbThread, i, offset,0, &(arg_list[i].zeroi_start), &(arg_list[i].zeroi_end));
		}
	}
}

void sync_pthreads(pthread_barrier_t * barrier, int thread_id)
{
	// Synchronization point
    int rc = pthread_barrier_wait(barrier);
    if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
    {
        printf("Could not wait on barrier.\n");
        exit(-1);
    }
    //std::cout<<"sync_pthreads\n";
}


void * chol_pthread(void * arg)
{
    //std::cout<<"BEGIN::chol_pthread\n";
	//Get arg as struct
	chol_pthread_args * args = (chol_pthread_args *)arg;
	//Matrices
	Matrix A = args->A;
	Matrix U = args->U;
    int NbThread = args->nbTh;

	//Copy work
	int copyi_start = args->copyi_start;
	int copyi_end = args->copyi_end;
	//Zero out work
	int zeroi_start = args->zeroi_start;
	int zeroi_end = args->zeroi_end;
	//Barrier to sync
	pthread_barrier_t * barrier = args->barrier;
	//Id
	int id = args->id;

    if (id == NbThread - 1) {
        zeroi_end--;
        copyi_end--;
    }


	//Iterators
	unsigned int i, j, k;
	unsigned int size = A.num_rows * A.num_columns;

	//Copy the contents of the A matrix into the working matrix U
	for (i = copyi_start; i <= copyi_end; i ++)
	{
		U.elements[i] = A.elements[i];
	}

	//Sync threads!!!
	sync_pthreads(barrier, id);

	// Perform the Cholesky decomposition in place on the U matrix
	for(k = 0; k < U.num_rows; k++)
	{
		//Only one thread does squre root and division
		if(id==0)
		{
			// Take the square root of the diagonal element
			U.elements[k * U.num_rows + k] = sqrt(U.elements[k * U.num_rows + k]);
			if(U.elements[k * U.num_rows + k] <= 0){
					 printf("Cholesky decomposition failed. \n");
					 return 0;
			}

			// Division step
			for(j = (k + 1); j < U.num_rows; j++)
			{
				U.elements[k * U.num_rows + j] /= U.elements[k * U.num_rows + k]; // Division step
			}
		}

		//Sync threads!!!!!
		sync_pthreads(barrier, id);

		//For this k iteration, split up i
		//Size of i range originally
		int isize = U.num_rows - (k + 1);
		int items_per_thread, items_last_thread;
		range_splitter(isize,NbThread, &items_per_thread, &items_last_thread);
		//Divy up work
		//Elim work
		int elimi_start, elimi_end;
		int offset = (k + 1); //To account for not starting at i=0 each time

		elimi_start=items_per_thread*id + offset;
		if(id == (NbThread-1))
		{
			//Last thread
			elimi_end = elimi_start + items_last_thread;
		}
		else
		{
			//Regular threads
			elimi_end = elimi_start + items_per_thread -1 ;
		}

		// Elimination step
		//printf("K Loop. I range %d to %d\n",(k + 1),U.num_rows-1);
		for(i = elimi_start; i <=elimi_end; i++)
		{
			for(j = i; j < U.num_rows; j++)
			{
				U.elements[i * U.num_rows + j] -= U.elements[k * U.num_rows + i] * U.elements[k * U.num_rows + j];
			}
		}

		//Sync threads!!!!!
		sync_pthreads(barrier, id);
	}

	//Sync threads!!!!!
	sync_pthreads(barrier, id);

	// As the final step, zero out the lower triangular portion of U
	for(i = zeroi_start; i <=zeroi_end; i++)
	{
		for(j = 0; j < i; j++)
		{
			U.elements[i * U.num_rows + j] = 0.0;
		}
	}
	//Don't sync, will join outside here
    //writeMatrix(U);
    //std::cout<<"END::chol_pthread\n";
    return NULL;
}


Matrix getCholesky_pthreads(const Matrix A,int NbThread)
{
	int matrixSize=A.num_rows;
    int i;
    Matrix U= allocate_matrix(matrixSize,matrixSize,0);
	pthread_t threads[NbThread];
	pthread_barrier_t barrier;
	pthread_barrier_init(&barrier, NULL,NbThread);
	chol_pthread_args args[NbThread];
	populate_thread_args(&args[0],A,U,&barrier,NbThread);
    for(i=0; i < NbThread; i++) { 
        int status=pthread_create(&threads[i],NULL,chol_pthread,&(args[i])); 
        if(status != 0) { fprintf(stderr, "pthread_create failed with i = %d. errno = %d, %s\n",i, errno, strerror(errno)); }
    }
    for(i=0; i < NbThread; i++) { pthread_join(threads[i],NULL); }
    return U;
}

//END::pthread Cholesky Factoristation
/*********************************************************************************************************************************************************/


/*********************************************************************************************************************************************************/
unsigned compareArrays(double *reference,double * device, int size)
{    
    for(int i=0; i<size*size; i++) {        
        float epsilon = 0.15;        
        int x = i / size;
        int y = i % size;
        if(x==y){ epsilon = 1; }        
        if (fabs(reference[i] - device[i]) > epsilon) {
            printf("\ni=%d : reference=%f  !=  device=%f   | x=%d y=%d   \n" , i, reference[i], device[i], x, y);
            return 0;
        }
    }
    return 1;
}

void checkSolution(Matrix MatRef,Matrix MatRes)
{
    unsigned res = compareArrays(MatRef.elements, MatRes.elements,MatRef.num_rows);
    printf("[INFO]:	%s\n", (1 == res) ? "WELL DONE PASSED :-)" : "FAILED");
}

/*********************************************************************************************************************************************************/

void getHipInformation()
{
  //BEGIN::INFO HIP AMD
    std::cout<<std::endl;
    int numDevices=0;
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    std::cout<<"[INFO]: Get numDevice                = "<<numDevices<<"\n";
    int deviceID=0;
    HIP_CHECK(hipGetDevice(&deviceID));
    std::cout<<"[INFO]: Get deviceID activated       = "<<deviceID<<"\n";
    deviceID=0;
    hipSetDevice(deviceID);

    hipDeviceProp_t devProp;
    for (int i = 0; i < numDevices; i++)
    {
                HIP_CHECK(hipSetDevice(i));
                HIP_CHECK(hipGetDeviceProperties(&devProp,i));
                std::cout<<"[INFO]:"<<std::endl;
                std::cout<<"[INFO]: DeviceID                     = "<<i<<std::endl;
                std::cout<<"[INFO]: Agent prop name              = "<< devProp.name<<std::endl;
                std::cout<<"[INFO]: System minor                 = "<< devProp.minor<<std::endl;
                std::cout<<"[INFO]: System major                 = "<< devProp.major<<std::endl;
                std::cout<<"[INFO]: Memory Clock Rate (KHz)      = "<< devProp.memoryClockRate<<std::endl;
                std::cout<<"[INFO]: Memory Bus Width (bits)      = "<< devProp.memoryBusWidth<<std::endl;
                std::cout<<"[INFO]: Peak Memory Bandwidth (GB/s) = "<< 2.0*devProp.memoryClockRate*(devProp.memoryBusWidth/8)/1.0e6<<std::endl;
                std::cout<<"[INFO]: max ThreadsPerBlock          = "<< devProp.maxThreadsPerBlock<<std::endl;
                std::cout<<"[INFO]: max ThreadsPerMultiProcessor = "<< devProp.maxThreadsPerMultiProcessor<<std::endl;
                std::cout<<"[INFO]: max ThreadsDim 3D            = "<< devProp.maxThreadsDim[0]<<" "<<devProp.maxThreadsDim[1]<<" "<<devProp.maxThreadsDim[2]<<std::endl;
                std::cout<<"[INFO]: max Grid Size 3D             = "<< devProp.maxGridSize[0]<<" "<<devProp.maxGridSize[1]<<" "<<devProp.maxGridSize[2]<<std::endl;
                std::cout<<"[INFO]: warpSize:                    = "<< devProp.warpSize << "\n";
                std::cout<<"[INFO]: regsPerBlock:                = "<< devProp.regsPerBlock << "\n";
                std::cout<<"[INFO]: concurrentKernels:           = "<< devProp.concurrentKernels << "\n";
                std::cout<<"[INFO]: total Global Mem             = "<< devProp.totalGlobalMem<<std::endl;
                std::cout<<"[INFO]: shared Mem Per Block         = "<< devProp.sharedMemPerBlock<<std::endl;
    }

    HIP_CHECK(hipSetDevice(0));
    std::cout<<std::endl;
    //END::INFO HIP AMD
}


/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/

void TestLevel001()
{
	const int matrixSize=2000;
    bool qView=true;
    qView=false;
	std::chrono::steady_clock::time_point t_begin,t_end;
	long int t_laps;
	
	Matrix MatA; 
    MatA = create_positive_definite_matrix(matrixSize,matrixSize); 
    //une matrice définie positive est une matrice positive inversible. XtMX>0.
	if (qView) { writeMatrix(MatA);	}
	
    std::cout << "\n";

    std::cout << "[INFO]: Method Serial"<< "\n";
	t_begin = std::chrono::steady_clock::now();
	Matrix MatU_Serial=getCholeskySerial(MatA);
	t_end = std::chrono::steady_clock::now();
	if (qView) { writeMatrix(MatU_Serial); }
	t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
	std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
    std::cout << "\n";
	
    /*
	t_begin = std::chrono::steady_clock::now();
	Matrix MatU_gpu1=getCholeskyGPUVers1(MatA);
	t_end = std::chrono::steady_clock::now();
	if (qView) { writeMatrix(MatU_gpu1); }
	checkSolution(MatU_Serial,MatU_gpu1);
	t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
	std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
    std::cout << "\n";
    */
	
    std::cout << "[INFO]: Method GPU HIP AMD"<< "\n";
	t_begin = std::chrono::steady_clock::now();
	Matrix MatU_gpu2=getCholeskyGPUVers2(MatA);
	t_end = std::chrono::steady_clock::now();
	if (qView) { writeMatrix(MatU_gpu2); }
	checkSolution(MatU_Serial,MatU_gpu2);
	t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
	std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
    std::cout << "\n";

    std::cout << "[INFO]: Method pthread"<< "\n";
    int NbThread=2;
    t_begin = std::chrono::steady_clock::now();
    Matrix MatU_pthreads=getCholesky_pthreads(MatA,NbThread);
    t_end = std::chrono::steady_clock::now();
    if (qView) { writeMatrix(MatU_pthreads); }
	checkSolution(MatU_Serial,MatU_pthreads);
	t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
	std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
    std::cout << "\n";
    std::cout << "\n";
    
    if (1==0) {
        std::cout << "[INFO]: Controle matrix product if A=tU*U \n";
        Matrix MatU_gpu2t=matrix_tanspose(MatU_gpu2);
        Matrix MatT=matrix_multiply(MatU_gpu2t,MatU_gpu2);
        if (qView) { writeMatrix(MatT); }
        checkSolution(MatA,MatT);
        free(MatU_gpu2t.elements);
        free(MatT.elements);
    }
	
	free(MatU_Serial.elements);
	//free(MatU_gpu1.elements);
	free(MatU_gpu2.elements);
	free(MatU_pthreads.elements);
	free(MatA.elements);
}

void TestLevel001beta()
{
	int matrixSize=1000;
	Matrix MatA; MatA = create_positive_definite_matrix(matrixSize,matrixSize); 
	Matrix MatU_gpu2=getCholeskyGPUVers2(MatA);
	free(MatU_gpu2.elements);
}

void TestLevel002()
{
	std::ofstream myfile;
	myfile.open ("Data.csv");
	long int t_laps;
    bool qView=true;
    qView=false;	
	bool qCTRL=true;
	qCTRL=false;

	myfile <<"DimMatrix,ModeSerial,ModeGpuHipAMD,pthread"<<"\n";	
	std::chrono::steady_clock::time_point t_begin,t_end;

	for (int i = 1; i <= 22; i++)
    {
		const int matrixSize=i*500;
		Matrix MatA; MatA = create_positive_definite_matrix(matrixSize,matrixSize); 
		myfile <<matrixSize<<",";


			std::cout << "[INFO]: Method Serial"<< "\n";
			t_begin = std::chrono::steady_clock::now();
			Matrix MatU_Serial=getCholeskySerial(MatA);
			t_end = std::chrono::steady_clock::now();
			if (qView) { writeMatrix(MatU_Serial); }
			t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
			std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
			std::cout << "\n";

			myfile<<t_laps<<",";


			std::cout << "[INFO]: Method GPU HIP AMD"<< "\n";
			t_begin = std::chrono::steady_clock::now();
			Matrix MatU_gpu2=getCholeskyGPUVers2(MatA);
			t_end = std::chrono::steady_clock::now();
			if (qView) { writeMatrix(MatU_gpu2); }
			if (qCTRL) { checkSolution(MatU_Serial,MatU_gpu2); }
			t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
			std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
			std::cout << "\n";

			myfile<<t_laps<<",";


			std::cout << "[INFO]: Method pthread"<< "\n";
			int NbThread=9;
			t_begin = std::chrono::steady_clock::now();
			Matrix MatU_pthreads=getCholesky_pthreads(MatA,NbThread);
			t_end = std::chrono::steady_clock::now();
			if (qView) { writeMatrix(MatU_pthreads); }
			if (qCTRL) {checkSolution(MatU_Serial,MatU_pthreads); }
			t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
			std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
			std::cout << "\n";
		
			myfile<<t_laps;

			free(MatU_Serial.elements);
			//free(MatU_gpu1.elements);
			free(MatU_gpu2.elements);
			free(MatU_pthreads.elements);
			free(MatA.elements);

			sleep(2);
		myfile<<"\n";
	}
	myfile.close();
}


void TestLevel003()
{

	std::ofstream myfile;
	myfile.open ("DataScaling.csv");

	long int t_laps;
    bool qView=true;
    qView=false;	
	bool qCTRL=true;
	qCTRL=false;

	myfile <<"DimMatrix,nbCPU,dt"<<"\n";	
	std::chrono::steady_clock::time_point t_begin,t_end;

	for (int i = 0; i <= 5; i++)
    {
		int NbThread=pow(2,i);
		const int matrixSize=NbThread*100;

		Matrix MatA; MatA = create_positive_definite_matrix(matrixSize,matrixSize); 
		myfile <<matrixSize<<","<<NbThread<<",";

			std::cout << "[INFO]: Method pthread"<< "\n";
			
			t_begin = std::chrono::steady_clock::now();
			Matrix MatU_pthreads=getCholesky_pthreads(MatA,NbThread);
			t_end = std::chrono::steady_clock::now();
			if (qView) { writeMatrix(MatU_pthreads); }
			t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
			std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
			std::cout << "\n";			
			myfile<<t_laps;
			free(MatU_pthreads.elements);
			free(MatA.elements);
			sleep(2);
		myfile<<"\n";
	}
	myfile.close();
}


int main() {
    getHipInformation();
	//TestLevel001();
	TestLevel002();	
	//TestLevel003();
}
