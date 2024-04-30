
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
#include <cmath>


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


#include "na.hpp"
#include "Tools.hpp"
#include "Taskflow_HPC.hpp"

#include <execution> //C++20
//#include <coroutine> //C++20
//#include "CoroutineScheduler.hpp" //C++20

//Links mpi
//#define USE_MPI
#ifdef USE_MPI
#include <mpi.h>
#endif

//Links omp
#define UseOpenMP
#ifdef UseOpenMP
	#include <omp.h>
#endif




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

__global__ void matrix_mult(double* C, double* A, double* B, int m, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int i = idx / n;
	int k = idx - n * i;
	if (n * m > idx) {
		for (int j = 0; j < n; j++) {
			C[idx] += A[n * i + j] * B[n * j + k];
		}
	}
}

__global__ void matrix_equal(volatile bool *Q, double* A, double* B, int nb, double deltaError) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (idx < nb)
		//if (abs(A[idx]-B[idx])>deltaError) { Q[0]=false; printf("F"); } else  { printf("T"); }
		if (abs(A[idx]-B[idx])>deltaError) { Q[0]=false;  } 
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

void saveMatrixView(const Matrix M, char *filename) 
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


void saveMatrix(const Matrix M, char *filename) 
{
    std::ofstream myfile;
    for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++)
			myfile<<M.elements[i * M.num_rows + j];
    }
    myfile<<"\n";
    myfile.close();
}

Matrix readMatrix(char *filename,const int nc,const int nr) 
{
	Matrix M= allocate_matrix(nc,nr,0);
	std::ifstream myfile;
	myfile.open (filename);
    for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++)
			myfile>>M.elements[i * M.num_rows + j];
    }
    myfile.close();
	return M;
}

void readFileViewInformation(char *filename) 
{
	FILE* FICH = NULL;
    int c = 0;
	FICH = fopen(filename, "r");
    if (FICH != NULL) { do { c = fgetc(FICH); printf("%c",c); } while (c != EOF); fclose(FICH); }
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

Matrix matrix_product(const Matrix A, const Matrix B) 
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

Matrix matrix_copy(const Matrix M) 
{
  Matrix R= allocate_matrix(M.num_columns,M.num_rows,0);
  int i,j;
  for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			R.elements[i * M.num_rows + j] = M.elements[i * M.num_rows + j];
  return R;
}

void matrix_copy_elements(Matrix R,const Matrix M) 
{
  int i,j;
  for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			R.elements[i * M.num_rows + j] = M.elements[i * M.num_rows + j];
}


void matrix_lower_triangular(Matrix M) 
{
    int i, j;
    for (i = 0; i < M.num_rows; i++)
        for (j = 0; j < i; j++)
            M.elements[i * M.num_rows + j] = 0.0;
}


//END:: BUILD INIT MATRIX
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


bool isFileExist(std::string ch)
{
    std::ifstream myfile;
    myfile.open(ch); bool qOK=false;
    if (myfile) { qOK=true; }
    myfile.close();
    return (qOK);
}

/*********************************************************************************************************************************************************/

/*********************************************************************************************************************************************************/
//BEGIN::Product Matrix

Matrix matrix_product_GPU(const Matrix A, const Matrix B) 
{
	int block_size = 512;
	int matrixSize=A.num_columns;
    Matrix C= allocate_matrix(matrixSize,matrixSize,0);

	hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	Matrix gpu_B = allocate_matrix_on_gpu(B);
	Matrix gpu_C = allocate_matrix_on_gpu(C);

	hipEventRecord(start, 0);   

	copy_matrix_to_device(gpu_A, A );
	copy_matrix_to_device(gpu_B, B );
	copy_matrix_to_device(gpu_C, C );
	
	int num_blocks = (matrixSize*matrixSize + block_size - 1) / block_size;
	
	dim3 thread_block(block_size, 1, 1);
	dim3 grid(num_blocks, 1);

	hipLaunchKernelGGL(matrix_mult,grid, thread_block,0,0,gpu_C.elements,gpu_A.elements,gpu_B.elements,matrixSize,matrixSize); 

	copy_matrix_from_device(C,gpu_C);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.elements);
	hipFree(gpu_B.elements);
	hipFree(gpu_C.elements);

	return C;
}


bool is_matrix_equal_GPU(const Matrix A, const Matrix B,const double deltaError) 
{
	int block_size = 512;
	int matrixSize=A.num_columns;
	int sizeQ = sizeof(bool) * 1;
    bool *h_Q = (bool *)malloc(sizeQ);
	h_Q[0]=true;

	hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	Matrix gpu_B = allocate_matrix_on_gpu(B);
	bool  *d_Q;    hipMalloc((void **)&d_Q,sizeQ);

	hipEventRecord(start, 0);   
	copy_matrix_to_device(gpu_A, A );
	copy_matrix_to_device(gpu_B, B );
	hipMemcpy(d_Q,h_Q,sizeQ, hipMemcpyHostToDevice);
	int num_blocks = (matrixSize*matrixSize + block_size - 1) / block_size;
	
	dim3 thread_block(block_size, 1, 1);
	dim3 grid(num_blocks, 1);
	hipLaunchKernelGGL(matrix_equal,grid, thread_block,0,0,d_Q,gpu_A.elements,gpu_B.elements,matrixSize*matrixSize,deltaError); 
	hipMemcpy(h_Q,d_Q,sizeof(bool), hipMemcpyDeviceToHost);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.elements);
	hipFree(gpu_B.elements);
	hipFree(d_Q);

	return (h_Q[0]);
}

bool is_matrix_equal_GPU(const Matrix A, const Matrix B) 
{
	double deltaError=0.000001;
	return(is_matrix_equal_GPU(A,B,deltaError));
}


void checkSolution_GPU(Matrix A,Matrix B)
{
	bool res=is_matrix_equal_GPU(A,B);
	printf("[INFO]:	%s\n", (true == res) ? "WELL DONE PASSED :-)" : "FAILED");
}

//END::Product Matrix
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

#ifdef UseOpenMP
Matrix getCholeskyOpenMPVers1(Matrix A,int num_threads)
{
	omp_set_num_threads(num_threads);
	double t1, t2;
	int k, i, j, l;
	int n = A.num_rows;
	Matrix U=matrix_copy(A);
	double c;
	t1 = omp_get_wtime();
	for(k=0; k<n; k++)
	{
		c=sqrt(U.elements[k*n+k]);  U.elements[k*n+k]=c;
		#pragma omp parallel for
		for(i=k+1; i<=n; i++) {
			U.elements[i+k*n]=U.elements[i+k*n]/c;
		}
		#pragma omp parallel for private(l,j)
		for (l=k+1; l<n; l++) {
			for (j=k+1; j<=l-1; j++) {
				U.elements[l+j*n]=U.elements[l+j*n]-U.elements[l+k*n]*U.elements[j+k*n];
			}
			U.elements[l*n+l]=U.elements[l*n+l]-U.elements[l+k*n]*U.elements[l+k*n];
		}
	}
	t2 = omp_get_wtime();
	//std::cout << "[INFO]: Elapsed microseconds inside: "<<t2-t1<< " us\n";
	matrix_lower_triangular(U);
	return U;
}

Matrix getCholeskyOpenMPVers2(Matrix A,int num_threads)
{
	omp_set_num_threads(num_threads);
	double t1, t2;
	int k, i, j, l;
	int n = A.num_rows;
	Matrix U=matrix_copy(A);
	double c;
	t1 = omp_get_wtime();
	for(k=0; k<n; k++)
	{
		c=sqrt(U.elements[k*n+k]); U.elements[k*n+k]=c;
		for(i=k+1; i<=n; i++) {
			#pragma omp task
			U.elements[i+k*n]=U.elements[i+k*n]/c;
		}

		#pragma omp taskwait
		for (l=k+1; l<=n; l++) {
			for (j=k+1; j<=l-1; j++) {
				#pragma omp task
				U.elements[l+j*n]=U.elements[l+j*n]-U.elements[l+k*n]*U.elements[j+k*n];
			}
			#pragma omp task
			U.elements[l*n+l]=U.elements[l*n+l]-U.elements[l+k*n]*U.elements[l+k*n];
		}
	}
	matrix_lower_triangular(U);
	t2 = omp_get_wtime();
	//std::cout << "[INFO]: Elapsed microseconds inside: "<<t2-t1<< " us\n";
	return U;
}
#endif


#ifdef USE_MPI
void getCholeskyMPIVers2(int argc, char *argv[])
{
	int n=atoi(argv[1]);
    Matrix MatA=allocate_matrix(n,n,0); 
	Matrix MatM=allocate_matrix(n,n,0); 

	//Matrix MatA;
	//Matrix MatM;

    bool qView=true;

	std::chrono::steady_clock::time_point t_begin,t_end;
	long int t_laps;

	int world_rank_mpi,world_size_mpi;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_mpi);
	
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name,&name_len);

    if (world_rank_mpi == 0) { 
      int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
      std::cout << "[INFO]: Name worlds processor: "<<processor_name<<"\n";
      std::cout << "[INFO]: Nb CPU available: "<<numCPU<< "\n";
      std::cout << "\n";
      std::cout << "[INFO]: Scan..."<<"\n";
    }
    std::cout << "[INFO]: rank: "<<world_rank_mpi<<" out of "<<world_size_mpi<<"\n";
   
    MPI_Barrier(MPI_COMM_WORLD); 

    if (world_rank_mpi == 0) { 
      std::cout << "\n";
      MatA = create_positive_definite_matrix(n,n); 
	  matrix_copy_elements(MatM,MatA);	
      if (qView) { writeMatrix(MatA);	}
      t_begin = std::chrono::steady_clock::now();
    }

	MPI_Barrier(MPI_COMM_WORLD); 


    for (int k=0; k<n; k++)
    {		  
      if (world_rank_mpi == 0){
        for (int j=0; j<k; j++)
        {
          MatM.elements[k * n + k]-=(MatM.elements[k * n + j] * MatM.elements[k * n + j]);	
        }
        MatM.elements[k * n + k]=sqrt(MatM.elements[k * n + k]);
        
        for (int p=1; p<n-k; p++) 
        {	
          for (int c=0; c<=k; c++)
          {
            for (int r=c; r<n; r++)
            {	
              MPI_Send(&MatM.elements[r * n + c], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
          }
          MPI_Recv(&MatM.elements[( p + k )* n + k], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }	
        if (k==n-1)
        {	
          for(int i=0; i<n-1; i++)
          {
            for(int j=i+1; j<n; j++)
            {
              MatM.elements[i * n + j]=0;	
            }	
          }
        }	
      }

      else if (world_rank_mpi<n-k){
        for (int c=0; c<=k; c++) 
        {
          for (int r=c; r<n; r++)
          {
            MPI_Recv(&MatM.elements[r * n + c], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        }
        
        for (int g=0; g<k; g++) //calculating non-diagonal elements concurrently
        {
          MatM.elements[(k+world_rank_mpi) * n + k]-=MatM.elements[k * n + g] * MatM.elements[(k+world_rank_mpi) * n + g];		
        }	
        MatM.elements[(k+world_rank_mpi) * n + k]/=MatM.elements[k * n + k];
        MPI_Send(&MatM.elements[(k+world_rank_mpi) * n + k], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      }   
    }	

	if (world_rank_mpi == 0) {
		t_end = std::chrono::steady_clock::now();
    	t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
        std::cout << "\n";
	    std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
        std::cout << "\n";

		std::string chName="DataMPI.csv";
        std::ofstream myfile;
        if (!isFileExist(chName)) {  myfile.open (chName); } else { myfile.open(chName,std::ios::app); }
        myfile<<world_size_mpi<<","<<n<<","<<t_laps<<"\n";
        myfile.close();
		
		if (qView) { 
            printf("[INFO]: Cholesky decomposition of matrix\n");
            if (qView) { writeMatrix(MatM); }
            if (1==1) {
                  std::cout << "[INFO]: Controle matrix product if A=tU*U \n";
                  Matrix MatMt=matrix_tanspose(MatM);
                  //Matrix MatT=matrix_product(MatL,MatLt);
				  Matrix MatT=matrix_product_GPU(MatM,MatMt); 
                  if (qView) { writeMatrix(MatT); }
                  //checkSolution(MatA,MatT);
				  checkSolution_GPU(MatA,MatT);
                  free(MatMt.elements);
                  free(MatT.elements);
            }
            
          }
		  
		free(MatM.elements);
	}
    MPI_Finalize();
}
#endif


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
  
	matrix_lower_triangular(U);
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

	matrix_lower_triangular(U);
		
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


void scanInformationSystem()
{
	int Value;
	std::cout <<"\n";
	std::cout << "[INFO]: Scan Information System..."<<"\n";
	Value=std::system("lscpu>InfoSystemCPU.txt");
	Value=std::system("lshw -C display>InfoSystemGPU.txt");
	std::cout <<"\n";
}

void getInformationCPU()
{
	std::cout <<"\n";
	std::cout << "[INFO]: Inormation CPU"<<"\n";
	readFileViewInformation("InfoSystemCPU.txt");
	std::cout <<"\n";
}

void getInformationGPU()
{
	std::cout <<"\n";
	std::cout << "[INFO]: Inormation GPU"<<"\n";
	readFileViewInformation("InfoSystemGPU.txt");
	std::cout <<"\n";
}

#ifdef USE_MPI
void getMpiInformation(int argc, char *argv[])
{
	//BEGIN::INFO MPI
	bool qFullInfoSystem=false;
    MPI_Init(NULL, NULL);
    int world_rank,world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name,&name_len);

    if (world_rank == 0) { 
	  	std::cout <<"\n";
      	int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
      	std::cout << "[INFO]: Name worlds processor: "<<processor_name<<"\n";
      	std::cout << "[INFO]: Nb CPU available: "<<numCPU<< "\n";
      	std::cout <<"\n";
      	std::cout << "[INFO]: Scan..."<<"\n";
    }
    std::cout << "[INFO]: rank: "<<world_rank<<" out of "<<world_size<<"\n";
    MPI_Finalize();
	//END::INFO MPI
}
#endif


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
        Matrix MatT=matrix_product(MatU_gpu2t,MatU_gpu2);
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

#ifdef UseOpenMP
void TestOpenMP()
{
	std::ofstream myfile;
	myfile.open ("DataScalingOpeMP.csv");

	long int t_laps;
    bool qView=true;
    qView=false;	
	bool qCTRL=true;
	qCTRL=false;

	myfile <<"DimMatrix,nbCPU,dt"<<"\n";	
	std::chrono::steady_clock::time_point t_begin,t_end;

	for (int i = 0; i <= 7; i++)
    {
		int NbThread=pow(2,i);
		const int matrixSize=NbThread*10;

		Matrix MatA; MatA = create_positive_definite_matrix(matrixSize,matrixSize); 
		if (qView) { writeMatrix(MatA); }

		myfile <<matrixSize<<","<<NbThread<<",";
			std::cout << "[INFO]: Method OpenMP"<< "\n";
			std::cout << "[INFO]: NbThread :"<<NbThread<< "\n";
			t_begin = std::chrono::steady_clock::now();
			//Matrix MatU_OpenMP=getCholeskyOpenMPVers1(MatA,NbThread);
			Matrix MatU_OpenMP=getCholeskyOpenMPVers2(MatA,NbThread);
			t_end = std::chrono::steady_clock::now();
			if (qView) { writeMatrix(MatU_OpenMP); }
			t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
			std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
			std::cout << "\n";			
			myfile<<t_laps;
			
			if (1==1) {
				std::cout << "[INFO]: Controle matrix product if A=tU*U \n";
				Matrix MatU_OpenMPt=matrix_tanspose(MatU_OpenMP);
				//Matrix MatT=matrix_product(MatU_OpenMPt,MatU_OpenMP);
				Matrix MatT=matrix_product_GPU(MatU_OpenMPt,MatU_OpenMP);
				//if (qView) { writeMatrix(MatT); }
				//checkSolution(MatA,MatT);
				checkSolution_GPU(MatA,MatT);
				std::cout << "\n";
				free(MatU_OpenMPt.elements);
				free(MatT.elements);
			}
	
			free(MatU_OpenMP.elements);
			free(MatA.elements);
			//sleep(2);
		myfile<<"\n";
	}
	myfile.close();
}
#endif



#ifdef USE_MPI
void TestMPI(int argc, char *argv[])
{
	getCholeskyMPIVers2(argc, argv);
}
#endif



int main(int argc, char *argv[])
 {
	//scanInformationSystem();
	//getInformationCPU();
	//getInformationGPU();
    //getHipInformation();
	//getMpiInformation(argc,argv);
	
	//TestMPI(argc,argv);
	TestOpenMP();

	//TestLevel001();
	//TestLevel002();	
	//TestLevel003();
	//TestLevel004();
	
}
