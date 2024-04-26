
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
//#include <eigen3/Eigen/Core>
//#include <eigen3/Eigen/Dense>




//#include "/home/u2/lemoine/SpecxProjects/external/Test13/na.hpp"
//#include "/home/u2/lemoine/SpecxProjects/external/Test13/Tools.hpp"
//#include "/home/u2/lemoine/SpecxProjects/external/Test13/TIT.hpp"

#include "na.hpp"
#include "Tools.hpp"
//#include "TITb.hpp"
//#include "Taskflow_HPC.hpp"

#include <execution> //C++20
//#include <coroutine> //C++20
//#include "CoroutineScheduler.hpp" //C++20


#include <cmath>

#include <mpi.h>


//Links HIP
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
//#include <hip/hip_runtime.h>



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

 typedef struct {
 	  unsigned int num_columns;
   	  unsigned int num_rows;
 	  unsigned int pitch; 
 	  double* elements;
  } Matrix;


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

bool isFileExist(std::string ch)
{
    std::ifstream myfile;
    myfile.open(ch); bool qOK=false;
    if (myfile) { qOK=true; }
    myfile.close();
    return (qOK);
}

int main(int argc, char** argv) {

    //mpirun -np 12 ./main
    if (argc < 3) 
    {
      //std::cout<<"add param"<< "\n";
      //getchar();
      exit (0);
    }

    int n=atoi(argv[1]);
    Matrix MatA; 
    //bool qView=true;
    bool qView=atoi(argv[2]);
    std::chrono::steady_clock::time_point t_begin,t_end;
	  long int t_laps;

    MPI_Init(NULL, NULL);
    int world_rank,world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name,&name_len);

    if (world_rank == 0) { 
      int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
      std::cout << "[INFO]: Name worlds processor: "<<processor_name<<"\n";
      std::cout << "[INFO]: Nb CPU available: "<<numCPU<< "\n";
      std::cout <<"\n";
      std::cout << "[INFO]: Scan..."<<"\n";
    }
    std::cout << "[INFO]: rank: "<<world_rank<<" out of "<<world_size<<"\n";
   
    MPI_Barrier(MPI_COMM_WORLD); 
	  
    double M[n][n];
    if (world_rank == 0) { 
      std::cout << "\n";
      MatA = create_positive_definite_matrix(n,n); 
      for (int i = 0; i <n; i++){
          for (int j = 0; j <n; j++){ M[i][j]=MatA.elements[i * n + j]; }
      }		
      printf("\n");
      if (qView) { writeMatrix(MatA);	}
      t_begin = std::chrono::steady_clock::now();
    }


    for (int k=0; k<n; k++)//k-column index (algorithm goes below of diagonal from left to right)
    {		  
      if(world_rank == 0){
        //0 procesor calculates every diagonal element
        for (int j=0; j<k; j++)
        {
          M[k][k]-=(M[k][j]*M[k][j]);	
        }
        M[k][k]=sqrt(M[k][k]);
        
        for (int p=1; p<n-k; p++) //p-processor index to which 0 processor sends a given task
        {	
          for (int c=0; c<=k; c++)//processor 0 sends already calculated values
          {
            for (int r=c; r<n; r++)
            {	
              MPI_Send(&M[r][c], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
          }
          MPI_Recv(&M[p+k][k], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }	
        if (k==n-1)
        {	//put 0's above diagonal
          for(int i=0; i<n-1; i++)
          {
            for(int j=i+1; j<n; j++)
            {
              M[i][j]=0;	
            }	
          }

          if (1==1) {
            t_end = std::chrono::steady_clock::now();
            t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
            std::cout << "\n";
	          std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
            std::cout << "\n";

            std::string chName="./DATA/DataMPI.csv";
            std::ofstream myfile;
            if (!isFileExist(chName)) {  myfile.open (chName); } else { myfile.open(chName,std::ios::app); }
            myfile<<world_size<<","<<n<<","<<t_laps<<"\n";
            myfile.close();
          }

          if (qView) { 
            Matrix MatL; 
            MatL.num_columns= n;
            MatL.num_rows=n;
            MatL.elements = (double *) malloc(MatL.num_rows*MatL.num_columns* sizeof(double));
            for(int k = 0; k < n*n; k++) MatL.elements[k] = 0.0;

            printf("[INFO]: Cholesky decomposition of matrix\n");
            for (int i = 0; i <n; i++){
              for (int j = 0; j <n; j++){ MatL.elements[i * n + j]=double(M[i][j]); }
            }

            if (qView) { writeMatrix(MatL); }

            if (1==0) {
                  std::cout << "[INFO]: Controle matrix product if A=tU*U \n";
                  Matrix MatLt=matrix_tanspose(MatL);
                  Matrix MatT=matrix_multiply(MatL,MatLt);
                  if (qView) { writeMatrix(MatT); }
                  //checkSolution(MatA,MatT);
                  free(MatLt.elements);
                  free(MatT.elements);
            }


            free(MatL.elements);
          }

        }	
      }

      else if (world_rank<n-k){
        for (int c=0; c<=k; c++) 
        {
          for (int r=c; r<n; r++)
          {
            MPI_Recv(&M[r][c], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        }
        
        for (int g=0; g<k; g++) //calculating non-diagonal elements concurrently
        {
          M[k+world_rank][k]-=M[k][g]*M[k+world_rank][g];		
        }	
        M[k+world_rank][k]/=M[k][k];
        MPI_Send(&M[k+world_rank][k], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      }   
    }	 

    if (world_rank == 0) { 
      free(MatA.elements); 
    }
    MPI_Finalize();
}  


