#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <mpi.h>

#define PI 3.14159265358979323846

unsigned int reverse(unsigned int, int, int);
void fft(double complex *, double complex *, int, int, int);

int main(int argc, char ** argv)
{
  

  MPI_Init(&argc, &argv);

  int rank, size, arrSize;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double complex * arr;
    double complex * y;
    double complex * result;
    double temp;

    FILE * fptr;
    fptr = fopen(argv[1], "r");

    fscanf(fptr, "%d", &arrSize);
    arr = (double complex *) malloc(sizeof(double complex)*arrSize);
    result = (double complex *) malloc(sizeof(double complex)*arrSize);
    y = (double complex *) malloc(sizeof(double complex)*(arrSize/size));

    for(int i = 0; i < arrSize; i++) 
    {
      fscanf(fptr, "%lf", &temp);
      arr[i] = temp;
    }

    fclose(fptr);

    fft(arr, y, arrSize, rank, size);

    MPI_Gather(y, arrSize/size, MPI_C_DOUBLE_COMPLEX, result, arrSize/size, 
               MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
      for(int i = 0; i < arrSize; i++)
        printf("y[%d] = %lf + %lf*i\n", i, creal(result[i]), cimag(result[i]) );
    }

    free((void *) arr);
    free((void *) y);
    free((void *) result);

  MPI_Finalize();

  return 0;
}


void fft(double complex * x, double complex * y, int n, int rank, int size)
{

  int r = (int) ceil(log2(n));
  int workToDo = n/size;
  int start = rank*workToDo;
  int end = start + workToDo;
  double complex * R = malloc(sizeof(double complex)*workToDo);
  double complex * S = malloc(sizeof(double complex)*workToDo);
  double complex * Sk = malloc(sizeof(double complex)*workToDo);
  double complex * temp;

  for(int i = start, j = 0; j < workToDo; i++, j++)
    R[j] = x[i];

  for(int m = 0; m < r; m++)
  {
    for(int i = 0; i < workToDo; i++)
    {
      Sk[i] = S[i] = R[i];
    }
    unsigned int bit = 1 << (r - m - 1);
    unsigned int notbit = ~bit;

    int dest, src;
    int splitPoint = size / (1 << (m+1));

    if(splitPoint > 0)
    {
      if( ( rank % (splitPoint*2) ) < splitPoint)
      {
        dest = src = rank + splitPoint;
        MPI_Sendrecv(S, workToDo, MPI_C_DOUBLE_COMPLEX, dest, 0,
                     Sk, workToDo, MPI_C_DOUBLE_COMPLEX,  src, 0,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      else
      {
        dest = src = rank - splitPoint;
        MPI_Sendrecv(Sk, workToDo, MPI_C_DOUBLE_COMPLEX, dest, 0,
                   S, workToDo, MPI_C_DOUBLE_COMPLEX,  src, 0,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

    }
    else
    {
      for(int i = 0; i < workToDo; i++)
        Sk[i] = S[i];
    }

    for(int i = start, l = 0; l < workToDo; i++, l++)
    {
      int j = (i & notbit) % workToDo;
      int k = (i | bit) % workToDo;

      int expFactor = reverse(i, r, m);
      R[l] = S[j] + Sk[k] * cexp( (2*PI*I*expFactor)/n );
    }
  }

  for(int i = 0; i < workToDo; i++)
  {
    y[i] = R[i];
  }

  free((void *) R);
  free((void *) S);
  free((void *) Sk);
}


//0 1 2 3 4 5 6 7
//0 1 1 0 1 0 1 1


//x x x x x 0 1 1

//x x x x x 1 1 0

//1 1 0 x x x x x

unsigned int reverse(unsigned int x, int r, int m)
{
    unsigned int y = 0;

    x = x >> (r-m-1);

    for(int i = 0; i < m+1; i++)
    {
      y |= x & 1;
      x = x >> 1;
      if(i < m)
        y = y << 1;
    }

    y = y << (r-m-1);

    return y;
}
