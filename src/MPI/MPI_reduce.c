#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define N 1000000000

int main(int argc, char ** argv)
{
  int rank, size, i, j;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  long localSum = 0L;
  long totalSum = 0L;
  int workToDo = N/size;
  int * arr;
  int start = workToDo*rank;
  int end = start+workToDo;
  
  arr = (int *) malloc(workToDo*sizeof(int));
  for( i = start, j= 0; i < end; i++,j++)
    arr[j] = i+1;

  for( i = 0; i < workToDo; i++)
    localSum += arr[i];
  
  MPI_Reduce(&localSum, &totalSum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  free((void *) arr);
  if(rank == 0)
    printf("The sum is x = %ld\n", totalSum);

  MPI_Finalize();
  return 0;
}
