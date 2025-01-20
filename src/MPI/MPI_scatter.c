#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#define N 1000000
#define X 100

int main(int argc, char ** argv)
{
  int rank, npes, i;
  int * randomInts;

  MPI_Init(&argc, &argv);

  //Get rank and number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  //Generate N random integers valued between 0 and X-1
  if(rank == 0)
  {
    randomInts = (int *) malloc(N*sizeof(int));
    for(i = 0; i < N; i++)
      randomInts[i] = rand()%X;
  }

  //Scatter the integers to each of npes processes
  int workToDo = N/npes;
  int * workArray = (int *) malloc(workToDo*sizeof(int));
  MPI_Scatter(randomInts, workToDo, MPI_INT, workArray, workToDo, MPI_INT, 0, MPI_COMM_WORLD);

  //Compute and print the average of each set of integers
  long sum = 0L;
  for(i = 0; i < workToDo; i++)
    sum += workArray[i];

  //Take care of extra integers
  if(rank == 0)
    if(workToDo * npes < N)
      for(i = workToDo*npes; i < N; i++)
        sum += randomInts[i];

  printf("The average of the integers in process %d is %lf\n", rank, sum/(double)workToDo);

  free((void *) workArray);

  if(rank == 0)
  {
    //Error check
    int j;
    for(j = 0; j < npes; j++)
    {
      sum = 0;
      for(i = j*workToDo; i < j*workToDo + workToDo; i++)
        sum += randomInts[i];

      if(j == 0)
        if(workToDo * npes < N)
          for(i = workToDo*npes; i < N; i++)
            sum += randomInts[i];

      printf("The actual average from the integers in process %d is %lf\n", j, sum/(double)workToDo);
    }
    free((void *)randomInts);
  }

  MPI_Finalize();
  return 0;
}
