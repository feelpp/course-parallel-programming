#include <mpi.h>
#include <stdio.h>
#include <string.h>
int main(int argc, char ** argv)
{
  int rank, size, tag = 792;
  char * str = "Hi";
  char recvStr[20];

  MPI_Status myStatus;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Request requestSend, requestRecv;

  if(rank > 0 && rank < size-1)
  {
//    MPI_Sendrecv(str, strlen(str)+1, MPI_CHAR, rank+1, tag,
//                 recvStr, 20, MPI_CHAR, rank-1, tag, MPI_COMM_WORLD,
//                 &myStatus);

    MPI_Isend(str, strlen(str)+1, MPI_CHAR, rank+1, tag, MPI_COMM_WORLD, 
        &requestSend);

    MPI_Irecv(recvStr, 20, MPI_CHAR, rank-1, tag, MPI_COMM_WORLD,
                 &requestRecv);

    int flag = 0;
    while( flag == 0 ) 
    {
      printf("Doing stuff\n");
      MPI_Test(&requestRecv, &flag, &myStatus);
    }
    MPI_Wait(&requestRecv, &myStatus);

    printf("Received %s in process rank %d\n", recvStr, rank);
    printf("The status values in rank %d are %d, %d, and %d\n", 
        rank, myStatus.MPI_SOURCE, myStatus.MPI_TAG, myStatus.MPI_ERROR);
  }
  else if(rank == 0)
  {
//    MPI_Sendrecv(str, strlen(str)+1, MPI_CHAR, rank+1, tag,
 //                recvStr, 20, MPI_CHAR, size-1, tag, MPI_COMM_WORLD,
 //                &myStatus);

    MPI_Isend(str, strlen(str)+1, MPI_CHAR, rank+1, tag, MPI_COMM_WORLD, 
        &requestSend);

    MPI_Irecv(recvStr, 20, MPI_CHAR, size-1, tag, MPI_COMM_WORLD,
                 &requestRecv);

    MPI_Wait(&requestRecv, &myStatus);

    printf("Received %s in process rank %d\n", recvStr, rank);
    printf("The status values in rank %d are %d, %d, and %d\n", 
        rank, myStatus.MPI_SOURCE, myStatus.MPI_TAG, myStatus.MPI_ERROR);
  }
  else
  {
//    MPI_Sendrecv(str, strlen(str)+1, MPI_CHAR, 0, tag,
//                 recvStr, 20, MPI_CHAR, rank-1, tag, MPI_COMM_WORLD,
//                 &myStatus);

    MPI_Isend(str, strlen(str)+1, MPI_CHAR, 0, tag, MPI_COMM_WORLD, 
        &requestSend);

    MPI_Irecv(recvStr, 20, MPI_CHAR, rank-1, tag, MPI_COMM_WORLD,
                 &requestRecv);

    MPI_Wait(&requestRecv, &myStatus);

    printf("Received %s in process rank %d\n", recvStr, rank);
    printf("The status values in rank %d are %d, %d, and %d\n", 
        rank, myStatus.MPI_SOURCE, myStatus.MPI_TAG, myStatus.MPI_ERROR);
  }

  MPI_Finalize();

  return 0;
}
