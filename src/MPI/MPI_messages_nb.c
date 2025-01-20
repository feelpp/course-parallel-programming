#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main (int argc, char** argv)
{
  int number_of_processes;
  int my_rank;
  int mpi_error_code;
  MPI_Request req;

  mpi_error_code = MPI_Init(&argc, &argv);

    mpi_error_code = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    mpi_error_code = MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

    //This is the server process.
    if(my_rank == 0)
    {
      printf("Hello from the server!\n");
     
      mpi_error_code = MPI_Isend("Hi from server!", 16, MPI_CHAR, 1, 0, 
        MPI_COMM_WORLD, &req);

      printf("Message was sent from server!\n");
    }
    else
    {
      printf("Hello from the client!\n");

      MPI_Status stat;
      char str[20];
      mpi_error_code = MPI_Irecv(str, 20, MPI_CHAR, 0, 0, MPI_COMM_WORLD, 
          &req);

      mpi_error_code = MPI_Wait(&req, &stat);

      printf("The message received in the client is: %s\n", str);
    }

  mpi_error_code = MPI_Finalize();
  return 0;
}
