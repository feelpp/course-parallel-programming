#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define NUM_MESSAGES 20

int main (int argc, char** argv)
{
  int number_of_processes;
  int my_rank;
  int mpi_error_code;
  MPI_Request req[NUM_MESSAGES];

  mpi_error_code = MPI_Init(&argc, &argv);

    mpi_error_code = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    mpi_error_code = MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

    //This is the server process.
    if(my_rank == 0)
    {
      printf("Hello from the server!\n");
     
      int i;
      char msg_to_send[100];
      for(i = 0; i < NUM_MESSAGES; i++)
      {
        sprintf(msg_to_send, "Hi from server, this is msg %d\n", i);
        mpi_error_code = MPI_Isend(msg_to_send, strlen(msg_to_send)+1, 
          MPI_CHAR, 1, i, MPI_COMM_WORLD, &req[i]);
      }

      printf("Message was sent from server!\n");
    }
    else
    {
      printf("Hello from the client!\n");

      char str[NUM_MESSAGES][100];
      MPI_Status statuses[NUM_MESSAGES];

      int i;
      int j;
      for(i = 0; i < NUM_MESSAGES; i++)
        mpi_error_code = MPI_Irecv(str[i], 100, 
          MPI_CHAR, 0, i, MPI_COMM_WORLD, &req[i]);

      for(j = 0; j < 999; j++)
        printf("I am the client and I am doing work!\n");

      MPI_Waitall(NUM_MESSAGES, req, statuses);

      for(i = 0; i < NUM_MESSAGES; i++)
      {
        printf("The message received in the client is: %s\n", str[i]);
      }

    }

  mpi_error_code = MPI_Finalize();
  return 0;
}
