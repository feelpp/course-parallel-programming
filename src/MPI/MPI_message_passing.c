/* A program to demonstrate simple message passing with
 * MPI.
 *
 * This program sends a simple message "Sending and receiving is fun!"
 * from the process with rank 0 to the process with rank 1
 * Processes with ranks greater than 1 will print a message indicating they
 * do no work.
 *
 * compile with: mpicc message_passing.c -o message_passing.exe
 * run with: mpirun -np 3 message_passing.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char ** argv)
{
  int my_rank;
  int number_of_processes;
  int mpi_error_code;
  char * my_message = "Sending and receiving is fun!";
  const int MAX_MESSAGE_SIZE = 100;
  char received_message[MAX_MESSAGE_SIZE];

  /* Initialize the MPI Framework */
  mpi_error_code = MPI_Init(&argc, &argv);

    /* Get my rank */
    mpi_error_code = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Get the number of processes */
    mpi_error_code = MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

    /* Do work! */
    if(my_rank == 0)
    {
      printf("Sending message from process %d to process %d\n", my_rank, 1);

      /* We need to send 
         1. The message, 
         2. The length of the message,
         3. The type of the message, 
         4. The rank of the receiving process,
         5. A message ID called the tag (arbitrary), and 
         6. The communicator
       */
      mpi_error_code = MPI_Send(my_message, strlen(my_message)+1,
        MPI_CHAR, 1, 789, MPI_COMM_WORLD);
    }
    else if(my_rank == 1)
    {
      /* The first parameter is what we receive - the data. The others are,
         2. the maximum message size
         3. the type of the message
         4. the rank of the process to receive data from
         5. the tag
         6. the communicator
         7. the message status
       */

      printf("Receiving message from process 0\n");
      mpi_error_code = MPI_Recv(received_message, MAX_MESSAGE_SIZE, MPI_CHAR,
        0, 789, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("The message is: %s\n", received_message);

    }
    else
    {
      printf("My process number is %d and I do not do anything\n", my_rank);
    }

  /* Exit MPI */
  mpi_error_code = MPI_Finalize();

  return 0;
}
