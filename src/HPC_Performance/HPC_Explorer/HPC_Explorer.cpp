#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <chrono>
#include <math.h>

//#include <sys/times.h>
//#include <sys/vtimes.h>

//#include </usr/include/sys/times.h>

//#include </usr/include/sys/vtimes.h>

#include <features.h>

#define ARRAY_LEN 10000


int main(int argc, char** argv) {

    //arguments NumOption QSee NbPingPong NameFile
    //mpirun -np 96 ./HPC_Explorer 1 0 1 Mat1.csv
    int NumOption=0;
    bool QSee=false;
    int NbPingPong=1;
    

    if (argc < 3) 
    {

    }
    else 
    {
        NumOption=atoi(argv[1]);
        QSee=atoi(argv[2]);
        NbPingPong=atoi(argv[3]);

    }	

    if (QSee) { printf("Nb PingPong=%i\n",NbPingPong); }

 
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Get_processor_name(processor_name, &name_len);

    MPI_Barrier(MPI_COMM_WORLD); 


    if (NumOption==1) {
    
    
  
    

    
    double t1 = MPI_Wtime();
    double MinTime,MaxTime,AvgTime;

    


    MPI_Win win;
    MPI_Aint size;
    float *baseptr=new float;

    if (world_rank == 0)
    {
      size = ARRAY_LEN * sizeof(float);
      MPI_Win_allocate_shared(size, sizeof(int), MPI_INFO_NULL,MPI_COMM_WORLD, &baseptr, &win);
    }
    else
    {
      int disp_unit;
      MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL,MPI_COMM_WORLD, &baseptr, &win);
      MPI_Win_shared_query(win, 0, &size, &disp_unit, &baseptr);
    }


    if (QSee) {  
      int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
      printf("NumCPU %d\n",numCPU);
      // If Error : There are not enough slots available in the system to satisfy the 120
      // slots that were requested by the application:
      printf("Rank %d, baseptr = %p  world_size = %d\n", world_rank, baseptr,world_size);

      

    }

    float *arr = baseptr;
    
    /*
    if (world_rank != 0)
    {
        arr[world_rank] = world_rank+0.1;
    }
    */

    double t2 = MPI_Wtime(); /*get time just before work section */


    bool QSave=true;
    //QSave=false;


    //BEGIN::PART OF MY WORK

    //=============================================================================================================
    

    
   
    int token;
    int NbData=1;
    int Data[1];
    double t0,dt,dt0;
    

    MPI_Status Status;

    //for (int i=0;i<world_size;i++)
    for (int i=0;i<world_size;i++)
    {
      if (i!=world_rank)
      {
        token=i+world_rank;
        t0 = MPI_Wtime();
        int mask=0;
        double MinTime0,MaxTime0,AvgTime0;
        //printf("Process %d send for process %d : %i\n", world_rank,i,token);
        double Sum=0.0;
        
        //printf("BLOCK 1: %i %i\n",world_rank,i); 
        for (int k=0;k<NbPingPong;k++) {
          MPI_Send(&Data, NbData, MPI_INT,i,mask,MPI_COMM_WORLD);
          MPI_Recv(&Data, NbData, MPI_INT,i,mask, MPI_COMM_WORLD,&Status);
          dt0 = MPI_Wtime()-t0;
          Sum=Sum+dt0;
        }
        //printf("BLOCK 2: %i %i\n",world_rank,i); 
        
        dt = MPI_Wtime()-t0;
        Sum=Sum/NbPingPong;

        arr[world_rank*world_size+i]=Sum;

        if (QSee) { 
          printf("Process %d send for process %d : %i : %1.10f %1.10f\n", world_rank,i,token,dt,Sum); 
          //printf("Statut Source = %i\n",Status.MPI_SOURCE);
          //printf("Statut Tage = %i\n",Status.MPI_TAG);
        }
        
      }
    } 








    //END: PART OF MY WORK
    //=============================================================================================================





    double MyTime=t2-t1; /*get time just after work section*/

    /*compute max, min, and average timing statistics*/

    if (1==0) {
        MPI_Reduce(&MyTime, &MaxTime, 1, MPI_DOUBLE, MPI_MAX, 0,MPI_COMM_WORLD);
        MPI_Reduce(&MyTime, &MinTime, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
        MPI_Reduce(&MyTime, &AvgTime, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    }



    //printf("Hello world from processor [%s], rank [%d] out of [%d] processors Time=[%1.10f]  \n",processor_name, world_rank, world_size,MyTime);


    if ((world_rank==0) && (QSee))
    {

        AvgTime /= world_size;
        printf("Min: %lf  Max: %lf  Avg:  %lf\n", MinTime, MaxTime,AvgTime);
    }


    MPI_Barrier(MPI_COMM_WORLD);




    //SAVE DATA
    


    if ((world_rank==0) && (1==1))
    { 
        if (QSave) {
            FILE* file = NULL;
            //file = fopen("Mat2.csv", "w");
            file = fopen(argv[4], "w");


            for (int i=0;i<world_size;i++)
            {
                for (int j=0;j<world_size;j++)
                {

                    fprintf(file, "%d,%d,%f\n",i,j,arr[i*world_size+j]);
                }
            }
            
            fclose(file);	
        }
    }

    MPI_Win_free(&win);

   

    if ((world_rank==0) && (1==1) && (QSee))
    { 

      double MatDist[1000][1000];   
      FILE* file = NULL;
      //file = fopen("Mat2.csv", "r");
      file = fopen(argv[4], "r");

      int i,j;
      float value;
      while(!feof(file))
      {
        int Value;
        Value=fscanf (file, "%i,%i,%f", &i,&j,&value);
        MatDist[i][j]=value;
      }
      fclose(file);	



      printf("\n");

      for (i=0;i<world_size;i++)
      {
        for (j=0;j<world_size;j++)
        {
          printf("[%i,%i]=%1.10f\t",i,j,MatDist[i][j]);
        }
        printf("\n");
      }
    }
   
    //MPI_Finalize();
    }

    if ((NumOption==2) && (world_rank==0)) {
        int Value;
        Value=std::system("lscpu>InfoSystemCPU.txt");
        Value=std::system("lshw -C display>InfoSystemGPU.txt");
    }


     MPI_Finalize();
}


