

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"





#include <specx/Data/SpDataAccessMode.hpp>
#include <specx/Utils/SpUtils.hpp>
#include <specx/Random/SpPhiloxGenerator.hpp>


#include "mcglobal.hpp>



#include <specx/Data/SpDataAccessMode.hpp>
#include <specx/Utils/SpUtils.hpp>

#include <specx/Task/SpTask.hpp>
#include <specx/Legacy/SpRuntime.hpp>
#include <specx/Utils/SpTimer.hpp>
#include <specx/Utils/small_vector.hpp>
#include <specx/Utils/SpBufferDataView.hpp>
#include <specx/Utils/SpArrayView.hpp>
#include <specx/Utils/SpHeapBuffer.hpp>



#include "UTester.hpp"
#include "utestUtils.hpp"



#include <algorithm>
#include <iostream>
#include <vector>
#include <string>



using namespace std;


int main(int argc, char **argv)
{

    int NumThreads = SpUtils::DefaultNumThreads();
    int NbThreads = 2;


    if (1==1)
    {

        NbThreads = 5;

        SpRuntime runtime(NbThreads);
        std::cout<<"Nb Thread="<<runtime.getNbThreads()<<"\n";
        std::cout<<"Nb CPU Workers="<<runtime.getNbCpuWorkers()<<"\n";
        int Num=runtime.getNbThreads();
        std::atomic<int> initVal(0);

            for(int idxThread = 0 ; idxThread < Num ; ++idxThread){
                runtime.task(SpParallelWrite(initVal),
                             [&](std::atomic<int>& initValParam){
                    
                    initValParam += 1;
                    
                    //cout<<initValParam<<" "<<Num<<"\n";
                    while(initValParam != Num){
                        //cout<<"NOT OK"<<"\n";
                        cout<<initValParam<<" ";
                        usleep(10000);
                        //cout<<initValParam<<" "<<Num<<"--";
                        
                    }
                    
                    
                        usleep(10);
                    
                });
            }

            runtime.waitAllTasks();
            runtime.generateDot("Runtime20.dot",true);
            runtime.generateTrace("Runtime20.svg");

            cout<<"\n";
    }//1==1


       

    NbThreads = 10;
    if (1==1)
    {
            SpRuntime runtime(NbThreads);
            std::cout<<"Nb Thread="<<runtime.getNbThreads()<<"\n";
            std::cout<<"Nb CPU Workers="<<runtime.getNbCpuWorkers()<<"\n";

            std::promise<long int> promises[10];

            int dumbVal = 0;

            for(int idxThread = 0 ; idxThread < runtime.getNbThreads() ; ++idxThread){
                runtime.task(SpParallelWrite(dumbVal),
                             [&,idxThread](int&){

                    //usleep(10);
                    promises[idxThread].set_value(idxThread);
                    const long int res = promises[(idxThread+1)%10].get_future().get();
                    //UASSERTETRUE(res == (idxThread+1)%10);
                    
                });
            }

            runtime.waitAllTasks();

            runtime.generateDot("Runtime22.dot",true);
            runtime.generateTrace("Runtime22.svg");
    }//End 1==1
    
}




