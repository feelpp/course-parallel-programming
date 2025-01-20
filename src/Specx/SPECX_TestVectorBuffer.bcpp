

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"



#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Random/SpPhiloxGenerator.hpp"


#include "mcglobal.hpp"



#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"
#include "Utils/SpTimer.hpp"
#include "Utils/small_vector.hpp"
#include "Utils/SpBufferDataView.hpp"
#include "Utils/SpArrayView.hpp"
#include "Utils/SpHeapBuffer.hpp"


#include "UTester.hpp"
#include "utestUtils.hpp"



#include <algorithm>
#include <iostream>
#include <vector>
#include <string>



using namespace std;

template <typename C> void print(const string& s, const C& c) {
    cout << s;
    for (const auto& e : c) { cout << e << " "; }
    cout << endl;
}

void fillVector(vector<int>& v)
{
    // A local static variable.
    static int nextValue = 1;

    // The lambda expression that appears in the following call to
    // the generate function modifies and uses the local static
    // variable nextValue.
    generate(v.begin(), v.end(), [] { return nextValue++; });
    //WARNING: this isn't thread-safe and is shown for illustration only
}


int gen()
{
    static int i = 0;
    return ++i;
}

class MyFunction001
{
public:
    // The required constructor for this example.
    explicit MyFunction001(int& evenCount)
        : m_evenCount(evenCount) { }

    // The function-call operator prints whether the number is
    // even or odd. If the number is even, this method updates
    // the counter.
    void operator()(int n) const {
        cout << n;

        if (n % 2 == 0) {
            cout << " is even " << endl;
            ++m_evenCount;
        } else {
            cout << " is odd " << endl;
        }
    }

private:
    // Default assignment operator to silence warning C4512.
    MyFunction001& operator=(const MyFunction001&);

    int& m_evenCount; // the number of even variables in the vector.
};




int main(int argc, char **argv)
{

    int NumThreads = SpUtils::DefaultNumThreads();

    const int NbLoops = 3;
    const int NbInnerLoops = 2;
    static_assert(NbInnerLoops <= NbLoops, "Nb inner loops cannot be greater than Nb Loops");

    const int NbDomains = 5;
    const int NbParticlesPerDomain = 10;
    const double BoxWidth = 1;
    const double displacement = 0.00001;

    const int NbReplicas = 5;
    std::array<double,NbReplicas> betas;
    std::array<double,NbReplicas> temperatures;

    const double MinTemperature = 0.5;
    const double MaxTemperature = 1.5;

    for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
        betas[idxReplica] = 1;
        temperatures[idxReplica] = MinTemperature + double(idxReplica)*(MaxTemperature-MinTemperature)/double(NbReplicas-1);
    }

    std::array<size_t,NbReplicas> cptGeneratedSeq = {0};

    ///////////////////////////////////////////////////////////////////////////
    /// With a possible failure move
    ///////////////////////////////////////////////////////////////////////////

    const bool runSeqMove = false;
    const bool runTaskMove = false;
    const bool runSpecMove = false;

    std::array<double,NbReplicas> energySeq = {0};

    const int MaxIterationToMove = 5;
    const double collisionLimit = 0.00001;


    if (1==1)
    {
        const int initVal = 1;
        int writeVal = 0;

        NumThreads=6;
        SpRuntime My_Runtime2(NumThreads); 

        small_vector<int> vs;
        std::cout << "std::allocator<int>:"                         << '\n'
            << "  sizeof (vs):     " << sizeof (vs)           << '\n'
            << "  Maximum size:    " << vs.max_size ()        << "\n\n";

        SpHeapBuffer<small_vector<int>> heapBuffer;

        int valueN=0;
        int valueM=0;

        for(int idx = 0 ; idx < 6 ; ++idx){
            auto vectorBuffer = heapBuffer.getNewBuffer();

            
            //On écrit dans 6 buffers
            My_Runtime2.task(SpWrite(vectorBuffer.getDataDep()),
            [&](SpDataBuffer<small_vector<int>> ) mutable
            {
                valueN=idx;  
                usleep(1000);
            }
            ).setTaskName("Write Vector Buffer");
            
            valueM=valueN;
            //Puis on lit 2 fois
            for(int idxSub = 0 ; idxSub < 2 ; ++idxSub){
                My_Runtime2.task(SpRead(vectorBuffer.getDataDep()),
                    [=](const SpDataBuffer<small_vector<int>>)
                    {
                        
                        cout<<"idx="<<idx<<" idSub="<<idxSub<<" valueN="<<valueN<<" valueM="<<valueM<<"\n";
                        usleep(2000);
                       
                    }
                    ).setTaskName("Read Vector Buffer");;
            }    
        }
        My_Runtime2.waitAllTasks();
        My_Runtime2.stopAllThreads();
        My_Runtime2.generateDot("RuntimeVectorBuffer.dot",true);
        My_Runtime2.generateTrace("RuntimeVectorBuffer.svg");
    }




}



 
#pragma GCC diagnostic pop

