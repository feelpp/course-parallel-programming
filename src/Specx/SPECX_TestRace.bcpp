

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


    std::array<unsigned int,3> SleepTimes{0, 500,1000};

    for (const int& value : SleepTimes) { cout << value << "  "; }; cout << "\n";



    //Small Horse Race

    string ch0 = "Small Horse Race";
    string chNb = to_string(4);
    ch0=ch0+chNb;

    NumThreads = SpUtils::DefaultNumThreads();
    std::cout<<"Nb Thread="<<NumThreads<<"\n";
    NumThreads = 2;
    std::cout<<"Nb Thread="<<NumThreads<<"\n";

    SpTimer T0;

    for(auto SleepTime : SleepTimes){
        
        SpRuntime My_Runtime(NumThreads); 

        My_Runtime.setSpeculationTest(
            [](const int, const SpProbability&) -> bool
            {
                return true;
            });

        const int arraySize = 6;
        //int val[arraySize] = {0};
        int val[arraySize];

        for (int i=0; i<arraySize; ++i)
        {
            val[i]=0;
        }

        cout<<"Array : ";
        for (const int& value : val) { cout << value << "  "; }; cout << "\n";


        UTestRaceChecker counterAccess;
        string ChInfo,ChInfoPlus;
        ChInfo = "Runtime Array Process";

        My_Runtime.task(SpReadArray(val,SpArrayView(arraySize)), 
        [](SpArrayAccessor<const int>& /*valParam*/){}).setTaskName("Small Horse Race");
 
            for(int idx = 0 ; idx < arraySize ; ++idx){
                ChInfoPlus = ChInfo+to_string(idx);

                My_Runtime.task(SpWrite(val[idx]),
                                      SpReadArray(val,SpArrayView(arraySize).removeItem(idx)),
                                      [SleepTime,idx,&counterAccess]
                                      (int& valParam, const SpArrayAccessor<const int>& valArray) -> bool {
                    {
                        counterAccess.lock();
                        counterAccess.addWrite(&valParam);
                        for(int idxTest = 0 ; idxTest < valArray.getSize() ; ++idxTest){
                            counterAccess.addRead(&valArray.getAt(idxTest));
                        }
                        counterAccess.unlock();
                    }

                    string ChNumidx=to_string(idx);

                    cout<<"TimeSleep="<<SleepTime<<" Idx="<<idx;
                    cout<<" valParam1="<<valParam;

                    if (1==1) {
                         if(idx == 1){
                            valParam += 2;
                        }
                        if(idx == 3){
                            valParam += 1;
                        }
                        if(idx == 5){
                            valParam += 10;
                        }
                    }
                    cout<<" valParam2="<<valParam<<"\n";

                    usleep(SleepTime);

                    {
                        counterAccess.lock();
                        counterAccess.releaseWrite(&valParam);
                        for(int idxTest = 0 ; idxTest < valArray.getSize() ; ++idxTest){
                            counterAccess.releaseRead(&valArray.getAt(idxTest));
                        }
                        counterAccess.unlock();
                    }

                    return (idx == 3 || idx == 5);
                }).setTaskName(ChInfoPlus); //END runtime;
            }
            My_Runtime.waitAllTasks();
            My_Runtime.stopAllThreads();

            My_Runtime.generateDot("TestRaceThread.dot",true);

            My_Runtime.generateTrace("TestRaceThread.svg");

        cout<<"Array : ";
        for (const int& value : val) { cout << value << "  "; }; cout << "\n";
        

    }//END FOR


    T0.stop();
    cout<<"Delta Time="<<T0.getElapsed()<<"\n";

    SpPhiloxGenerator<double> randGen(0);


}



 
#pragma GCC diagnostic pop

