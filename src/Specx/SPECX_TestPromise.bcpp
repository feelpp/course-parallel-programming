

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

        cout<<"===================================\n";

        NumThreads=2;
        SpRuntime My_Runtime3(NumThreads); 

        //SpRuntime<SpSpeculativeModel::SP_MODEL_2> My_Runtime3;
        //cout<<":"<<My_Runtime3.getValue()<<"\n";

        My_Runtime3.setSpeculationTest(
            [](const int /*inNbReadyTasks*/,
            const SpProbability& /*inProbability*/) -> bool
            {
                // Always speculate
                //usleep(50000);
                return true;
            } 
        );

        int val = 0;
        std::promise<int> promise3;
        //la promesse que doit remplir le thread annexe

         My_Runtime3.task(SpRead(val)
            , 
            [&promise3](const int& /*valParam*/)
            {
                usleep(100);
                promise3.get_future().get();
                //Retourne un objet future qui a le même état asynchrone associé que cet objet promise.
            }
        ).setTaskName("First task");

       
        
        for(int idx = 0; idx < 1; idx++) {
          My_Runtime3.task(SpWrite(val), 
              [](int& valParam)  
              {
                cout<<"CTRL Val in certain task="<<valParam<<"\n";
                usleep(500);
              }
          ).setTaskName("Certain task -- " + std::to_string(idx));
        }

        
        
        const int nbUncertainTasks = 6;

        for(int idx = 0 ; idx < nbUncertainTasks ; ++idx){
            My_Runtime3.task(SpPotentialWrite(val), 
               [](int& valParam) -> bool
                {
                   cout<<"CTRL Val in uncertain task="<<valParam<<"\n";
                   usleep(1000);
                   return true;
                }
            ).setTaskName("Uncertain task -- " + std::to_string(idx));
        }
        

        /*
        for(int idx = 2; idx < 4; idx++) {
          My_Runtime3.task(SpWrite(val), 
              [](int& valParam)  
              {
                usleep(1500);
              }
           ).setTaskName("Certain task -- " + std::to_string(idx));
        }
        */
        
        My_Runtime3.task(SpWrite(val), 
              []([[maybe_unused]] int& valParam)
              {
                usleep(2000);
              }
        ).setTaskName("Last-task");

        promise3.set_value(0);
        //L'opération se comporte comme si set_value , set_exception , 
        //set_value_at_thread_exit et set_exception_at_thread_exit acquéraient 
        //un seul mutex associé à l'objet de promesse lors de la mise à jour de 
        //l'objet de promesse.
    
        //Une exception est levée s'il n'y a pas d'état partagé ou si 
        //l'état partagé stocke déjà une valeur ou une exception.
        //Les appels à cette fonction n'introduisent pas de courses de 
        //données avec les appels à get_future (ils n'ont donc pas besoin 
        //de se synchroniser les uns avec les autres).



        My_Runtime3.waitAllTasks();

        My_Runtime3.generateDot("RuntimePromise.dot",true);
        My_Runtime3.generateTrace("RuntimePromose.svg");


    }


}



 
#pragma GCC diagnostic pop

