

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"






#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Random/SpPhiloxGenerator.hpp"


#include "mcglobal.hpp"


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



    SpRuntime runtime(NumThreads); 
    std::cout<<"Nb Thread="<<NumThreads<<"\n";
    SpTimer timer;


    int v=1; 
    int r=0;

    std::cout<<"v="<<v<<" r="<<r<<"\n";

    runtime.task(SpRead(v),SpWrite(r),
    [] (const int &,int r1) {
        r1=10;
    } ).setTaskName("First Runtime");

    runtime.task(SpRead(v),
    [] (const int &) {
        
    } ).setTaskName("Second Runtime");

    std::cout<<"v="<<v<<" r="<<r<<"\n";

    //=============================================
    int p;
    auto Area=[=] (int a,int b)->int
    {
        return p*a*b;
    };
    p=10;
    std::cout<<"Result1="<<Area(2,3)<<"\n";

    //=============================================
    int p2=100;
    auto Area2=[&] (int a,int b)
     {
        p2=50;
        return p2*a*b;
    };
    std::cout<<"Result2="<<Area2(2,3)<<"\n";

    //=============================================

    
    auto Area3=[] (int a,int b)
     {
        return a*b;
    };

    std::cout<<"Result3="<<Area3(2,3)<<"\n";

    //=============================================
    int v1;
    int g=3;
    auto Area4=[&,g] (int a,int b)
     {
        return v1+a*b;
    };
    v1=1;
    std::cout<<"Result3="<<Area3(2,3)<<"\n";


    int x = 1;
    auto valueLambda = [=]() { std::cout << x << std::endl; };
    auto refLambda = [&]() { std::cout << x << std::endl; };
    x = 13;
    valueLambda();
    refLambda();


    //=============================================
    int m = 0;
    int n = 0;
    [&, n] (int a) mutable { m = ++n + a; }(4);
    std::cout <<"m="<<m << std::endl <<"n="<< n << std::endl;

    m = 0;
    n = 0;
    auto Fonc1=[&,n] (int a) mutable { m = n++ + a; n=20; };
    Fonc1(6);
    std::cout <<"m="<<m << std::endl <<"n="<< n << std::endl;


    //m = 0;
    //n = 0;
    //auto Fonc2=[&,n] (int a) { m = n++ + a; n=20; }; // Error no mutable
    //Fonc2(6);
    //std::cout <<"m="<<m << std::endl <<"n="<< n << std::endl;



    const int elementCount = 9;

    // Create a vector object with each element set to 1.
    vector<int> vvv0 {1,2,3,4,5,6,7,8,9,10,11};
    cout << "v0: "<< vvv0.data() << endl;
    for (const int& value : vvv0) { cout << value << "  "; }
    cout << "\n";

    cout <<vvv0[4]<<"\n";
    cout <<vvv0.at(4)<<"\n"; //Better to used give exception if up range 

    cout << "\n";

    vector<int> vvv1(10);
    for (const int& value : vvv1) { cout << value << "  "; }
    cout << "\n";
  
    // using std::generate
    std::generate(vvv1.begin(), vvv1.end(), gen);
    for (const int& value : vvv1) { cout << value << "  "; }
    cout << "\n";
    vector<int>::iterator i1;
    for (i1 = vvv1.begin(); i1 != vvv1.end(); ++i1) {
        cout << *i1 << " ";
    }
    cout << "\n";




    cout << "\n";
    vector<int> vvv(elementCount, 1);


    cout << "v: "<< vvv.data() << endl;
    for (const int& value : vvv) { cout << value << "  "; }
    cout << "\n";


    // These variables hold the previous two elements of the vector.
    int xx = 1;
    int yy = 1;

    // Sets each element in the vector to the sum of the
    // previous two elements.
    generate_n(vvv.begin() + 2,
        elementCount - 2,
        [=]() mutable throw() -> int { // lambda is the 3rd parameter
        // Generate current value.
        int n = xx + yy;
        // Update previous two values.
        xx = yy;
        yy = n;
        return n;
    });
    print("vector v after call to generate_n() with lambda: ", vvv);

    // Print the local variables x and y.
    // The values of x and y hold their initial values because
    // they are captured by value.
    cout << "x: " << xx << " y: " << yy << endl;

    // Fill the vector with a sequence of numbers
    fillVector(vvv);
    print("vector v after 1st call to fillVector(): ", vvv);
    // Fill the vector with the next sequence of numbers
    fillVector(vvv);
    print("vector v after 2nd call to fillVector(): ", vvv);


    cout << endl;


    //Assign the lambda expression that adds two numbers to an auto variable.
    auto f1 = [](int x, int y) { return x + y; };

    cout << f1(2, 3) << endl;

    // Assign the same lambda expression to a function object.
    function<int(int, int)> f2 = [](int x, int y) { return x + y; };

    cout << f2(3, 4) << endl;

    cout << endl;

    /*int yw = 32;
    auto answer = [yw]() constexpr
    {
        int xw = 10;
        return yw + xw;
    };

    constexpr int Increment(int n)
    {
        return [n] { return n + 1; }();
    }
    */



   // Create a vector object that contains 9 elements.
    vector<int> vn;
    for (int i = 1; i < 10; ++i) {
        vn.push_back(i);
    }

    // Count the number of even numbers in the vector by
    // using the for_each function and a function object.
    int evenCount = 0;
    for_each(vn.begin(), vn.end(), MyFunction001(evenCount));

    // Print the count of even numbers to the console.
    cout << "There are " << evenCount
        << " even numbers in the vector." << endl;

    cout << endl;

    const int NbTab=10;
    double Tab[NbTab]={0};
    for (int i=0; i<NbTab; ++i)
    {
        Tab[i]=0.01*double(i);
    }
    UTestRaceChecker counterAccess;


    runtime.task(SpReadArray(Tab,SpArrayView(NbTab)), 
        [](SpArrayAccessor<const double>& ){}
    ).setTaskName("Runtime read Array");


    for (const double& value : Tab) { cout << value << "  "; }; cout << "\n";



    for(int idx = 0 ; idx < NbTab ; ++idx){
                runtime.task(SpWrite(Tab[idx]),
                             SpReadArray(Tab,SpArrayView(NbTab).removeItem(idx)),
                                     
                [idx](double& valParam, const SpArrayAccessor<const double>& valArray) -> bool
                {
                    {
                       
                            if(idx == 3){
                                valParam += 1;
                            }
                            if(idx == 5){
                                valParam += 10;
                            }
                            usleep(1000);

                            cout<<"valParam="<<valParam<<"\n";


                        return (idx == 3 || idx == 5); 
                    } 
                }
                ).setTaskName("Runtime Array Process"); //END runtime
            } //END FOR


    for (const double& value : Tab) { cout << value << "  "; }; cout << "\n";

    //specx part




    runtime.waitAllTasks();

    runtime.stopAllThreads();

    timer.stop();

    runtime.generateDot("Test.dot",true);

    runtime.generateTrace("Test.svg");
  

    //std::array<unsigned int,2> SleepTimes{0, 500};

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

    //cout<<"Value SpPhiloGenerator="<<randGen<<"\n";


    //**********************************************************

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
        My_Runtime2.generateDot("Runtime2.dot",true);
        My_Runtime2.generateTrace("Runtime2.svg");
    }


  //**********************************************************

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

        My_Runtime3.generateDot("Runtime3.dot",true);
        My_Runtime3.generateTrace("Runtime3.svg");


    }



    if (1==1)
    {

        SpDebug::Controller.isEnable();

        cout<<"===================================\n";
            const int NbThreads = 2;
            SpRuntime My_Runtime4(NbThreads);

            int readVal = 0;
            std::promise<int> promise0;
            std::promise<int> promise1;
            std::promise<int> promise2;

            My_Runtime4.task(SpRead(readVal),
                [&](const int& /*readVal*/)
                {
                   promise0.get_future().get();
                   usleep(2000);
                }).setTaskName("Promise0");

            My_Runtime4.task(SpRead(readVal),
                [&](const int& /*readVal*/)
                {
                    promise1.get_future().get();
                    usleep(2000);
                }).setTaskName("Promise1");

            My_Runtime4.task(SpRead(readVal),
                [&](const int& /*readVal*/)
                {
                    promise2.get_future().get();
                    usleep(2000);
                }).setTaskName("Promise2");

            My_Runtime4.waitRemain(3);

            promise0.set_value(0);

            //My_Runtime4

            My_Runtime4.waitRemain(2);

            promise1.set_value(0);

            My_Runtime4.waitRemain(1);

            promise2.set_value(0);

            My_Runtime4.waitAllTasks();

            My_Runtime4.generateDot("Runtime4.dot",true);
            My_Runtime4.generateTrace("Runtime4.svg");
        }

    if (1==1)
    {



    


        {

        
        const int NbTasks = 10;
        const int NbThreads = 10;

        std::promise<int> promises[NbTasks];
        //Fix Taches Pronises
        SpRuntime My_Runtime5(NbThreads);

        std::cout<<"Nb Thread="<<My_Runtime5.getNbThreads()<<"\n";
        std::cout<<"Nb CPU Workers="<<My_Runtime5.getNbCpuWorkers()<<"\n";

            int readVal = 0;

            for(int idxTask = 0 ; idxTask < NbTasks ; ++idxTask){
                My_Runtime5.task(SpRead(readVal),
                    [&,idxTask](const int& /*readVal*/)
                    {
                        promises[idxTask].get_future().get();
                        usleep(1000);
                        //usleep(100*NbTasks);
                    });
            }

            for(int idxTask = NbTasks ; idxTask < 2*NbTasks ; ++idxTask){
                My_Runtime5.waitRemain(idxTask); //mutexFinishedTasks
                
            }


            //for(int idxTask = 0 ; idxTask < NbTasks ; ++idxTask){
            //    My_Runtime5.waitRemain(idxTask); //mutexFinishedTasks
            //}

            
            for(int idxTaskPromise = 0 ; idxTaskPromise < NbTasks ; ++idxTaskPromise)
            {
                promises[idxTaskPromise].set_value(0);
                //Une exception est levée s'il n'y a pas d'état partagé ou si 
                //l'état partagé stocke déjà une valeur ou une exception.

                //usleep(2000);
                for(int idxTask = NbTasks-idxTaskPromise-1 ; idxTask < 2*NbTasks ; ++idxTask)
                {
                    My_Runtime5.waitRemain(idxTask);
                }
            }
            

            My_Runtime5.waitAllTasks();

            My_Runtime5.generateDot("Runtime5.dot",true);
            My_Runtime5.generateTrace("Runtime5.svg");
        }


            const int NbThreads = 10;
            SpRuntime My_Runtime6(NbThreads);

            int readVal = 0;
            std::promise<int> promise0;
            std::promise<int> promise1;


            My_Runtime6.task(SpRead(readVal),
                [&](const int& /*readVal*/)
                {
                   promise0.get_future().get();
                   usleep(2000);
                }).setTaskName("Promise0");

            My_Runtime6.task(SpRead(readVal),
                [&](const int& /*readVal*/)
                {
                    promise1.get_future().get();
                    usleep(2000);
                }).setTaskName("Promise1");

 
            My_Runtime6.waitRemain(3);

            promise0.set_value(0);
            My_Runtime6.waitRemain(2);

            promise1.set_value(0);



            My_Runtime6.waitAllTasks();

            My_Runtime6.generateDot("Runtime6.dot",true);
            My_Runtime6.generateTrace("Runtime6.svg");

    }


    if (1==0)
    {

        cout<<"===================================\n";



            const int NbThreads = 2;
            SpRuntime My_Runtime5(NbThreads);

            int readVal = 0;
            

            My_Runtime5.task(SpRead(readVal),
                [&](const int& /*readVal*/)
                {
                   
                   usleep(2000);
                }).setTaskName("Promise0");

            My_Runtime5.task(SpRead(readVal),
                [&](const int& /*readVal*/)
                {
                    
                    usleep(2000);
                }).setTaskName("Promise1");


            My_Runtime5.waitRemain(2);

            My_Runtime5.waitAllTasks();

            My_Runtime5.generateDot("Runtime5.dot",true);
            My_Runtime5.generateTrace("Runtime5.svg");
        }


   if (1==1)
    {        
        const int NBB=500;
        const int NbTasks = NBB;
        const int NbThreads = NBB;

        
        double Tab[NbTasks]={0};
        for (int i=0; i<NbTasks; ++i)
        {
            Tab[i]=0.01*double(i);
        }
        UTestRaceChecker counterAccess;

        std::promise<int> promises[NbTasks];
        //Fix Taches Pronises
        SpRuntime My_Runtime7(NbThreads);

        std::cout<<"Nb Thread="<<My_Runtime7.getNbThreads()<<"\n";
        std::cout<<"Nb CPU Workers="<<My_Runtime7.getNbCpuWorkers()<<"\n";

            int readVal = 0;

            for(int idxTask = 0 ; idxTask < NbTasks ; ++idxTask){
                My_Runtime7.task(
                    SpRead(readVal),
                    SpReadArray(Tab,SpArrayView(NbTab))
                    ,
                    [&,idxTask](
                        const int& /*readVal*/,
                        SpArrayAccessor<const double>&)
                    {
                        promises[idxTask].get_future().get();
                        usleep(1000);
                        
                    }).setTaskName("Runtime Array Process"+to_string(idxTask));

                
                
                
            }

            for (const double& value : Tab) { cout << value << "  "; }; cout << "\n";


            //for(int idxTask = 0 ; idxTask < NbTasks ; ++idxTask)

            //My_Runtime7.task(SpWrite(Tab[0]),
            //    [](double& valParam)
            //    {
            //    }
            //    );


/*
            for(int idx = 0 ; idx < NbTasks ; ++idx){
                My_Runtime7.task(SpWrite(Tab[idx]),
                             SpReadArray(Tab,SpArrayView(NbTasks).removeItem(idx)),
                                     
                [idx](double& valParam, const SpArrayAccessor<const double>& valArray) -> bool
                {
                    {
                            if(idx == 3){
                                valParam += 1;
                            }
                            if(idx == 5){
                                valParam += 10;
                            }
                            usleep(1000);
                            cout<<"valParam="<<valParam<<"\n";
                        return (idx == 3 || idx == 5); 
                    } 
                }
                ).setTaskName("Runtime Array Process"); //END runtime
            } //END FOR
*/
            

            for(int idxTask = NbTasks ; idxTask < 2*NbTasks ; ++idxTask){
                
                
                My_Runtime7.waitRemain(idxTask); //mutexFinishedTasks
                
            }


            //for(int idxTask = 0 ; idxTask < NbTasks ; ++idxTask){
            //    My_Runtime5.waitRemain(idxTask); //mutexFinishedTasks
            //}

            
            for(int idxTaskPromise = 0 ; idxTaskPromise < NbTasks ; ++idxTaskPromise)
            {
                promises[idxTaskPromise].set_value(0);

                

                //Une exception est levée s'il n'y a pas d'état partagé ou si 
                //l'état partagé stocke déjà une valeur ou une exception.

                //usleep(2000);
                for(int idxTask = NbTasks-idxTaskPromise-1 ; idxTask < 2*NbTasks ; ++idxTask)
                {
                    My_Runtime7.waitRemain(idxTask);
                }
            }

            for (const double& value : Tab) { cout << value << "  "; }; cout << "\n";
            

            My_Runtime7.waitAllTasks();

            My_Runtime7.generateDot("Runtime7.dot",true);
            My_Runtime7.generateTrace("Runtime7.svg");
        }



if (1==1)
    {
         
    }


}



 
#pragma GCC diagnostic pop

