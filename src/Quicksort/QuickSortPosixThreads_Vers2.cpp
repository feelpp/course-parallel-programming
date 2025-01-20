// C++ program to implement the Quick Sort
// using POSIX Thread
#include <bits/stdc++.h>
#include <pthread.h>
#include <ctime>
#include <chrono>
#include <math.h>

using namespace std;

// Structure
struct data_set {
	int start_index;
	int end_index;
	int* data;
};

// Function to perform swap operations
void swap(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}

// Partition function for making
// partition in array
int partition(int arr[], int left_index,
			int right_index)
{
	// Declaration and initialization
	// choosing pivot element form which
	// we make partition

	// Here pivot is last element of
	// the array
	int pivot = arr[right_index];
	int i = left_index - 1;

	// Making array as per requirement
	// arranging element smaller than
	// pivot on left side and larger
	// then pivot on right side
	for (int j = left_index;
		j <= right_index - 1; j++) {

		if (arr[j] < pivot) {
			i++;
			swap(&arr[i], &arr[j]);
		}
	}

	swap(&arr[i + 1], &arr[right_index]);

	// Returning the partition index
	return i + 1;
}

// Quicksort Function for sorting
// array
void* quick_sort(void* data)
{
	// Retrieving back the data sent
	// from thread
	struct data_set* info = (struct data_set*)data;

	// Declaration of left index
	int left_index, right_index, index;

	// Initialization of left and
	// right index
	left_index = info->start_index;
	right_index = info->end_index;

	// Recursive call of quick_sort
	// function
	if (left_index < right_index) {

		// Declaration of pthread and
		// pthread attribute type object
		pthread_attr_t attr;
		pthread_t first_thread;
		pthread_t second_thread;

		// Making two pointers of type
		// data_set for making again
		// call form thread
		struct data_set* info1 = new data_set;
		struct data_set* info2 = new data_set;

		// Their initialization
		info1->data = info->data;
		info2->data = info->data;

		// Initialize of pthread attribute
		pthread_attr_init(&attr);

		// For setting the set detach
		// state of attribute
		pthread_attr_setdetachstate(
			&attr, PTHREAD_CREATE_JOINABLE);

		// Partition the array for any
		// recursive call
		index = partition(info->data,
						left_index,
						right_index);

		info1->start_index = left_index;
		info1->end_index = index - 1;

		// Create pthread type object and
		// printing the error if any
		if (pthread_create(&first_thread,
						&attr, quick_sort,
						info1)) {
			cout << "Error in creating thread "
				<< endl;

			// Exiting in case of not
			// creation of thread
			exit(-1);
		}

		info2->start_index = index + 1;
		info2->end_index = right_index;

		// Creating pthread type object
		// and print the error
		if (pthread_create(&second_thread,
						&attr, quick_sort,
						info2)) {
			cout << "Error in creating thread "
				<< endl;

			// Exiting in case of not
			// creation of thread
			exit(-1);
		}

		// Joining the threads
		pthread_join(first_thread, NULL);
		pthread_join(second_thread, NULL);
	}

	return NULL;
}

// Driver Code
int main()
{
    int NbProcs=1;
    struct data_set* info = new data_set;
    FILE* file = NULL;
	file = fopen("Result_QuickSort_PosixThreads_Vers2.csv", "w");
	fprintf(file, "NbElements,TimeDuration,NbProcs\n");
	fclose(file);	

	

	for( int k0 = 1; k0 < 100; k0++ )
    {
		file = fopen("Result_QuickSort_PosixThreads_Vers2.csv", "a");
		// Declaration
		//int N=50;
		//int N=pow(10,k0);
		int N=pow(2,k0);

		std::cout<<"\n";
		std::cout<<"Quicksort "<<N<<" ints on "<<NbProcs<<"\n";

		// Declaration of array
		int A[N];

		double start_time, run_time;

		for( int i = 0; i < N; i++ )
		{
		A[i] = rand() % 50 +1;
		// printf("%d\n", arr[i]);
		}

		start_time = time(NULL);

		auto start_time2= std::chrono::steady_clock::now();



		// Initialize of structure of
		// data_set type
		info->data = A;
		info->start_index = 0;
		info->end_index = N - 1;

		// Declaration of pthread object
		pthread_t thread_id;

		// Creating and pthread object and
		// printing the array of any
		if (pthread_create(&thread_id, NULL,
						quick_sort,
						info)) {
			cout << "Error in creating thread"
				<< endl;

			// Exit in case of error
			exit(-1);
		}

		// Joining the pthread object
		int r1 = pthread_join(thread_id, NULL);

		// Printing the array if any in case
		// of joining
		if (r1) {
			cout << "Error in joining thread"
				<< endl;

			// Exiting in case of error
			exit(-1);
		}

		run_time = time(NULL) - start_time;

		auto run_time2=std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::steady_clock::now()-start_time2);

	
		std::cout<<"Execution Time in ms since start :"<<run_time2.count()<<"\n";
		std::cout<<"\n";


		// Exiting from pthread programming
		

	
		fprintf(file, "%d,%ld,%d\n",N,run_time2.count(),NbProcs);
		fclose(file);
	}
	
		
pthread_exit(NULL);
	return 0;
}


