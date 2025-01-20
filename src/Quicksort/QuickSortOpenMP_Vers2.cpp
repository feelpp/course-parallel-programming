// C++ program to implement the Quick Sort
// using OMI
#include <bits/stdc++.h>
#include <omp.h>

#include <chrono>
#include <math.h>

using namespace std;

// Function to swap two numbers a and b
void swap(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}

// Function to perform the partitioning
// of array arr[]
int partition(int arr[], int start, int end)
{
	// Declaration
	int pivot = arr[end];
	int i = (start - 1);

	// Rearranging the array
	for (int j = start; j <= end - 1; j++) {
		if (arr[j] < pivot) {
			i++;
			swap(&arr[i], &arr[j]);
		}
	}
	swap(&arr[i + 1], &arr[end]);

	// Returning the respective index
	return (i + 1);
}

// Function to perform QuickSort Algorithm
// using openmp
void quicksort(int arr[], int start, int end)
{
	// Declaration
	int index;

	if (start < end) {

		// Getting the index of pivot
		// by partitioning
		index = partition(arr, start, end);

// Parallel sections
#pragma omp parallel sections
		{
#pragma omp section
			{
				// Evaluating the left half
				quicksort(arr, start, index - 1);
			}
#pragma omp section
			{
				// Evaluating the right half
				quicksort(arr, index + 1, end);
			}
		}
	}
}




// Driver Code
int main()
{
    FILE* file = NULL;
	file = fopen("Result_QuickSort_OpenMP_Ver2.csv", "w");
	fprintf(file, "NbElements,TimeDuration,NbProcs\n");
	fclose(file);	
	
    int NbProcs=1;

    for( int k0 = 1; k0 < 100; k0++ )
    {
		file = fopen("Result_QuickSort_OpenMP_Vers2.csv", "a");
		// Declaration
		//int N=50000;
		int N=pow(2,k0);
		std::cout<<"\n";
		std::cout<<"Quicksort "<<N<<" ints on "<<NbProcs<<"\n";


		// Declaration of array
		int arr[N];


		double start_time, run_time;

		for( int i = 0; i < N; i++ )
		{
		arr[i] = rand() % 50 +1;
		// printf("%d\n", arr[i]);
		}

		// Calling quicksort having parallel
		// code implementation

		start_time = omp_get_wtime();
		auto start_time2= std::chrono::steady_clock::now();

		quicksort(arr, 0, N - 1);

		run_time = omp_get_wtime() - start_time;
		printf("\n Execution time was %lf seconds\n ",run_time);

		auto run_time2=std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::steady_clock::now()-start_time2);

		std::cout<<"Execution Time in ms since start :"<<run_time2.count()<<"\n";
		std::cout<<"\n";

		fprintf(file, "%d,%ld,%d\n",N,run_time2.count(),NbProcs);
		fclose(file);
	}


	return 0;
}


