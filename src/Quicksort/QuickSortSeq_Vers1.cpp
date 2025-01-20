// C++ code to implement quicksort

#include <bits/stdc++.h>
#include <chrono>
#include <math.h>


using namespace std;

// This function takes last element as pivot,
// places the pivot element at its correct position
// in sorted array, and places all smaller to left
// of pivot and all greater elements to right of pivot
int partition(int arr[], int low, int high)
{
	// Choosing the pivot
	int pivot = arr[high];

	// Index of smaller element and indicates
	// the right position of pivot found so far
	int i = (low - 1);

	for (int j = low; j <= high - 1; j++) {

		// If current element is smaller than the pivot
		if (arr[j] < pivot) {

			// Increment index of smaller element
			i++;
			swap(arr[i], arr[j]);
		}
	}
	swap(arr[i + 1], arr[high]);
	return (i + 1);
}

// The main function that implements QuickSort
// arr[] --> Array to be sorted,
// low --> Starting index,
// high --> Ending index
void quickSort(int arr[], int low, int high)
{
	if (low < high) {

		// pi is partitioning index, arr[p]
		// is now at right place
		int pi = partition(arr, low, high);

		// Separately sort elements before
		// partition and after partition
		quickSort(arr, low, pi - 1);
		quickSort(arr, pi + 1, high);
	}
}

// Driver Code
int main()
{
    FILE* file = NULL;
	file = fopen("Result_QuickSort_Seq.csv", "w");
	fprintf(file, "NbElements,TimeDuration,NbProcs\n");
	fclose(file);	
	
    int NbProcs=1;

	for( int k0 = 1; k0 < 100; k0++ )
    {
		file = fopen("Result_QuickSort_Seq.csv", "a");

		int Nn=pow(2,k0);
		std::cout<<"\n";
		std::cout<<"Quicksort "<<Nn<<" ints on "<<NbProcs<<"\n";


		// Declaration of array
		int arr[Nn];

		for( int i = 0; i < Nn; i++ )
		{
			arr[i] = rand() % 50 +1;
		}


		int N = sizeof(arr) / sizeof(arr[0]);

		auto start_time2= std::chrono::steady_clock::now();
		// Function call
		quickSort(arr, 0, N - 1);

		auto run_time2=std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::steady_clock::now()-start_time2);

		std::cout<<"Execution Time in ms since start :"<<run_time2.count()<<"\n";
		std::cout<<"\n";

		fprintf(file, "%d,%ld,%d\n",Nn,run_time2.count(),NbProcs);
		fclose(file);

	}

	return 0;
}
