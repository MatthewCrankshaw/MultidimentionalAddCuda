
#include<iostream>
#include<vector>
#include<cmath>
#include<ctime>
#include<string>

using namespace std;

#include<cuda.h>

#define ULL unsigned long long

const long MAXDIM = 10;
const double RMIN = 2.0;
const double RMAX = 8.0;




//Global kernel code that runs on the device
__global__ void count_in(unsigned long long*dev_count, long dev_ntotal,long dev_ndim, long dev_halfb, double dev_rsquare, long dev_base){
	
	int pos = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(pos > dev_ntotal) return;

	double rtestsq = 0;
	long idx = 0;

	long index[MAXDIM];

	for (long i = 0; i < dev_ndim; ++i){
		index[i] = 0;
	}
  	
  	while (pos != 0) {
    	long rem = pos % dev_base;
    	pos = pos / dev_base;
    	index[idx] = rem;
    	++idx;
  	}

	for(long k = 0; k < dev_ndim; ++k){
		double xk = index[k] - dev_halfb;
		rtestsq += xk * xk;
	}

	if(rtestsq < dev_rsquare){
		//This needs to be an atomic add
		atomicAdd(dev_count, 1);
	}
}

void err_check(cudaError_t err, char* text){
	if(err != cudaSuccess){
		std::cout << "============================Cuda Error: " << cudaGetErrorString(cudaGetLastError()) << " in " << text << std::endl;
	}
}


//calculate the power using long integers
long powlong(long n, long k)
/* Evaluate n**k where both are long integers */
{
  long p = 1;
  for (long i = 0; i < k; ++i) p *= n;
  return p;
}
					
void count_in_seq(ULL &count,int thread, int block,long ndim, long halfb, double rsquare, long base){ 

	int num = (block * 1024) + thread;
	double rtestsq = 0;
	long idx = 0;

	long index[MAXDIM];

	for (long i = 0; i < ndim; ++i){
		index[i] = 0;
	}
  	
  	while (num != 0) {
    	long rem = num % base;
    	num = num / base;
    	index[idx] = rem;
    	++idx;
  	}

	for(long k = 0; k < ndim; ++k){
		double xk = index[k] - halfb;
		rtestsq += xk * xk;
	}

	if(rtestsq< rsquare){

		//This needs to be an atomic add
		count++;
	}
}


//========================================
// Main Function
//========================================

int main(int argc, char **argv){ 
  // You can make this larger if you want
  const long ntrials = 10;

  for (long n = 0; n < ntrials; ++n) {

    // Get a random value for the hypersphere radius between the two limits
    const double radius = drand48() * (RMAX - RMIN) + RMIN;

    // Get a random value for the number of dimensions between 1 and
    // MAXDIM inclusive
    const long  ndim = lrand48() % (MAXDIM - 1) + 1;
    std::cout << "Trial Number " << n << " Radius " << radius << " Dimensions " << ndim << " ... " << std::endl;

    ULL count = 0;

    const long halfb = static_cast<long>(floor(radius));
  	const long base = 2 * halfb + 1;
  	const double rsquare = radius * radius;
  	const long ntotal = powlong(base, ndim);


    //CUDA part
    //=======================================================
  	//we need to split the problem into each pixel being an integer point
  	cout << "count: " << count << endl;
  	ULL *dev_count;

  	cudaMalloc((void**)&dev_count, sizeof(unsigned long long));

  	cudaMemcpy(dev_count, &count, sizeof(unsigned long long), cudaMemcpyHostToDevice);


  	int threadsPerBlock = 1024; 
  	int numBlocks = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

  	cout << "Number of Threads Per Block: " << threadsPerBlock << endl;
  	cout << "Number of Blocks Per Grid: " << numBlocks << endl;
  	cout << "N Total: " << ntotal << endl;

  	count_in<<<threadsPerBlock, numBlocks>>>(dev_count, ntotal, ndim, halfb, rsquare, base);

  	cudaMemcpy(&count, dev_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);


  	// //sequential
  	// int thread = 0;
  	// int block = 0;
  	// for(long n = 0; n < ntotal; ++n){
  	// 	count_in_seq(count, thread, block, ndim , halfb, rsquare, base);
  	// 	if(thread+1 > 1024){
  	// 		thread = 0;
  	// 		block+=1;
  	// 	}else{
  	// 		thread++;
  	// 	}
  	// }

    std::cout << " -> " << "count" << " " << count << "\n" << std::endl;

  }
}