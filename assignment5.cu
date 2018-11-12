
#include<iostream>
#include<vector>
#include<cmath>
#include<ctime>
#include<string>

using namespace std;

#include<cuda.h>

const long MAXDIM = 10;
const double RMIN = 2.0;
const double RMAX = 8.0;


//Global kernel code that runs on the device
__global__ void count_in(){

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

void addone(long *index, long ndim,long base, long i)
/* Add one to a digital counter of given base. When one digit hits
   maximum, it is necessary to carry one over into the next
   column. This is done recursively here. */
{
  long newv = index[i] + 1;
  if (newv >= base) {
    index[i] = 0;
    if (i < ndim - 1) addone(index, ndim, base, i+1);
  }
  else {
    index[i] = newv;
  }
}

void count_in_seq(long *idx, bool *sphereList, long &count, int pos,long ndim, long halfb, double rsquare, long base){ 
	double rtestsq = 0;
	bool inSphere = false; 

	for(long k = 0; k < ndim; ++k){
		double xk = idx[k] - halfb;
		rtestsq += xk * xk;
	}

	if(rtestsq< rsquare) inSphere = true;
	addone(idx, ndim, base, 0);

	if(inSphere){
  		count++;
  	}
}


//========================================
// Main Function
//========================================

int main(int argc, char **argv){ 
  // You can make this larger if you want
  const long ntrials = 3;

  for (long n = 0; n < ntrials; ++n) {

    // Get a random value for the hypersphere radius between the two limits
    const double radius = drand48() * (RMAX - RMIN) + RMIN;

    // Get a random value for the number of dimensions between 1 and
    // MAXDIM inclusive
    const long  ndim = lrand48() % (MAXDIM - 1) + 1;
    std::cout << "Trial Number " << n << " Radius " << radius << " Dimensions " << ndim << " ... " << std::endl;

    long *count = 0;

    const long halfb = static_cast<long>(floor(radius));
  	const long base = 2 * halfb + 1;
  	const double rsquare = radius * radius;
  	const long ntotal = powlong(base, ndim);

  	long *index = new long[ndim];
  	bool *isInSphereList = new bool[ntotal];

  	for(int i = 0; i < ndim; i++){
  		index[i] = 0;
  	}
  	for(int i = 0; i < ntotal; i++){
  		isInSphereList[i] = false;
  	}

    //CUDA part
    //=======================================================
  	//we need to split the problem into each pixel being an integer point

  	long *dev_index; 
  	bool *dev_isInSphereList;
  	long *dev_count;
  	long *dev_ndim; 
  	long *dev_halfb; 
  	double *dev_rsquare;
  	long *dev_base;

  	cudaError_t err = cudaMalloc((void**)&dev_index, sizeof(long)*ndim);
  	err_check(err, "index malloc");
  	err = cudaMalloc((void**)&dev_isInSphereList, sizeof(bool)*ntotal);
  	err_check(err, "isInSphereList malloc");
  	err = cudaMalloc((void**)&dev_count, sizeof(long));
  	err_check(err, "count malloc");
  	err = cudaMalloc((void**)&dev_ndim, sizeof(long));
  	err_check(err, "ndim malloc");
  	err = cudaMalloc((void**)&dev_halfb, sizeof(long));
  	err_check(err, "halfb malloc");
  	err = cudaMalloc((void**)&dev_rsquare, sizeof(double));
  	err_check(err, "rsquare malloc");
  	err = cudaMalloc((void**)&dev_base, sizeof(long));
  	err_check(err, "base malloc");

  	err = cudaMemcpy(dev_index, index, sizeof(long)*ndim, cudaMemcpyHostToDevice);
  	err_check(err, "index cpy");
  	err = cudaMemcpy(dev_isInSphereList, isInSphereList, sizeof(bool)*ntotal, cudaMemcpyHostToDevice);
  	err_check(err, "isinspherelist cpy");
  	err = cudaMemcpy(dev_count, &count, sizeof(long), cudaMemcpyHostToDevice);
  	err_check(err, "count cpy");
  	err = cudaMemcpy(dev_ndim, &ndim, sizeof(long), cudaMemcpyHostToDevice);
  	err_check(err, "ndim cpy");
  	err = cudaMemcpy(dev_halfb, &halfb, sizeof(long), cudaMemcpyHostToDevice);
  	err_check(err, "halfb cpy");
  	err = cudaMemcpy(dev_rsquare, &rsquare, sizeof(double), cudaMemcpyHostToDevice);
  	err_check(err, "rsquare cpy");
  	err = cudaMemcpy(dev_base, &base, sizeof(long),cudaMemcpyHostToDevice);
  	err_check(err, "base cpy");

  	int threadsPerBlock = 1024; 
  	int numBlocks = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

  	cout << "Number of Threads Per Block: " << threadsPerBlock << endl;
  	cout << "Number of Blocks Per Grid: " << numBlocks << endl;
  	cout << "N Total: " << ntotal << endl;

  	count_in<<<threadsPerBlock, numBlocks>>>();

  	err = cudaMemcpy(count, dev_count, sizeof(long), cudaMemcpyDeviceToHost);
  	err_check(err, "count cpy to host");


  	//sequential
  	for(long n = 0; n < ntotal; ++n){
  		count_in_seq(index, isInSphereList, count, n, ndim , halfb, rsquare, base);
  	}

    std::cout << " -> " << "count" << " " << count << std::endl;

    free(index);
  }
}