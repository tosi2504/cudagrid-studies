#include <iostream>
#include <chrono>
#include <random>
#include "errorcheck.h"
#include "benchmark.h"


template<class T>
__global__ void ker_copy(T * const dst, const T * const src, const unsigned long len) {
	const unsigned gIdx = blockIdx.x*blockDim.x + threadIdx.x;
	if (gIdx < len) dst[gIdx] = src[gIdx];
}

template<class T, unsigned blocksize>
inline void copy(T * const dst, const T * const src, const unsigned long len) {
	const unsigned numBlocks = (len + blocksize - 1)/blocksize;
	ker_copy <T> <<<numBlocks, blocksize>>> (dst, src, len);
	CCE(  cudaDeviceSynchronize()  );
}

template<class T>
__global__ void ker_fill(T * const arr, const unsigned long len) {
	const unsigned gIdx = blockIdx.x*blockDim.x + threadIdx.x;
	if (gIdx < len) arr[gIdx] = gIdx;
}

using T = float;
constexpr unsigned long N = 500000000;
constexpr unsigned blocksize = 256; 
int main () {
	// allocate arrays
	T * a, * b;
	CCE(  cudaMalloc(&a, sizeof(T)*N)  );
	CCE(  cudaMalloc(&b, sizeof(T)*N)  );

	const unsigned numBlocks = (N + blocksize - 1)/blocksize;
	ker_fill<<<numBlocks, blocksize>>>(a, N);

	double resTime = 0;
	auto func = copy<T,blocksize>;
	BENCHMARK(resTime, 1000, func, b, a, N);
	std::cout << "Resulting time average in us : " << resTime << std::endl;
	std::cout << "Bandwidth in GB/s : " << N*sizeof(T)*2/(resTime*1000) << std::endl;
}
