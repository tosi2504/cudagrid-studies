#include "cudatools.h"
#include "sgemm.h"
#include "matrix.h"
#include "datatypes.h"
#include <chrono>
#include <iostream>
#include <cuda/std/complex>

decltype(std::chrono::high_resolution_clock::now()) start;
decltype(std::chrono::high_resolution_clock::now()) stop;

inline void timing_start() {
	start = std::chrono::high_resolution_clock::now();
}

inline long timing_stop() {
	stop = std::chrono::high_resolution_clock::now();
	auto delta_t = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); // fix type issue
	return delta_t.count();
}


constexpr unsigned long N = 2048;
constexpr uint BN = 128;
static_assert(N%BN == 0);
constexpr uint BK = 8;
static_assert(N%BK == 0);
constexpr uint TN = 8;
using T = complexF;
// using T = float;
int main () {
	Matrix<T> A(N), B(N), C(N);
	A.fill_random(0);
	A.upload();
	B.fill_random(1);
	B.upload();
	C.fill_random(2);
	C.upload();


	dim3 gridDim(N/BN, N/BN, 1);
	dim3 blockDim((BN*BN)/(TN*TN), 1, 1);
	std::cout << "USING " << (BN*BN)/(TN*TN) << " THREADS PER BLOCK" << std::endl;

	T alpha = 1;
	T beta = 0;
	unsigned reps = 50;
	std::cout << "STARTED TIMING" << std::endl;
	timing_start();
	for (unsigned rep = 0; rep < reps; rep++) {
		sgemm_2D_blocktiling<T, N, BN, BK, TN> <<< gridDim , blockDim >>> (C.d_data, A.d_data, B.d_data, alpha, beta);
		// sgemm_2D_blocktiling_complex<T, N, BN, BK, TN> <<< gridDim , blockDim >>> (C.d_data, A.d_data, B.d_data, alpha, beta);
		CLCE();
		CCE(cudaDeviceSynchronize());
	}
	unsigned microsecs = timing_stop();
	std::cout << "BANDWIDTH (MByte/s): " << sizeof(T)*reps*(4*N*N)/(double)microsecs << std::endl;
    const long factor = (isComplex<T>::value) ? 4 : 1;
	std::cout << "ARITHETICS (GFLOPS/s) (TODO): " << reps*(factor*2*N*N*N + N*N)/((double)microsecs*1000) << std::endl;
	CLCE();

	// check for correctness
	C.download();
	uint i = N-1;
	uint j = N-1;
	std::cout << "i: " << i << " j: " << j << std::endl;
	// std::cout << Matrix::matmul(A, B, i, j) << " <---> " << C.get(i,j) << std::endl;
	// checkMatmul(C, A, B);
}
