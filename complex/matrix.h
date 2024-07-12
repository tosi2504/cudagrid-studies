#pragma once

#include "cudatools.h"
#include "datatypes.h"
#include <random>
#include <cassert>

template<class T>
struct Matrix {
	const int N;
	T * h_data; 
	T * d_data;

	Matrix(int N):
		N(N)
	{
		h_data = new T[N*N];
		CCE(cudaMalloc(&d_data, sizeof(T)*N*N));
	}
	~Matrix() {
		delete[] h_data;
		CCE(cudaFree(d_data));
	}
	T get(uint i, uint j) const {
		return h_data[i*N + j];
	}
	T & get(uint i, uint j) {
		return h_data[i*N + j];
	}
	void upload() {
		CCE(cudaMemcpy(d_data, h_data, sizeof(T)*N*N, cudaMemcpyHostToDevice));
	}
	void download() {
		CCE(cudaMemcpy(h_data, d_data, sizeof(T)*N*N, cudaMemcpyDeviceToHost));
	}
	void fill_random(unsigned seed) {
		std::default_random_engine rng(seed);
		std::uniform_real_distribution<float> dist(0.0, 1.0);
		for (int i = 0; i < N*N; i++) h_data[i] = dist(rng);
	}
	void fill_ones() {
		for (int i = 0; i < N*N; i++) h_data[i] = 1;
	}
	void fill_zeros() {
		for (int i = 0; i < N*N; i++) h_data[i] = 0;
	}
	static T matmul(const Matrix & A, const Matrix & B, uint i, uint j) {
		assert(A.N == B.N);
		uint N = A.N;
		T res = 0;
		for (uint k = 0; k < N; ++k) {
			res += A.h_data[i*N + k] * B.h_data[k*N + j];
		}
		return res;
	} 
}; 

template<class T>
void checkMatmul(const Matrix<T> & C, const Matrix<T> & A, const Matrix<T> & B) {
	C.download();
	uint N = C.N;
	std::cout << "Start errorcheck:" << std::endl;
	bool doBreak = false;
	for (uint i = 0; i < N; i++) {
		for (uint j = 0; j < N; j++) {
			if (Matrix<T>::matmul(A, B, i, j) != C.get(i,j)) {
				std::cout << "Error detected: " << std::endl;
				std::cout << "i: " << i << " j: " << j << std::endl;
				doBreak = true;
				break;
			}
		}
		if (doBreak) break;
	}
	if (not doBreak) std::cout << "No errors" << std::endl;
}

template<class T, unsigned N>
struct MatrixBatch {
    T * d_data;
    T * h_data;
    const unsigned numMatrices;

    MatrixBatch(unsigned numMatrices):
        numMatrices(numMatrices) 
    {
        h_data = new T[N*N*numMatrices];
        CCE(  cudaMalloc(&d_data, sizeof(T)*numMatrices*N*N)  );
    }
    ~MatrixBatch() {
        delete [] h_data;
        CCE(  cudaFree(d_data)  );
    }
    void upload() const {
        CCE(  cudaMemcpy(d_data, h_data, N*N*numMatrices*sizeof(T), cudaMemcpyHostToDevice)  );
    }
    void download() const {
        CCE(  cudaMemcpy(h_data, d_data, N*N*numMatrices*sizeof(T), cudaMemcpyDeviceToHost)  );
    }
    void fill_random(unsigned seed) const {
        for (unsigned i = 0; i < N*N*numMatrices; i++) {
            std::default_random_engine rng(seed);
            uniform_distribution<T> dist(0, 1);
            h_data[i] = dist(rng);
        }
    }
    void fill_ones() const {
        for (unsigned i = 0; i<N*N*numMatrices; i++) h_data[i] = 1;
    }
};

