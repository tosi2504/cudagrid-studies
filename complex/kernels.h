#pragma once

#include <assert.h>
#include "cudatools.h"

namespace copy {
    template<class T>
    __global__ void naive(T * d_out, const T * d_in, unsigned N) {
        const unsigned i_start = threadIdx.x + blockDim.x*blockIdx.x;
        const unsigned stride = blockDim.x * gridDim.x;
        for (unsigned i = i_start; i < N*N; i += stride) {
            d_out[i] = d_in[i];
        }
    }

    template<class T>
    double calcBW(unsigned N, unsigned reps, unsigned long micros) {
        return sizeof(T) * (long)2*N*N * (long)reps / (double)micros;
    }
}

namespace transpose {
    template<class T>
    __global__ void naive(T * d_out, const T * d_in, unsigned N) {
        // get global index
        const unsigned i_start = threadIdx.x + blockDim.x*blockIdx.x;
        const unsigned stride = blockDim.x * gridDim.x;
        for (unsigned i = i_start; i < N*N; i+= stride) {
            const unsigned x = i%N;
            const unsigned y = i/N;
            d_out[x*N + y] = d_in[y*N + x];
        }
    }

    // assert(numThreads == TN*TN)
    template<class T, unsigned TN>
    __global__ void shmem(T * d_out, const T * d_in, unsigned N) {
        const unsigned numTilesPerSide = N / TN;
        const unsigned xx = (blockIdx.x%numTilesPerSide) * TN;
        const unsigned yy = (blockIdx.x/numTilesPerSide) * TN;
        const unsigned x  = threadIdx.x % TN;
        const unsigned y  = threadIdx.x / TN;

        // shmem
        __shared__ T tile[TN*TN];
        tile[y*TN + x] = d_in[(yy+y)*N + (xx + x)]; // coalesced gmem access
        __syncthreads();
        d_out[(xx+y)*N + (yy+x)] = tile[x*TN + y];  // coalesced gmem access, I thought lol wtf
    }

    template<class T>
    double calcBW(unsigned N, unsigned reps, unsigned long micros) {
        return sizeof(T) * (long)2*N*N * (long)reps / (double)micros;
    }

    template<class T>
    void test(const T * h_out, const T * h_in, unsigned N) {
        for (unsigned x = 0; x < N; x++) {
            for (unsigned y = 0; y < N; y++) {
                if (not isEqual(h_out[y*N+x], h_in[x*N+y])) {
                    std::cout << "Test transpose: Error at x: " << x << ", y: " << y << std::endl;
                    return;
                }
            }
        }
        std::cout << "Test transpose: Test successful" << std::endl;
    }
}

namespace repeatedMatmul {
    template<class T, unsigned N, unsigned iterations>
    __global__ void naive(T * d_Ys, const T * d_As, const T * d_Xs) {
        const unsigned N_ceil32 = CEIL(N, 32);
        const unsigned numThreads = blockDim.x;
        assert(numThreads >= N_ceil32);

        const unsigned iMatrix = blockIdx.x;
        
        // shift to correct matrix in batch
        const T * A = d_As + N*N*iMatrix;
        const T * X = d_Xs + N*N*iMatrix;
        T * Y = d_Ys + N*N*iMatrix;

        //shmem (everything row major?)
        extern __shared__ T shmem[];
        T * sA = shmem;
        T * sX = shmem + N*N;
        // T * sY = shmem + N*N + N*N;

        // fill sA and sX
        for (unsigned i = threadIdx.x; i < N*N; i+=numThreads) {
            const unsigned col = i % N;
            const unsigned row = i / N;
            sA[i] = A[row*N + col];
            sX[i] = X[row*N + col];
        }
        __syncthreads();

        // iterations loop
        for (unsigned _ = 0; _ < iterations; _++) {
            const unsigned rowStride = numThreads / N_ceil32;
            const unsigned dRow = threadIdx.x / N_ceil32;
            const unsigned iCol = threadIdx.x % N_ceil32;
            if (iCol >= N) return; // access guard
            for (unsigned iiRow = 0; iiRow < N; iiRow+=rowStride) {
                const unsigned iRow = iiRow + dRow;
                if (iRow >= N) break; // access guard
            
                T dotRes = 0;
                for (unsigned k = 0; k < N; k++) {
                    dotRes += sA[iRow*N + k] * sX[k*N + iCol];
                }
                Y[iRow*N + iCol] = dotRes;
            }
        }
    }

    template<class T, unsigned N>
    constexpr unsigned calcShmemSize() {
        return sizeof(T) * N*N * 2;
    }

    template<class T, unsigned N>
    double calcBW(unsigned numMatrices, unsigned reps, unsigned long micros) {
        return sizeof(T) * (long)reps * (long)(3*N*N) * (long)(numMatrices) / (double)micros;
    }

    template<class T, unsigned N>
    void test(
        const MatrixBatch<T,N> & Ys, 
        const MatrixBatch<T,N> & As, 
        const MatrixBatch<T,N> & Xs
    ) {
        const unsigned numMatrices = Ys.numMatrices;
        assert(numMatrices == Xs.numMatrices);
        assert(numMatrices == As.numMatrices);
        for (unsigned iMatrix = 0; iMatrix < numMatrices; iMatrix++) {
            T * A = As.h_data + iMatrix*N*N;
            T * X = Xs.h_data + iMatrix*N*N;
            T * Y = Ys.h_data + iMatrix*N*N;
            for (unsigned row = 0; row < N; row++) {
                for (unsigned col = 0; col < N; col++) {
                    T tempRes = 0;
                    for (unsigned k = 0; k < N; k++) {
                        tempRes += A[row*N + k] * X[k*N + col];
                    }
                    if (not isEqual(Y[row*N + col], tempRes)) {
                        std::cout << "Test: Mismatch at iMatrix: " << iMatrix;
                        std::cout << ", row: " << row << ", col:" << col;
                        std::cout << "  --> You suck! *look of disgust and distain*" << std::endl;
                        return;
                    }
                }
            }
        }
        std::cout << "Test: All tests successful, no mismatches! *smirk* *blows kiss*" << std::endl;
    }
}
